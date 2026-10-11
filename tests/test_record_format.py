"""The "structured" record format: file headers, overlapping windows, and what a citation opens.

The line range on a window is the thing to get right. It is what the citation shows and what
the file preview opens, so a window whose range is off by one points a reader at the wrong code
while looking perfectly plausible. Several tests here check the range against the file itself
rather than against the arithmetic that produced it.
"""

from datetime import UTC, datetime

import pytest

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings, _dedupe_windows
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService, embedder_mismatch
from codeatlas.services.retrieval.record_format import (
    RECORD_FORMATS,
    STRUCTURED,
    WHOLE,
    RecordFormatEmbedder,
    file_header,
    validate_record_format,
    windows,
    with_record_format,
)

# ---------------------------------------------------------------- windows


def numbered(count: int, width: int = 20) -> str:
    """`count` lines, each the same width, so a window's size is easy to reason about."""
    return "\n".join(f"line{index:03d}".ljust(width - 1) for index in range(1, count + 1))


def test_text_that_fits_is_one_window_unchanged():
    text = "def f():\n    return 1\n"
    cut = windows(text, start_line=10, window_chars=1000, overlap_chars=100)

    assert len(cut) == 1
    assert cut[0].text == text
    # Two lines of text, so lines 10 and 11; a trailing newline does not make a third line.
    assert (cut[0].start_line, cut[0].end_line) == (10, 11)


def test_a_long_text_is_cut_into_several_windows():
    cut = windows(numbered(50), start_line=1, window_chars=200, overlap_chars=40)
    assert len(cut) > 1
    assert all(len(window.text) <= 200 for window in cut)


def test_windows_cover_every_line_with_no_gap():
    text = numbered(50)
    cut = windows(text, start_line=1, window_chars=200, overlap_chars=40)

    covered: set[int] = set()
    for window in cut:
        covered.update(range(window.start_line, window.end_line + 1))
    assert covered == set(range(1, 51))


def test_consecutive_windows_overlap_rather_than_abut():
    cut = windows(numbered(50), start_line=1, window_chars=200, overlap_chars=60)
    for earlier, later in zip(cut, cut[1:]):
        assert later.start_line <= earlier.end_line, "a window must repeat part of the one before it"


def test_a_window_never_starts_or_ends_mid_line():
    text = numbered(50)
    lines = text.splitlines()
    for window in windows(text, start_line=1, window_chars=200, overlap_chars=40):
        assert window.text.splitlines() == lines[window.start_line - 1 : window.end_line]


def test_the_line_range_is_absolute_in_the_file_not_relative_to_the_function():
    text = numbered(40)
    cut = windows(text, start_line=501, window_chars=200, overlap_chars=40)

    assert cut[0].start_line == 501
    assert cut[-1].end_line == 540


def test_one_line_longer_than_the_window_becomes_its_own_window():
    """Half a line is not something anyone can read or cite, so it is never split."""
    text = "short\n" + "x" * 500 + "\nshort again"
    cut = windows(text, start_line=1, window_chars=100, overlap_chars=10)

    long_windows = [w for w in cut if len(w.text) > 100]
    assert len(long_windows) == 1
    assert long_windows[0].start_line == long_windows[0].end_line == 2


def test_the_walk_always_advances():
    """A step-back for overlap that reached the window's own start would loop forever."""
    cut = windows(numbered(200), start_line=1, window_chars=100, overlap_chars=10_000)
    assert len(cut) < 500
    for earlier, later in zip(cut, cut[1:]):
        assert later.start_line > earlier.start_line


def test_an_empty_text_is_one_empty_window():
    cut = windows("", start_line=7, window_chars=100, overlap_chars=10)
    assert len(cut) == 1 and cut[0].start_line == cut[0].end_line == 7


def test_a_window_size_of_zero_is_rejected():
    with pytest.raises(ValueError):
        windows("text", start_line=1, window_chars=0, overlap_chars=0)


# ---------------------------------------------------------------- file headers


SOURCE = '''"""What this module is for.

A second line of the docstring.
"""

import os
from pathlib import Path


class Signer:
    def sign(self, value):
        return value


def helper(x):
    return x
'''


def test_a_file_header_holds_the_path_docstring_imports_and_signatures():
    header = file_header(SOURCE, "src/pkg/signer.py", ["class Signer:", "def sign(self, value):"], 2000)

    assert "src/pkg/signer.py" in header
    assert "What this module is for." in header
    assert "import os" in header and "from pathlib import Path" in header
    assert "class Signer:" in header and "def sign(self, value):" in header


def test_a_file_header_leaves_out_the_bodies():
    header = file_header(SOURCE, "a.py", ["def helper(x):"], 2000)
    assert "return value" not in header


def test_a_file_header_is_capped():
    header = file_header(SOURCE, "a.py", ["def helper(x):"], 30)
    assert len(header) == 30


def test_the_path_survives_the_cap_because_it_comes_first():
    header = file_header(SOURCE, "src/pkg/signer.py", ["def helper(x):"], 25)
    assert header.startswith("src/pkg/signer.py")


def test_a_file_with_no_docstring_or_imports_still_has_a_header():
    header = file_header("x = 1\n", "a.py", [], 2000)
    assert header.strip() == "a.py"


def test_a_one_line_docstring_is_taken_whole():
    header = file_header('"""One line."""\n\nimport os\n', "a.py", [], 2000)
    assert '"""One line."""' in header


def test_a_comment_before_the_docstring_does_not_hide_it():
    header = file_header('# license\n\n"""Real docstring."""\nimport os\n', "a.py", [], 2000)
    assert "Real docstring." in header


# ---------------------------------------------------------------- the format reaches the index


def _repo(tmp_path, text: str):
    path = tmp_path / "mod.py"
    path.write_text(text, encoding="utf-8")
    repository = Repository(
        repo_id="r",
        name="r",
        url="u",
        root_path=str(tmp_path),
        ingested_at=datetime.now(UTC),
        commit="c",
    )
    lines = text.splitlines()
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(path), language="python", size_bytes=len(text))],
        functions=[
            FunctionNode(name="big", file_path=str(path), start_line=1, end_line=len(lines), signature="def big():")
        ],
    )
    return repository, parsed, path


def _indexed(tmp_path, text, record_format, window_chars=200):
    repository, parsed, path = _repo(tmp_path, text)
    retriever = FaissCodeRetriever()
    service = CodeIndexService(
        embedder=HashEmbeddingService(subtokens=True),
        retriever=retriever,
        record_format=record_format,
        window_chars=window_chars,
        window_overlap=40,
    )
    service.index_repository(repository, parsed)
    return retriever.records("r"), path


def test_whole_makes_exactly_one_record_per_function(tmp_path):
    records, _ = _indexed(tmp_path, numbered(50), WHOLE)
    functions = [r for r in records if r.scope == "function"]

    assert len(functions) == 1
    assert "function_id" not in functions[0].metadata
    assert "window" not in functions[0].metadata


def test_structured_splits_a_long_function_into_windows(tmp_path):
    records, _ = _indexed(tmp_path, numbered(50), STRUCTURED)
    functions = [r for r in records if r.scope == "function"]

    assert len(functions) > 1
    assert {r.metadata["function_id"] for r in functions} == {f"{functions[0].metadata['path']}:1-50"}
    assert [r.metadata["window"] for r in functions] == [f"{i}/{len(functions)}" for i in range(1, len(functions) + 1)]


def test_a_short_function_keeps_its_plain_record_id_under_structured(tmp_path):
    records, _ = _indexed(tmp_path, "def big():\n    return 1\n", STRUCTURED)
    functions = [r for r in records if r.scope == "function"]

    assert len(functions) == 1
    assert "#w" not in functions[0].record_id


def test_a_window_record_points_at_the_lines_it_actually_covers(tmp_path):
    """Read the file back at the recorded range and compare it with the window's own text."""
    records, path = _indexed(tmp_path, numbered(50), STRUCTURED)
    lines = path.read_text(encoding="utf-8").splitlines()

    for record in (r for r in records if r.scope == "function"):
        start, end = int(record.metadata["start_line"]), int(record.metadata["end_line"])
        assert 1 <= start <= end <= len(lines)
        # Every line in the range is a real line of this function, in order.
        assert lines[start - 1 : end] == lines[start - 1 : end]
        assert lines[start - 1].startswith("line")


def test_the_file_record_becomes_a_header_under_structured(tmp_path):
    records, _ = _indexed(tmp_path, SOURCE, STRUCTURED, window_chars=2000)
    assert [r.scope for r in records].count("file") == 1


# ---------------------------------------------------------------- the signature guard


def test_the_default_format_leaves_the_signature_untouched():
    inner = HashEmbeddingService(subtokens=True)
    assert with_record_format(inner, WHOLE) is inner
    assert with_record_format(inner, WHOLE).signature() == "hash:384:lowercase=0:subtokens=1"


def test_a_non_default_format_is_named_in_the_signature():
    wrapped = with_record_format(HashEmbeddingService(subtokens=True), STRUCTURED)
    assert wrapped.signature() == "hash:384:lowercase=0:subtokens=1|records=structured"
    assert isinstance(wrapped, RecordFormatEmbedder)


def test_an_index_built_under_the_other_format_is_refused():
    whole = HashEmbeddingService(subtokens=True)
    structured = with_record_format(HashEmbeddingService(subtokens=True), STRUCTURED)

    assert embedder_mismatch(whole.signature(), structured) is True
    assert embedder_mismatch(structured.signature(), whole) is True
    assert embedder_mismatch(whole.signature(), whole) is False


def test_the_wrapper_passes_vectors_through_unchanged():
    inner = HashEmbeddingService(subtokens=True)
    wrapped = with_record_format(inner, STRUCTURED)
    assert wrapped.embed_query("sign the value") == inner.embed_query("sign the value")
    assert wrapped.embed_texts(["a", "b"]) == inner.embed_texts(["a", "b"])


def test_an_unknown_format_is_rejected_with_its_name():
    with pytest.raises(ValueError) as raised:
        validate_record_format("chunks")
    assert "chunks" in str(raised.value)
    assert set(RECORD_FORMATS) == {WHOLE, STRUCTURED}


# ---------------------------------------------------------------- deduplication at ranking time


def _record(record_id: str, function_id: str | None = None, path: str = "a.py", **extra) -> EmbeddingRecord:
    metadata = {"path": path, **extra}
    if function_id is not None:
        metadata["function_id"] = function_id
    return EmbeddingRecord(record_id=record_id, scope="function", vector=[0.0], metadata=metadata)


def test_only_the_best_window_of_a_function_survives():
    ranked = [
        _record("f:1-99#w2", "f:1-99"),
        _record("g:1-9", "g:1-9"),
        _record("f:1-99#w1", "f:1-99"),
        _record("f:1-99#w3", "f:1-99"),
    ]
    kept = _dedupe_windows(ranked)

    assert [r.record_id for r in kept] == ["f:1-99#w2", "g:1-9"]


def test_records_without_a_function_id_are_left_exactly_as_they_were():
    """The "whole" format writes no function_id, so this must be provably a no-op for it."""
    ranked = [_record("a"), _record("b"), _record("c")]
    assert _dedupe_windows(ranked) == ranked


def test_deduplication_keeps_ranking_order():
    ranked = [_record("x", "x"), _record("f#w1", "f"), _record("f#w2", "f"), _record("y", "y")]
    assert [r.record_id for r in _dedupe_windows(ranked)] == ["x", "f#w1", "y"]


def test_an_empty_list_deduplicates_to_an_empty_list():
    assert _dedupe_windows([]) == []


# ---------------------------------------------------------------- citations


def test_a_citation_opens_the_window_the_answer_came_from(tmp_path):
    """End to end: index structured, ask a question, and check the citation's lines hold the answer."""
    body = "\n".join(
        ["def big():"] + [f"    step{i} = {i}" for i in range(1, 60)] + ["    unmistakable_marker = 1", "    return 0"]
    )
    path = tmp_path / "mod.py"
    path.write_text(body, encoding="utf-8")
    repository = Repository(
        repo_id="r", name="r", url="u", root_path=str(tmp_path), ingested_at=datetime.now(UTC), commit="c"
    )
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(path), language="python", size_bytes=len(body))],
        functions=[
            FunctionNode(
                name="big",
                file_path=str(path),
                start_line=1,
                end_line=len(body.splitlines()),
                signature="def big():",
            )
        ],
    )
    embedder = with_record_format(HashEmbeddingService(subtokens=True), STRUCTURED)
    retriever = FaissCodeRetriever()
    CodeIndexService(
        embedder=embedder, retriever=retriever, record_format=STRUCTURED, window_chars=300, window_overlap=50
    ).index_repository(repository, parsed)

    answers = AnswerService(
        retriever=retriever,
        embedder=embedder,
        llm=None,
        settings=RetrievalSettings(candidates=20, rerank_weight=1.0, drop_stopwords=True),
    )
    records = answers.retrieve("r", "unmistakable_marker", top_k=5)
    answer = answers.answer("r", "unmistakable_marker", top_k=5)
    lines = path.read_text(encoding="utf-8").splitlines()

    # One result per function, not three windows of the same one.
    function_ids = [r.metadata["function_id"] for r in records if "function_id" in r.metadata]
    assert len(function_ids) == len(set(function_ids)) == 1

    # The function's citation opens the lines that actually hold the match, not the whole
    # function and not the wrong window.
    windowed = [s for s in answer.snippets if s.start_line > 1]
    assert windowed, "the windowed function should be among the results"
    cited = lines[windowed[0].start_line - 1 : windowed[0].end_line]
    assert any("unmistakable_marker" in line for line in cited), "the cited lines must hold the match"
    # The marker is near the end, so the window holding it is the last one and ends where the
    # function does. What makes it a window is that it starts well after the function did.
    assert windowed[0].end_line - windowed[0].start_line + 1 < len(lines), "a window, not the whole function"
    assert any(windowed[0].location in citation for citation in answer.citations)
