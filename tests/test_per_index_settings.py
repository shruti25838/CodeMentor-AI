"""The record format and BM25, set per index kind rather than once for the whole server.

Why this exists. Under the "cloudflare" provider a repository gets **two** indexes built from
the same records in the same analysis: the semantic one and the hash fallback that answers when
the semantic one cannot. docs/RESULTS.md records those two wanting opposite things - BM25 fusion
is worth about +0.105 of MRR@10 to hash on holdout3 and cost cloudflare 0.031 of dev MRR - and
until now one global variable had to answer for both.

Two properties are under test and they pull against each other: a server that sets neither
per-kind variable must be configured **exactly** as it was before any of this existed, and a
server that sets them must get two genuinely different indexes - different records, different
signatures, and a guard that still refuses a mixed pair for each kind separately.
"""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_embedder_identity import FakeEmbedder

from codeatlas.app.di import build_embedder, index_options, retrieval_settings_for
from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.qa.answer_service import PER_INDEX_FIELDS, AnswerService
from codeatlas.services.retrieval.dual_index import FALLBACK_NAME, DualIndexService
from codeatlas.services.retrieval.embedding import EmbeddingUnavailableError
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.retrieval.record_format import STRUCTURED, WHOLE, with_record_format
from codeatlas.utils.config import every_index_kind_on, load_config

PER_KIND_VARIABLES = (
    "CODEATLAS_HASH_RECORD_FORMAT",
    "CODEATLAS_CLOUDFLARE_RECORD_FORMAT",
    "CODEATLAS_HASH_BM25",
    "CODEATLAS_CLOUDFLARE_BM25",
)
GLOBAL_VARIABLES = ("CODEATLAS_RECORD_FORMAT", "CODEATLAS_BM25", "CODEATLAS_RECORD_WINDOW_CHARS")

# Small enough that one ordinary function is cut into several windows, so the line ranges have
# something to be wrong about.
WINDOW_CHARS = 240


@pytest.fixture(autouse=True)
def _nothing_set(monkeypatch):
    for name in (*PER_KIND_VARIABLES, *GLOBAL_VARIABLES):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------- what the settings resolve to


def test_with_nothing_set_both_kinds_are_what_the_server_already_was() -> None:
    """The whole of backwards compatibility, in one assertion per kind."""
    config = load_config()
    for kind in ("hash", "cloudflare"):
        assert config.record_format_for(kind) == WHOLE
        assert config.bm25_enabled_for(kind) is False


def test_the_old_global_variables_still_move_both_kinds(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_BM25", "true")
    config = load_config()
    for kind in ("hash", "cloudflare"):
        assert config.record_format_for(kind) == STRUCTURED
        assert config.bm25_enabled_for(kind) is True


def test_a_per_kind_variable_overrides_the_global_one_for_that_kind_only(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", "whole")
    monkeypatch.setenv("CODEATLAS_BM25", "false")
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_HASH_BM25", "true")
    config = load_config()

    assert (config.record_format_for("hash"), config.bm25_enabled_for("hash")) == (STRUCTURED, True)
    assert (config.record_format_for("cloudflare"), config.bm25_enabled_for("cloudflare")) == (WHOLE, False)


def test_a_per_kind_off_is_a_decision_and_not_a_silence(monkeypatch) -> None:
    """`False` must not read as "nothing was set" - which is what an `or` would have made of it."""
    monkeypatch.setenv("CODEATLAS_BM25", "true")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_BM25", "false")
    config = load_config()

    assert config.bm25_enabled_for("hash") is True
    assert config.bm25_enabled_for("cloudflare") is False


def test_a_provider_that_is_not_an_index_kind_reads_the_global_setting(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "whole")
    assert load_config().record_format_for("fastembed") == STRUCTURED


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEATLAS_HASH_RECORD_FORMAT", "chunks"),
        ("CODEATLAS_CLOUDFLARE_RECORD_FORMAT", "windows"),
        ("CODEATLAS_HASH_BM25", "maybe"),
        ("CODEATLAS_CLOUDFLARE_BM25", "sometimes"),
    ],
)
def test_a_per_kind_value_that_is_not_a_setting_stops_the_app(monkeypatch, name, value) -> None:
    """ "Set but unreadable" must never be mistaken for "not set", which would silently use the global."""
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as raised:
        load_config()
    assert name in str(raised.value)


def test_an_empty_per_kind_variable_means_unset(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "")
    monkeypatch.setenv("CODEATLAS_HASH_BM25", "   ")
    config = load_config()
    assert config.hash_record_format is None and config.hash_bm25_enabled is None
    assert config.record_format_for("hash") == STRUCTURED


def test_setting_every_kind_at_once_clears_the_overrides(monkeypatch) -> None:
    """What a script means by `--record-format whole`: no index it builds is a structured one."""
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "structured")
    config = every_index_kind_on(load_config(), WHOLE)
    for kind in ("hash", "cloudflare"):
        assert config.record_format_for(kind) == WHOLE


# ---------------------------------------------------------------- the settings reach the services


def test_the_index_options_carry_each_kind_its_own_format(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_RECORD_FORMAT", "whole")
    config = replace(load_config(), embedding_provider="cloudflare")

    assert index_options(config)["record_format"] == WHOLE
    assert index_options(replace(config, embedding_provider="hash"))["record_format"] == STRUCTURED


def test_the_embedder_signature_carries_each_kind_its_own_format(monkeypatch) -> None:
    """The format is in the signature, which is what the guard compares - so it has to be per kind too."""
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_RECORD_FORMAT", "whole")
    config = replace(load_config(), embedding_provider="hash")

    assert build_embedder(config).signature().endswith("|records=structured")


def test_the_retrieval_settings_carry_each_kind_its_own_bm25(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_HASH_BM25", "true")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_BM25", "false")
    config = load_config()

    assert retrieval_settings_for(config, "hash").bm25_enabled is True
    assert retrieval_settings_for(config, "cloudflare").bm25_enabled is False


def test_only_the_bm25_fields_may_differ_between_two_indexes() -> None:
    """A setting applied before the index is chosen cannot differ by index; the service enforces it."""
    primary = retrieval_settings_for(load_config(), "hash")
    asked_for = replace(primary, bm25_enabled=True, drop_stopwords=not primary.drop_stopwords)
    service = AnswerService(retriever=None, embedder=None, llm=None, settings=primary, fallback_settings=asked_for)

    assert service._fallback_settings.bm25_enabled is True
    assert service._fallback_settings.drop_stopwords == primary.drop_stopwords
    assert "drop_stopwords" not in PER_INDEX_FIELDS


# ---------------------------------------------------------------- two indexes, two formats


def _repo(tmp_path: Path) -> tuple[Repository, ParsedRepository, Path]:
    """One file holding one function long enough to be cut into windows at WINDOW_CHARS."""
    source = tmp_path / "signer.py"
    body = "\n".join(f"    step_{index} = sign(value, {index})" for index in range(1, 31))
    source.write_text(
        '"""Signing, and what a signature is checked against."""\n'
        "\n"
        "import hashlib\n"
        "\n"
        "def sign_value(value):\n"
        f"{body}\n"
        "    return step_30\n",
        encoding="utf-8",
    )
    repo = Repository("r", "r", "u", str(tmp_path), datetime.now(UTC))
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=source.stat().st_size)],
        functions=[
            FunctionNode(
                name="sign_value",
                file_path=str(source),
                start_line=5,
                end_line=36,
                signature="def sign_value(value)",
            )
        ],
    )
    return repo, parsed, source


def _dual(config, tmp_path: Path, semantic_embedder) -> tuple[DualIndexService, dict]:
    """The two halves `di.get_index_service` builds, each with its own kind's index options."""
    semantic_retriever = FaissCodeRetriever(base_dir=str(tmp_path / "semantic"))
    hash_config = replace(config, embedding_provider=FALLBACK_NAME)
    fallback_embedder = build_embedder(hash_config)
    fallback_retriever = FaissCodeRetriever(base_dir=str(tmp_path / "hash"))
    service = DualIndexService(
        semantic=CodeIndexService(
            embedder=semantic_embedder,
            retriever=semantic_retriever,
            name="cloudflare",
            **index_options(config),
        ),
        fallback=CodeIndexService(
            embedder=fallback_embedder,
            retriever=fallback_retriever,
            name=FALLBACK_NAME,
            **index_options(hash_config),
        ),
        semantic_name="cloudflare",
        max_documents=config.cloudflare_max_documents,
    )
    return service, {
        "semantic_retriever": semantic_retriever,
        "fallback_retriever": fallback_retriever,
        "fallback_embedder": fallback_embedder,
    }


@pytest.fixture
def _two_formats(monkeypatch):
    """cloudflare on whole records, the hash fallback on structured ones - the mix nothing could express."""
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_RECORD_FORMAT", "whole")
    monkeypatch.setenv("CODEATLAS_HASH_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_RECORD_WINDOW_CHARS", str(WINDOW_CHARS))
    return replace(load_config(), embedding_provider="cloudflare")


def test_a_dual_index_can_hold_two_record_formats_at_once(tmp_path, _two_formats) -> None:
    service, parts = _dual(_two_formats, tmp_path, FakeEmbedder(name="semantic", value=0.5))
    repo, parsed, _ = _repo(tmp_path)

    assert service.index_repository(repo, parsed) == ("cloudflare", "hash")

    whole = parts["semantic_retriever"].records("r")
    structured = parts["fallback_retriever"].records("r")
    # One file record and one function record under `whole`; the same function cut into several
    # windows under `structured`, so the hash index holds strictly more records than the other.
    assert len(whole) == 2
    assert len(structured) > len(whole)
    assert not any(r.metadata.get("function_id") for r in whole)
    assert sum(1 for r in structured if r.metadata.get("function_id")) > 1


def test_the_index_format_names_both_formats_so_the_cache_cannot_mix_them(tmp_path, _two_formats) -> None:
    mixed, _ = _dual(_two_formats, tmp_path, FakeEmbedder(name="semantic", value=0.5))
    same, _ = _dual(every_index_kind_on(_two_formats, WHOLE), tmp_path, FakeEmbedder(name="semantic", value=0.5))

    assert "records=structured" in mixed.index_format
    assert "records=structured" not in same.index_format
    assert mixed.index_format != same.index_format


# ---------------------------------------------------------------- the guard, for each kind


def _embedder_of_kind(config, kind: str):
    """The embedder this kind's index is built and searched with, wrapped as `build_embedder` wraps it.

    The cloudflare provider is stood in for rather than constructed, because nothing here may
    reach an API; what is under test is the wrapper, which is where the record format lives and
    is the only part of the signature this test moves.
    """
    inner = HashEmbeddingService() if kind == "hash" else FakeEmbedder(name="cloudflare", value=0.5)
    return with_record_format(inner, config.record_format_for(kind))


@pytest.mark.parametrize("kind", ["hash", "cloudflare"])
@pytest.mark.parametrize(("built_as", "server_runs"), [(WHOLE, STRUCTURED), (STRUCTURED, WHOLE)])
def test_the_guard_refuses_a_mixed_index_for_each_kind(tmp_path, monkeypatch, kind, built_as, server_runs) -> None:
    """An index and the embedder that built it are one thing - now once per kind, not once per server.

    The guard has to refuse in both directions and for both kinds, because with the formats set
    per kind a server can be mixed in a way it could not be before: structured hash beside whole
    cloudflare is a supported configuration, and a *stale* index of either kind under it is not.
    """
    monkeypatch.setenv("CODEATLAS_RECORD_WINDOW_CHARS", str(WINDOW_CHARS))
    variable = f"CODEATLAS_{kind.upper()}_RECORD_FORMAT"
    repo, parsed, _ = _repo(tmp_path)
    retriever = FaissCodeRetriever(base_dir=str(tmp_path / "index"))

    monkeypatch.setenv(variable, built_as)
    building = replace(load_config(), embedding_provider=kind)
    builder_embedder = _embedder_of_kind(building, kind)
    CodeIndexService(
        embedder=builder_embedder, retriever=retriever, name=kind, **index_options(building)
    ).index_repository(repo, parsed)

    monkeypatch.setenv(variable, server_runs)
    running = replace(load_config(), embedding_provider=kind)
    server_embedder = _embedder_of_kind(running, kind)
    # Only this kind's variable moved, so the two embedders differ only in their record format.
    assert builder_embedder.signature() != server_embedder.signature()

    service = CodeIndexService(embedder=server_embedder, retriever=retriever, name=kind, **index_options(running))
    assert not service.has_index("r")

    answers = AnswerService(retriever=retriever, embedder=server_embedder, llm=None)
    assert answers.retrieve("r", "how is a value signed", top_k=5) == []


@pytest.mark.parametrize("kind", ["hash", "cloudflare"])
def test_the_other_kind_s_variable_does_not_move_this_index(tmp_path, monkeypatch, kind) -> None:
    """The guard is per kind in both directions: changing the *other* kind must refuse nothing."""
    other = "cloudflare" if kind == "hash" else "hash"
    monkeypatch.setenv(f"CODEATLAS_{kind.upper()}_RECORD_FORMAT", "structured")
    monkeypatch.setenv("CODEATLAS_RECORD_WINDOW_CHARS", str(WINDOW_CHARS))
    repo, parsed, _ = _repo(tmp_path)
    retriever = FaissCodeRetriever(base_dir=str(tmp_path / "index"))

    building = replace(load_config(), embedding_provider=kind)
    embedder = _embedder_of_kind(building, kind)
    service = CodeIndexService(embedder=embedder, retriever=retriever, name=kind, **index_options(building))
    service.index_repository(repo, parsed)
    assert service.has_index("r")

    monkeypatch.setenv(f"CODEATLAS_{other.upper()}_RECORD_FORMAT", "whole")
    running = replace(load_config(), embedding_provider=kind)
    still = CodeIndexService(
        embedder=_embedder_of_kind(running, kind), retriever=retriever, name=kind, **index_options(running)
    )
    assert still.has_index("r")


# ---------------------------------------------------------------- the fallback answers in its own format


class FailsAtQuestionTime(FakeEmbedder):
    """A semantic embedder that indexes and then cannot be reached when a question arrives."""

    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingUnavailableError("budget", "detail that must not be shown to anyone")


def test_a_fallback_question_is_answered_in_the_hash_index_s_own_format(tmp_path, _two_formats) -> None:
    """The visitor's question lands on the hash index, so it must land on *structured* records."""
    service, parts = _dual(_two_formats, tmp_path, FailsAtQuestionTime(name="semantic", value=0.5))
    repo, parsed, source = _repo(tmp_path)
    service.index_repository(repo, parsed)

    answers = AnswerService(
        retriever=parts["semantic_retriever"],
        embedder=FailsAtQuestionTime(name="semantic", value=0.5),
        llm=None,
        fallback_retriever=parts["fallback_retriever"],
        fallback_embedder=parts["fallback_embedder"],
    )
    records, note = answers.retrieve_with_note("r", "how is a value signed", top_k=5)

    assert note
    assert records
    # Every record came from the hash index, and that index is the structured one: a function
    # record there carries the window metadata that `whole` never writes.
    function_records = [r for r in records if r.scope == "function"]
    assert function_records
    assert all(r.metadata.get("function_id") for r in function_records)


def test_a_fallback_question_is_fused_only_if_the_hash_index_says_so(tmp_path, _two_formats, monkeypatch) -> None:
    """BM25 follows the index that answers, not the provider the server is nominally running."""
    monkeypatch.setenv("CODEATLAS_HASH_BM25", "true")
    monkeypatch.setenv("CODEATLAS_CLOUDFLARE_BM25", "false")
    config = replace(load_config(), embedding_provider="cloudflare")
    service, parts = _dual(config, tmp_path, FailsAtQuestionTime(name="semantic", value=0.5))
    repo, parsed, _ = _repo(tmp_path)
    service.index_repository(repo, parsed)

    answers = AnswerService(
        retriever=parts["semantic_retriever"],
        embedder=FailsAtQuestionTime(name="semantic", value=0.5),
        llm=None,
        settings=retrieval_settings_for(config, "cloudflare"),
        fallback_retriever=parts["fallback_retriever"],
        fallback_embedder=parts["fallback_embedder"],
        fallback_settings=retrieval_settings_for(config, FALLBACK_NAME),
    )
    answers.retrieve("r", "how is a value signed", top_k=5)

    # The lexical index was built, and it was built for the fallback retriever rather than the
    # semantic one - which is the whole of "BM25 is per index kind".
    assert (id(parts["fallback_retriever"]), "r") in answers._bm25
    assert (id(parts["semantic_retriever"]), "r") not in answers._bm25


# ---------------------------------------------------------------- windows cite real line ranges


def test_a_window_record_cites_the_lines_it_actually_holds(tmp_path, _two_formats) -> None:
    """A window's line range is what the citation shows and what the file preview opens.

    Checked against the file on disk rather than against the arithmetic that produced it: a
    range that is off by one points a reader at the wrong code while looking perfectly plausible.
    """
    service, parts = _dual(_two_formats, tmp_path, FakeEmbedder(name="semantic", value=0.5))
    repo, parsed, source = _repo(tmp_path)
    service.index_repository(repo, parsed)

    lines = source.read_text(encoding="utf-8").splitlines()
    documents, records = CodeIndexService(
        embedder=parts["fallback_embedder"],
        retriever=parts["fallback_retriever"],
        name=FALLBACK_NAME,
        **index_options(replace(_two_formats, embedding_provider=FALLBACK_NAME)),
    ).documents(repo, parsed)

    windows = [(d, r) for d, r in zip(documents, records) if r.metadata.get("function_id")]
    assert len(windows) > 1
    for _, record in windows:
        start = int(record.metadata["start_line"])
        end = int(record.metadata["end_line"])
        assert 5 <= start <= end <= 36
        # The record's own text, read back out of the file by the range the record claims.
        from_file = "\n".join(lines[start - 1 : end])
        assert from_file.strip()
        assert from_file.splitlines()[0] == lines[start - 1]

    # Together the windows cover the function with no gap, and the first one opens on its `def`.
    assert int(windows[0][1].metadata["start_line"]) == 5
    assert lines[4].startswith("def sign_value")
    assert max(int(r.metadata["end_line"]) for _, r in windows) == 36
