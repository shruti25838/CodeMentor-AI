"""Retrieval hands later steps real code, capped, with the path and line range it came from.

No model key is needed: AnswerService is built with no model where the text does not matter,
and with a recording fake where the prompt itself is what is being checked.
"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.agents.retrieval_agent import RetrievalAgent, format_retrieval
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever
from codeatlas.services.retrieval.snippets import (
    CLIPPED_MARKER,
    Snippet,
    clip_snippet,
    fit_snippets,
    render_snippets,
)


class StubRetriever(CodeRetriever):
    def __init__(self, records: list[EmbeddingRecord]) -> None:
        self._records = records

    def index(self, repo_id, records) -> None:
        return None

    def search(self, repo_id, query_vector, top_k):
        return self._records[:top_k]

    def has_index(self, repo_id) -> bool:
        return True

    def remove(self, repo_id) -> None:
        return None


class StubEmbedder(EmbeddingService):
    def embed_texts(self, texts):
        return [[1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0]


class RecordingModel(BaseChatModel):
    """Keeps every prompt it is given, so a test can inspect what the model would have seen."""

    state: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"prompts": [], "error": None})()

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.state.prompts.append("\n".join(str(m.content) for m in messages))
        if self.state.error:
            raise self.state.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="model answer"))])


@pytest.fixture
def repo(tmp_path):
    """A file with a known function at known lines, plus a large file."""
    src = tmp_path / "pkg"
    src.mkdir()
    small = src / "core.py"
    small.write_text(
        "import os\n\n\ndef add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    big = src / "big.py"
    big.write_text("\n".join(f"line_{i} = {i}" for i in range(2000)), encoding="utf-8")
    return small, big


def function_record(path, name, start, end) -> EmbeddingRecord:
    return EmbeddingRecord(
        record_id=f"{path}:{start}-{end}",
        scope="function",
        vector=[1.0],
        metadata={
            "path": str(path),
            "name": name,
            "signature": f"def {name}(a, b):",
            "start_line": str(start),
            "end_line": str(end),
        },
    )


def file_record(path) -> EmbeddingRecord:
    return EmbeddingRecord(
        record_id=str(path), scope="file", vector=[1.0], metadata={"path": str(path), "language": "python"}
    )


def service(records, llm=None, **kwargs) -> AnswerService:
    return AnswerService(retriever=StubRetriever(records), embedder=StubEmbedder(), llm=llm, **kwargs)


# ---------- snippets carry real code and a true line range ----------


def test_function_snippet_is_the_real_code_with_its_line_range(repo) -> None:
    small, _ = repo
    result = service([function_record(small, "add", 4, 5)]).answer("r", "how does add work")

    assert len(result.snippets) == 1
    snippet = result.snippets[0]
    assert "def add(a, b):" in snippet.text
    assert "return a + b" in snippet.text
    assert (snippet.start_line, snippet.end_line) == (4, 5)
    assert not snippet.clipped


def test_citations_describe_exactly_the_snippets_that_were_read(repo) -> None:
    small, _ = repo
    result = service([function_record(small, "add", 4, 5), function_record(small, "sub", 8, 9)]).answer("r", "q")

    assert len(result.citations) == len(result.snippets)
    for citation, snippet in zip(result.citations, result.snippets, strict=True):
        assert citation.startswith(snippet.location)


def test_citation_has_a_line_range_even_for_a_whole_file(repo) -> None:
    """File-scope citations used to have no line range at all."""
    small, _ = repo
    result = service([file_record(small)]).answer("r", "q")
    assert "(lines 1-" in result.citations[0]


# ---------- the caps that stop a request exceeding the model's token allowance ----------


def test_a_whole_file_is_clipped_rather_than_sent_entire(repo) -> None:
    """A 2,000-line file used to be sent in full; one /ask asked for 10,391 tokens of an 8,000 limit."""
    _, big = repo
    result = service([file_record(big)], snippet_max_chars=500).answer("r", "q")

    snippet = result.snippets[0]
    assert snippet.clipped
    assert len(snippet.text) <= 500 + len(CLIPPED_MARKER) + 1
    assert snippet.end_line < 2000


def test_total_context_is_capped_across_snippets(repo) -> None:
    small, _ = repo
    records = [function_record(small, "add", 1, 9) for _ in range(20)]
    result = service(records, snippet_max_chars=200, total_max_chars=400).answer("r", "q")

    assert len(render_snippets(result.snippets)) <= 400
    assert len(result.snippets) < 20


def test_the_model_only_ever_sees_capped_context(repo) -> None:
    _, big = repo
    model = RecordingModel()
    service([file_record(big)], llm=model, snippet_max_chars=300, total_max_chars=600).answer("r", "q")

    prompt = model.state.prompts[0]
    assert len(prompt) < 3000, "the whole 2,000-line file must not reach the model"
    assert "line_0 = 0" in prompt, "but the start of the file should"


def test_clip_keeps_whole_lines_and_reports_the_range_it_kept() -> None:
    text = "\n".join(f"line {i}" for i in range(100))
    snippet = clip_snippet("a.py", text, start_line=10, end_line=109, max_chars=40)

    assert snippet.clipped
    assert snippet.text.endswith(CLIPPED_MARKER)
    assert snippet.start_line == 10
    assert snippet.end_line < 109
    # Every kept line is whole.
    for line in snippet.text.splitlines():
        assert line == CLIPPED_MARKER or line.startswith("line ")


def test_a_single_overlong_line_is_still_returned() -> None:
    snippet = clip_snippet("a.py", "x" * 500, start_line=3, end_line=3, max_chars=50)
    assert snippet.clipped
    assert snippet.text.startswith("x")
    assert len(snippet.text) <= 50 + len(CLIPPED_MARKER)


def test_fit_drops_whole_snippets_rather_than_cutting_them() -> None:
    snippets = [Snippet("a.py", 1, 2, "a" * 100), Snippet("b.py", 1, 2, "b" * 100), Snippet("c.py", 1, 2, "c" * 100)]
    kept = fit_snippets(snippets, max_chars=260)

    assert len(kept) == 2
    assert all(len(s.text) == 100 for s in kept), "kept snippets are whole"


# ---------- what the next step receives ----------


def test_retrieval_output_contains_the_code_not_only_a_summary(repo) -> None:
    """The mentor used to receive a prose summary and had to guess at the code."""
    small, _ = repo
    agent = RetrievalAgent(service([function_record(small, "add", 4, 5)], llm=RecordingModel()))
    output = agent.run("how does add work", "repo")

    assert "def add(a, b):" in output
    assert "return a + b" in output
    assert "pkg/core.py" in output.replace("\\", "/")
    assert "(lines 4-5)" in output


def test_retrieve_returns_the_snippets_themselves(repo) -> None:
    small, _ = repo
    agent = RetrievalAgent(service([function_record(small, "add", 4, 5)]))
    result = agent.retrieve("q", "repo")

    assert [s.location.replace("\\", "/").split("/")[-1] for s in result.snippets] == ["core.py (lines 4-5)"]


def test_retrieve_without_a_repo_id_returns_no_snippets() -> None:
    assert RetrievalAgent(service([])).retrieve("q", None).snippets == []


def test_format_retrieval_without_snippets_does_not_wrap_the_summary() -> None:
    assert format_retrieval("No relevant code locations found.", []) == "No relevant code locations found."


# ---------- the model failing must not lose the locations ----------


def test_a_model_failure_still_returns_the_snippets_and_citations(repo) -> None:
    """This is the 413 path: the request was rejected, but the code that was read is still true."""
    from codeatlas.services.llm.quota import FILES_HEADING, MESSAGES, TOO_LARGE

    small, _ = repo
    model = RecordingModel()
    model.state.error = RuntimeError("Error code: 413 - request too large")
    result = service([function_record(small, "add", 4, 5)], llm=model).answer("r", "q")

    assert result.snippets, "snippets survive a failed model call"
    assert result.citations
    # The visitor is told why, and the locations sit underneath a heading rather than
    # standing in for an answer.
    assert MESSAGES[TOO_LARGE] in result.answer
    assert FILES_HEADING in result.answer
    assert "core.py (lines 4-5)" in result.answer.replace("\\", "/")


def test_no_records_gives_an_honest_empty_answer() -> None:
    result = service([]).answer("r", "q")
    assert result.snippets == []
    assert result.citations == []
    assert "No relevant code locations found." in result.answer
