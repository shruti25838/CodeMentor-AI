from datetime import UTC, datetime
from pathlib import Path

from codeatlas.app.di import get_retrieval_settings
from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.models.function_node import FunctionNode
from codeatlas.models.parsed_repository import ParsedRepository
from codeatlas.models.repository import Repository
from codeatlas.models.source_file import SourceFile
from codeatlas.services.qa.answer_service import AnswerService, RetrievalSettings, _is_test_path
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService, split_identifier
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.retrieval.interfaces import CodeRetriever


class RecordingRetriever(CodeRetriever):
    def __init__(self, records: list[EmbeddingRecord] | None = None) -> None:
        self.records = records or []
        self.searched_top_k: int | None = None
        self.indexed: list[EmbeddingRecord] = []

    def index(self, repo_id: str, records: list[EmbeddingRecord]) -> None:
        self.indexed = records

    def search(self, repo_id: str, query_vector: list[float], top_k: int) -> list[EmbeddingRecord]:
        self.searched_top_k = top_k
        return self.records[:top_k]

    def has_index(self, repo_id: str) -> bool:
        return True

    def remove(self, repo_id: str) -> None:
        return None


class RecordingEmbedder(EmbeddingService):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.queries: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


def _file_record(path: Path, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(record_id=str(path), scope="file", vector=vector, metadata={"path": str(path)})


def test_split_identifier():
    assert split_identifier("verify_signature") == ["verify", "signature"]
    assert split_identifier("URLSafeTimedSerializer") == ["url", "safe", "timed", "serializer"]
    assert split_identifier("load_payload2") == ["load", "payload", "2"]
    assert split_identifier("x") == ["x"]


def test_hash_embedder_options():
    plain = HashEmbeddingService()
    assert plain.embed_query("Signer") != plain.embed_query("signer")  # default: case-sensitive, as before
    lower = HashEmbeddingService(lowercase=True)
    assert lower.embed_query("Signer") == lower.embed_query("signer")
    sub = HashEmbeddingService(subtokens=True)
    dot = sum(a * b for a, b in zip(sub.embed_query("verify_signature"), sub.embed_query("signature")))
    assert dot > 0
    assert sum(a * b for a, b in zip(plain.embed_query("verify_signature"), plain.embed_query("signature"))) == 0


def test_candidates_set_the_search_depth(tmp_path):
    retriever = RecordingRetriever()
    AnswerService(retriever, RecordingEmbedder()).retrieve("r", "q", top_k=3)
    assert retriever.searched_top_k == 10  # the old fixed depth
    AnswerService(retriever, RecordingEmbedder(), settings=RetrievalSettings(candidates=25)).retrieve("r", "q", 3)
    assert retriever.searched_top_k == 25


def test_rerank_weight_zero_keeps_vector_order_and_one_orders_by_keywords(tmp_path):
    words = tmp_path / "words.py"
    words.write_text("signature check")
    other = tmp_path / "other.py"
    other.write_text("unrelated")
    records = [_file_record(other, [1.0, 0.0]), _file_record(words, [0.0, 1.0])]
    by_keyword = AnswerService(RecordingRetriever(records), RecordingEmbedder()).retrieve("r", "signature check", 2)
    assert [r.record_id for r in by_keyword] == [str(words), str(other)]
    by_vector = AnswerService(
        RecordingRetriever(records), RecordingEmbedder(), settings=RetrievalSettings(rerank_weight=0.0)
    ).retrieve("r", "signature check", 2)
    assert [r.record_id for r in by_vector] == [str(other), str(words)]


def test_rerank_subtokens_match_parts_of_identifiers(tmp_path):
    code = tmp_path / "code.py"
    code.write_text("def verify_signature(): pass")
    other = tmp_path / "other.py"
    other.write_text("nothing here")
    records = [_file_record(other, [1.0, 0.0]), _file_record(code, [1.0, 0.0])]
    plain = AnswerService(RecordingRetriever(records), RecordingEmbedder()).retrieve("r", "signature", 2)
    assert plain[0].record_id == str(other)  # no whole-word match, vector order kept
    split = AnswerService(
        RecordingRetriever(records), RecordingEmbedder(), settings=RetrievalSettings(rerank_subtokens=True)
    ).retrieve("r", "signature", 2)
    assert split[0].record_id == str(code)


def test_drop_stopwords_cleans_the_search_question():
    embedder = RecordingEmbedder()
    AnswerService(RecordingRetriever(), embedder, settings=RetrievalSettings(drop_stopwords=True)).retrieve(
        "r", "How does the Signer check a signature?", 5
    )
    assert embedder.queries == ["Signer check signature"]
    embedder = RecordingEmbedder()
    AnswerService(RecordingRetriever(), embedder).retrieve("r", "How does it work?", 5)
    assert embedder.queries == ["How does it work?"]


def test_skip_tests_filters_test_files(tmp_path):
    src = tmp_path / "src" / "app.py"
    test = tmp_path / "tests" / "test_app.py"
    for path in (src, test):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("app")
    records = [_file_record(test, [1.0, 0.0]), _file_record(src, [1.0, 0.0])]
    kept = AnswerService(
        RecordingRetriever(records), RecordingEmbedder(), settings=RetrievalSettings(skip_tests=True)
    ).retrieve("r", "app", 5)
    assert [r.record_id for r in kept] == [str(src)]


def test_is_test_path():
    assert _is_test_path("/r/tests/conftest.py")
    assert _is_test_path("/r/src/test_x.py")
    assert _is_test_path("C:\\r\\tests\\helpers.py")
    assert _is_test_path("/r/pkg/x_test.py")
    assert not _is_test_path("/r/src/flask/testing.py")
    assert not _is_test_path("/r/src/contest.py")


def _index(tmp_path, **options) -> RecordingEmbedder:
    source = tmp_path / "pkg" / "mod.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def run():\n    return 'x' * 100\n")
    parsed = ParsedRepository(
        repository_id="r",
        files=[SourceFile(path=str(source), language="python", size_bytes=10)],
        functions=[FunctionNode(name="run", file_path=str(source), start_line=1, end_line=2, signature="def run():")],
    )
    repo = Repository(repo_id="r", name="r", url="u", root_path=str(tmp_path), ingested_at=datetime.now(UTC))
    embedder = RecordingEmbedder()
    CodeIndexService(embedder, RecordingRetriever(), **options).index_repository(repo, parsed)
    return embedder


def test_index_defaults_embed_the_full_text(tmp_path):
    embedder = _index(tmp_path)
    assert embedder.texts[0] == "def run():\n    return 'x' * 100\n"


def test_index_max_chars_and_metadata_prefix(tmp_path):
    assert _index(tmp_path, max_chars=8).texts[0] == "def run("
    texts = _index(tmp_path, prefix_metadata=True).texts
    assert texts[0].startswith("pkg/mod.py\ndef run")
    assert texts[1].startswith("pkg/mod.py run\ndef run")


def test_server_settings_come_from_config():
    assert get_retrieval_settings() == RetrievalSettings(
        candidates=10, rerank_weight=1.0, rerank_subtokens=False, drop_stopwords=False, skip_tests=False
    )
