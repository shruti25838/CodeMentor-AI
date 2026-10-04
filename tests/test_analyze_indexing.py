import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.di import get_index_service, get_repo_state_store, get_repository_loader
from codeatlas.app.main import create_app
from codeatlas.controllers.analyze_controller import INDEX_FAILED_DETAIL, INDEX_TIMEOUT_DETAIL
from codeatlas.models.repository import Repository
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.state.repo_state_store import RepoStateStore
from codeatlas.utils.config import AppConfig

CONFIG = AppConfig(
    embedding_provider="hash",
    embedding_model="",
    index_dir=".codeatlas/indexes",
    state_dir=".codeatlas/state",
    llm_provider="groq",
    llm_model="",
    llm_temperature=0.2,
    api_key=None,
    auth_enabled=False,
)

FILES = {
    "pkg/__init__.py": "from pkg.core import add\n",
    "pkg/core.py": "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
    "pkg/cli.py": "from pkg.core import add, sub\n\n\ndef main():\n    print(add(1, 2), sub(3, 1))\n",
}


class FakeLoader:
    """Stands in for a git clone: writes a small Python project into a fresh folder."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.loaded: list[Repository] = []

    def load(self, repo_url: str) -> Repository:
        repo_id = str(uuid.uuid4())
        root = self.base / repo_id
        for rel, text in FILES.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text, encoding="utf-8")
        repo = Repository(repo_id, "demo", repo_url, str(root), datetime.now(UTC))
        self.loaded.append(repo)
        return repo


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Env:
    def __init__(self, tmp_path: Path, embedder=None, **index_kwargs) -> None:
        self.loader = FakeLoader(tmp_path / "repos")
        self.index_dir = tmp_path / "indexes"
        self.state = RepoStateStore(base_dir=str(tmp_path / "state"))
        self.index = CodeIndexService(
            embedder=embedder or HashEmbeddingService(),
            retriever=FaissCodeRetriever(base_dir=str(self.index_dir)),
            **index_kwargs,
        )
        app = create_app(CONFIG)
        app.dependency_overrides[get_repository_loader] = lambda: self.loader
        app.dependency_overrides[get_index_service] = lambda: self.index
        app.dependency_overrides[get_repo_state_store] = lambda: self.state
        self.client = TestClient(app)

    def analyze(self):
        return self.client.post("/analyze-repo", json={"repo_url": "https://github.com/example/demo"})

    def assert_nothing_left(self) -> None:
        repo = self.loader.loaded[-1]
        assert not Path(repo.root_path).exists(), "partial clone was not removed"
        assert list(self.index_dir.glob(f"{repo.repo_id}.*")) == [], "partial index was not removed"
        assert self.state.get(repo.repo_id) is None
        assert self.client.get("/repos").json()["repo_ids"] == []


def test_success_indexes_before_answering(tmp_path):
    env = Env(tmp_path)
    resp = env.analyze()
    assert resp.status_code == 200
    body = resp.json()
    # Shape the frontend (lib/api.ts indexRepository) and earlier clients rely on.
    assert set(body) == {"repository_id", "file_count", "dependency_edges", "indexing_status"}
    assert body["indexing_status"] == "ready"
    assert body["file_count"] == 3
    repo_id = body["repository_id"]
    assert env.state.get(repo_id) is not None
    assert sorted(p.suffix for p in env.index_dir.glob(f"{repo_id}.*")) == [".faiss", ".pkl"]
    assert env.client.get("/repos").json()["repo_ids"] == [repo_id]


class FailingEmbedder(HashEmbeddingService):
    def embed_texts(self, texts):
        raise RuntimeError("embedding backend crashed")


def test_indexing_failure_returns_500_and_cleans_up(tmp_path):
    env = Env(tmp_path, embedder=FailingEmbedder())
    resp = env.analyze()
    assert resp.status_code == 500
    assert resp.json() == {"detail": INDEX_FAILED_DETAIL}
    env.assert_nothing_left()


def test_parse_failure_returns_500_and_cleans_up(tmp_path, monkeypatch):
    env = Env(tmp_path)
    from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser

    def boom(self, repo):
        raise ValueError("parser crashed")

    monkeypatch.setattr(TreeSitterAstParser, "parse_repository", boom)
    resp = env.analyze()
    assert resp.status_code == 500
    env.assert_nothing_left()


class SlowEmbedder(HashEmbeddingService):
    """Each batch 'takes' 10 seconds of fake time."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.batches = 0

    def embed_texts(self, texts):
        self.batches += 1
        self.clock.now += 10
        return super().embed_texts(texts)


def test_timeout_during_embedding_returns_504_and_cleans_up(tmp_path):
    clock = FakeClock()
    embedder = SlowEmbedder(clock)
    env = Env(tmp_path, embedder=embedder, timeout_seconds=25, batch_size=1, clock=clock)
    resp = env.analyze()
    assert resp.status_code == 504
    assert resp.json() == {"detail": INDEX_TIMEOUT_DETAIL}
    # Stopped between batches once past the limit, not after embedding everything.
    assert embedder.batches == 3
    env.assert_nothing_left()


def test_timeout_after_last_batch_still_stores_nothing(tmp_path):
    clock = FakeClock()
    env = Env(tmp_path, embedder=SlowEmbedder(clock), timeout_seconds=5, batch_size=1000, clock=clock)
    assert env.analyze().status_code == 504
    env.assert_nothing_left()


class ObservingEmbedder(HashEmbeddingService):
    """Records what clients could see while indexing is in progress."""

    def __init__(self) -> None:
        super().__init__()
        self.env: Env | None = None
        self.seen: list[tuple] = []

    def embed_texts(self, texts):
        repo = self.env.loader.loaded[-1]
        self.seen.append((self.env.state.get(repo.repo_id), self.env.client.get("/repos").json()["repo_ids"]))
        return super().embed_texts(texts)


def test_no_half_ready_repo_while_indexing(tmp_path):
    embedder = ObservingEmbedder()
    env = Env(tmp_path, embedder=embedder)
    embedder.env = env
    assert env.analyze().status_code == 200
    assert embedder.seen and all(state is None and listed == [] for state, listed in embedder.seen)


def test_faiss_remove_deletes_memory_and_files(tmp_path):
    from codeatlas.models.embedding_record import EmbeddingRecord

    retriever = FaissCodeRetriever(base_dir=str(tmp_path))
    retriever.index("r1", [EmbeddingRecord(record_id="a", scope="file", vector=[1.0, 0.0], metadata={})])
    assert list(tmp_path.glob("r1.*"))
    retriever.remove("r1")
    assert list(tmp_path.glob("r1.*")) == []
    assert retriever.search("r1", [1.0, 0.0], 1) == []
    retriever.remove("never-indexed")


@pytest.mark.parametrize("value,expected", [(None, 120), ("30", 30)])
def test_index_timeout_config(monkeypatch, value, expected):
    from codeatlas.utils.config import load_config

    if value is None:
        monkeypatch.delenv("CODEATLAS_INDEX_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("CODEATLAS_INDEX_TIMEOUT_SECONDS", value)
    assert load_config().index_timeout_seconds == expected
