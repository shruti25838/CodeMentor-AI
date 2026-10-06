import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from test_analyze_indexing import CONFIG, Env, FakeLoader
from test_git_loader import _make_source_repo

from codeatlas.models.repository import Repository
from codeatlas.services.ingestion import git_loader
from codeatlas.services.ingestion.git_loader import GitRepositoryLoader, repo_cache_url
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.state.repo_state_store import RepoStateStore

URL = "https://github.com/example/demo"
SHA_A = "a" * 40
SHA_B = "b" * 40


class CommitLoader(FakeLoader):
    """FakeLoader whose remote reports a commit, like git ls-remote."""

    def __init__(self, base: Path, head: str | None = SHA_A) -> None:
        super().__init__(base)
        self.head = head
        self.head_calls = 0

    def remote_head(self, repo_url: str) -> str | None:
        self.head_calls += 1
        return self.head

    def load(self, repo_url: str) -> Repository:
        return replace(super().load(repo_url), commit=self.head or "")


def _env(tmp_path, head=SHA_A, config=CONFIG) -> Env:
    return Env(tmp_path, loader=CommitLoader(tmp_path / "repos", head), config=config)


def _analyze(env: Env, url: str = URL):
    resp = env.client.post("/analyze-repo", json={"repo_url": url})
    assert resp.status_code == 200
    return resp


def _stages(resp) -> set[str]:
    return {part.split(";")[0] for part in resp.headers["Server-Timing"].split(", ")}


def test_same_repo_and_commit_reuses_the_index(tmp_path):
    env = _env(tmp_path)
    first = _analyze(env)
    second = _analyze(env)
    assert second.json() == first.json()
    assert len(env.loader.loaded) == 1
    assert "clone" not in _stages(second)
    assert {"remote_head", "cache_lookup"} <= _stages(second)


@pytest.mark.parametrize("variant", [URL + ".git", URL.upper().replace("HTTPS", "https"), URL + "/tree/main"])
def test_other_spellings_of_the_url_hit(tmp_path, variant):
    env = _env(tmp_path)
    first = _analyze(env)
    assert _analyze(env, variant).json()["repository_id"] == first.json()["repository_id"]


def test_index_built_with_other_embedding_settings_is_not_reused(tmp_path):
    env = _env(tmp_path)
    first = _analyze(env)
    assert env.state.get(first.json()["repository_id"]).index_format == env.index.index_format
    env.index = CodeIndexService(
        embedder=HashEmbeddingService(subtokens=True),
        retriever=FaissCodeRetriever(base_dir=str(env.index_dir)),
    )
    second = _analyze(env)
    assert second.json()["repository_id"] != first.json()["repository_id"]
    assert "clone" in _stages(second)
    # Saved with the new format, so the next request hits again.
    assert _analyze(env).json()["repository_id"] == second.json()["repository_id"]


def test_state_saved_before_index_formats_is_not_reused(tmp_path):
    env = _env(tmp_path)
    first = _analyze(env)
    repo_id = first.json()["repository_id"]
    env.state.save(repo_id, replace(env.state.get(repo_id), index_format=""))
    assert _analyze(env).json()["repository_id"] != repo_id


def test_new_commit_misses_and_indexes_again(tmp_path):
    env = _env(tmp_path)
    first = _analyze(env)
    env.loader.head = SHA_B
    second = _analyze(env)
    assert second.json()["repository_id"] != first.json()["repository_id"]
    assert len(env.loader.loaded) == 2
    # And the new commit is now cached.
    assert _analyze(env).json()["repository_id"] == second.json()["repository_id"]


def test_different_repo_misses(tmp_path):
    env = _env(tmp_path)
    first = _analyze(env)
    assert _analyze(env, "https://github.com/example/other").json()["repository_id"] != first.json()["repository_id"]


def test_unknown_remote_commit_always_indexes(tmp_path):
    env = _env(tmp_path, head=None)
    _analyze(env)
    _analyze(env)
    assert len(env.loader.loaded) == 2


def test_missing_clone_folder_or_index_misses(tmp_path):
    env = _env(tmp_path)
    repo_id = _analyze(env).json()["repository_id"]
    git_loader._remove_dir(Path(env.state.get(repo_id).root_path))
    second = _analyze(env).json()["repository_id"]
    assert second != repo_id
    env.index.discard(second)
    assert _analyze(env).json()["repository_id"] not in (repo_id, second)


def test_cache_can_be_turned_off(tmp_path):
    env = _env(tmp_path, config=replace(CONFIG, index_cache_enabled=False))
    _analyze(env)
    _analyze(env)
    assert env.loader.head_calls == 0
    assert len(env.loader.loaded) == 2


def test_cache_survives_a_restart_when_the_disk_does(tmp_path):
    env = _env(tmp_path)
    repo_id = _analyze(env).json()["repository_id"]
    # A new process: state and index are read back from disk.
    env.state = RepoStateStore(base_dir=str(tmp_path / "state"))
    env.index = CodeIndexService(
        embedder=HashEmbeddingService(), retriever=FaissCodeRetriever(base_dir=str(tmp_path / "indexes"))
    )
    assert env.state.get(repo_id).commit == SHA_A
    assert _analyze(env).json()["repository_id"] == repo_id
    assert len(env.loader.loaded) == 1


def test_invalid_url_is_rejected_before_any_git_call(tmp_path):
    env = _env(tmp_path)
    resp = env.client.post("/analyze-repo", json={"repo_url": "https://example.com/a/b"})
    assert resp.status_code == 400
    assert env.loader.head_calls == 0


def test_repo_cache_url_normalizes():
    assert repo_cache_url("https://GitHub.com/Pallets/Flask.git/") == "https://github.com/pallets/flask"


def test_clone_records_the_checked_out_commit_and_remote_head_matches(tmp_path, monkeypatch):
    src = tmp_path / "src"
    url = _make_source_repo(src, {"a.py": "x = 1\n"}, allow_filter=True)
    expected = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"))
    assert loader._clone(url, tmp_path / "clones" / str(uuid.uuid4())) == expected.strip()
    # Local file:// URLs are not on an allowed host; skip only that check.
    monkeypatch.setattr(git_loader, "validate_repo_url", lambda u: u)
    assert loader.remote_head(url) == expected.strip()
    assert loader.remote_head((tmp_path / "missing").as_uri()) is None


def test_old_state_files_without_a_commit_never_hit(tmp_path):
    env = _env(tmp_path)
    repo_id = _analyze(env).json()["repository_id"]
    state = env.state.get(repo_id)
    env.state.save(repo_id, replace(state, commit=""))
    assert _analyze(env).json()["repository_id"] != repo_id


def test_a_repo_seen_for_the_first_time_skips_the_remote_lookup(tmp_path):
    env = _env(tmp_path)
    resp = _analyze(env)
    assert env.loader.head_calls == 0
    assert "remote_head" not in _stages(resp)
    _analyze(env, "https://github.com/example/other")
    assert env.loader.head_calls == 0
