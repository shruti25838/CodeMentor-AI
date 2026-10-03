import subprocess

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.di import get_repository_loader
from codeatlas.app.main import create_app
from codeatlas.services.ingestion.git_loader import (
    NOT_FOUND_MESSAGE,
    GitRepositoryLoader,
    RepoCloneError,
    _classify_git_error,
    validate_repo_url,
)


def _make_source_repo(path, files: dict[str, str]):
    """Create a local git repo with the given files committed; return its file:// URL."""
    path.mkdir()
    for name, content in files.items():
        (path / name).write_text(content)
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    return path.as_uri()


@pytest.mark.parametrize(
    "url",
    ["", "not a url", "ftp://github.com/a/b", "https://github.com", "https://github.com/onlyowner", "-oProxy=x"],
)
def test_invalid_urls_rejected(url):
    with pytest.raises(RepoCloneError) as exc:
        validate_repo_url(url)
    assert exc.value.status_code == 400


def test_credentials_in_url_rejected():
    with pytest.raises(RepoCloneError):
        validate_repo_url("https://user:token@github.com/a/b")


def test_github_web_url_normalized():
    assert validate_repo_url("https://github.com/a/b/tree/main/src") == "https://github.com/a/b"


@pytest.mark.parametrize(
    "stderr",
    [
        "remote: Repository not found.\nfatal: repository 'https://github.com/a/b/' not found",
        "fatal: could not read Username for 'https://github.com': terminal prompts disabled",
    ],
)
def test_missing_or_private_repo_gets_friendly_message(stderr):
    err = _classify_git_error(stderr)
    assert err.status_code == 404
    assert err.message == NOT_FOUND_MESSAGE


def test_clone_within_limits_checks_out_files(tmp_path):
    url = _make_source_repo(tmp_path / "src", {"a.py": "x = 1\n", "b.py": "y = 2\n"})
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_files=10, max_bytes=10_000)
    dest = tmp_path / "clones" / "ok"
    loader._clone(url, dest)
    assert (dest / "a.py").read_text().strip() == "x = 1"


def test_too_many_files_rejected_and_cleaned_up(tmp_path):
    url = _make_source_repo(tmp_path / "src", {f"f{i}.py": "x\n" for i in range(5)})
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_files=3)
    dest = tmp_path / "clones" / "big"
    with pytest.raises(RepoCloneError) as exc:
        loader._clone(url, dest)
    assert exc.value.status_code == 413
    assert "5 files" in exc.value.message
    assert not dest.exists()


def test_too_large_rejected_and_cleaned_up(tmp_path):
    url = _make_source_repo(tmp_path / "src", {"big.txt": "a" * 5000})
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_bytes=1000)
    dest = tmp_path / "clones" / "big"
    with pytest.raises(RepoCloneError) as exc:
        loader._clone(url, dest)
    assert exc.value.status_code == 413
    assert not dest.exists()


def test_timeout_gives_friendly_error(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", fake_run)
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), timeout_seconds=5)
    with pytest.raises(RepoCloneError) as exc:
        loader.load("https://github.com/a/b")
    assert exc.value.status_code == 504
    assert "5 seconds" in exc.value.message


def test_analyze_endpoint_returns_readable_detail(tmp_path):
    app = create_app()
    app.dependency_overrides[get_repository_loader] = lambda: GitRepositoryLoader(base_dir=str(tmp_path))
    resp = TestClient(app).post("/analyze-repo", json={"repo_url": "not a url"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)
    assert "valid repository URL" in resp.json()["detail"]
