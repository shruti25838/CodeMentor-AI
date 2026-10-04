import os
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.di import get_repository_loader
from codeatlas.app.main import create_app
from codeatlas.services.ingestion.git_loader import (
    NOT_FOUND_MESSAGE,
    UNSUPPORTED_HOST_MESSAGE,
    GitRepositoryLoader,
    RepoCloneError,
    _classify_git_error,
    validate_repo_url,
)


def _make_source_repo(path, files: dict[str, str | bytes], allow_filter: bool = False):
    """Create a local git repo with the given files committed; return its file:// URL.

    allow_filter=True lets clients partial-clone from it, like GitHub/GitLab; without it the
    filter is ignored, like a host that doesn't support partial clone.
    """
    path.mkdir()
    for name, content in files.items():
        if isinstance(content, bytes):
            (path / name).write_bytes(content)
        else:
            (path / name).write_text(content)
    git = ["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    if allow_filter:
        subprocess.run([*git, "config", "uploadpack.allowFilter", "true"], check=True)
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


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://github.com/a/b/tree/main/src", "https://github.com/a/b"),
        ("https://github.com/a/b.git", "https://github.com/a/b.git"),
        ("https://gitlab.com/group/sub/repo/-/tree/main", "https://gitlab.com/group/sub/repo"),
        ("https://bitbucket.org/team/repo/src/main/README.md", "https://bitbucket.org/team/repo"),
        # Owner/repo literally named like a web-path marker must survive.
        ("https://github.com/src/tree", "https://github.com/src/tree"),
    ],
)
def test_allowed_host_urls_normalized(url, expected):
    assert validate_repo_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a/b",
        "https://github.com.evil.example/a/b",
        "https://evilgithub.com/a/b",
        "https://gist.github.com/a/b",
        "http://localhost/a/b",
        "https://127.0.0.1/a/b",
    ],
)
def test_other_hosts_rejected_with_friendly_message(url):
    with pytest.raises(RepoCloneError) as exc:
        validate_repo_url(url)
    assert exc.value.status_code == 400
    assert exc.value.message == UNSUPPORTED_HOST_MESSAGE


def test_non_default_port_rejected():
    with pytest.raises(RepoCloneError) as exc:
        validate_repo_url("https://github.com:8443/a/b")
    assert exc.value.status_code == 400


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


class _CloneSpy:
    """Records the extra flags of each clone the loader runs."""

    def __init__(self, loader):
        self.calls = []
        original = loader._git_clone

        def spy(extra, *args, **kwargs):
            self.calls.append(list(extra))
            return original(extra, *args, **kwargs)

        loader._git_clone = spy


def test_partial_clone_path_checks_out_files(tmp_path):
    url = _make_source_repo(tmp_path / "src", {"a.py": "x = 1\n", "b.py": "y = 2\n"}, allow_filter=True)
    loader = GitRepositoryLoader(
        base_dir=str(tmp_path / "clones"), max_files=10, max_bytes=10_000, max_download_bytes=10**9
    )
    spy = _CloneSpy(loader)
    dest = tmp_path / "clones" / "ok"
    loader._clone(url, dest)
    # Filtered probe, then the full clone.
    assert spy.calls == [["--filter=blob:none"], []]
    assert (dest / "a.py").read_text().strip() == "x = 1"


def test_file_count_rejected_before_file_contents_downloaded(tmp_path):
    url = _make_source_repo(tmp_path / "src", {f"f{i}.py": "x\n" for i in range(5)}, allow_filter=True)
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_files=3)
    spy = _CloneSpy(loader)
    dest = tmp_path / "clones" / "big"
    with pytest.raises(RepoCloneError) as exc:
        loader._clone(url, dest)
    assert exc.value.status_code == 413
    assert "5 files" in exc.value.message
    # Only the filtered (no file contents) clone ran.
    assert spy.calls == [["--filter=blob:none"]]
    assert not dest.exists()


def test_fallback_when_host_ignores_filter_reuses_single_clone(tmp_path):
    # Without uploadpack.allowFilter the source ignores --filter, like a host without partial clone.
    url = _make_source_repo(tmp_path / "src", {"a.py": "x = 1\n"})
    loader = GitRepositoryLoader(
        base_dir=str(tmp_path / "clones"), max_files=10, max_bytes=10_000, max_download_bytes=10**9
    )
    spy = _CloneSpy(loader)
    dest = tmp_path / "clones" / "ok"
    loader._clone(url, dest)
    assert spy.calls == [["--filter=blob:none"]]
    assert (dest / "a.py").read_text().strip() == "x = 1"


def test_fallback_still_enforces_file_count(tmp_path):
    url = _make_source_repo(tmp_path / "src", {f"f{i}.py": "x\n" for i in range(5)})
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_files=3)
    dest = tmp_path / "clones" / "big"
    with pytest.raises(RepoCloneError) as exc:
        loader._clone(url, dest)
    assert exc.value.status_code == 413
    assert not dest.exists()


@pytest.mark.parametrize("allow_filter", [True, False])
def test_too_large_rejected_and_cleaned_up(tmp_path, allow_filter):
    url = _make_source_repo(tmp_path / "src", {"big.txt": "a" * 5000}, allow_filter=allow_filter)
    # Download cap set high so this exercises the ls-tree size check, not the cap.
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_bytes=1000, max_download_bytes=10**9)
    dest = tmp_path / "clones" / "big"
    with pytest.raises(RepoCloneError) as exc:
        loader._clone(url, dest)
    assert exc.value.status_code == 413
    assert "the limit is" in exc.value.message
    assert not dest.exists()


@pytest.mark.parametrize("allow_filter", [True, False])
def test_download_cap_stops_clone_and_cleans_up(tmp_path, allow_filter):
    # Random bytes don't compress, so the download really is ~2 MB.
    url = _make_source_repo(tmp_path / "src", {"blob.bin": os.urandom(2_000_000)}, allow_filter=allow_filter)
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), max_bytes=10**9, max_download_bytes=1_000_000)
    dest = tmp_path / "clones" / "big"
    with pytest.raises(RepoCloneError) as exc:
        loader._clone(url, dest)
    assert exc.value.status_code == 413
    assert "too large to download" in exc.value.message
    assert not dest.exists()


def test_timeout_kills_git_and_gives_friendly_error(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    def slow_popen(cmd, *args, **kwargs):
        # Stand in for a git command that never finishes; leave taskkill etc. alone.
        if cmd[0] == "git":
            cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", slow_popen)
    loader = GitRepositoryLoader(base_dir=str(tmp_path / "clones"), timeout_seconds=1)
    start = time.monotonic()
    with pytest.raises(RepoCloneError) as exc:
        loader.load("https://github.com/a/b")
    assert time.monotonic() - start < 10
    assert exc.value.status_code == 504
    assert "1 seconds" in exc.value.message


def test_analyze_endpoint_returns_readable_detail(tmp_path):
    app = create_app()
    app.dependency_overrides[get_repository_loader] = lambda: GitRepositoryLoader(base_dir=str(tmp_path))
    resp = TestClient(app).post("/analyze-repo", json={"repo_url": "not a url"})
    assert resp.status_code == 400
    assert isinstance(resp.json()["detail"], str)
    assert "valid repository URL" in resp.json()["detail"]


def test_analyze_endpoint_rejects_other_host(tmp_path):
    app = create_app()
    app.dependency_overrides[get_repository_loader] = lambda: GitRepositoryLoader(base_dir=str(tmp_path))
    resp = TestClient(app).post("/analyze-repo", json={"repo_url": "https://example.com/a/b"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == UNSUPPORTED_HOST_MESSAGE
