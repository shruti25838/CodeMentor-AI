import contextlib
import os
import re
import shutil
import stat
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from codeatlas.models.repository import Repository
from codeatlas.services.ingestion.interfaces import RepositoryLoader


class RepoCloneError(Exception):
    """A clone failure with a message that is safe and friendly to show the user."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


INVALID_URL_MESSAGE = "That doesn't look like a valid repository URL. Use a link like https://github.com/owner/repo."
NOT_FOUND_MESSAGE = (
    "Couldn't access that repository. Check the URL is correct and that the repository is public "
    "(private repositories are not supported)."
)

# git stderr fragments that mean "doesn't exist, or needs credentials we don't have".
_NOT_FOUND_MARKERS = (
    "repository not found",
    "could not read username",
    "terminal prompts disabled",
    "authentication failed",
    "could not be found",
    "returned error: 401",
    "returned error: 403",
    "returned error: 404",
)
_BAD_HOST_MARKERS = ("could not resolve host", "unable to access", "failed to connect")


def _normalize_repo_url(url: str) -> str:
    """Convert GitHub web URLs to git-cloneable URLs (strip /tree/branch, /blob/..., etc)."""
    s = url.strip().rstrip("/")
    # Remove GitHub path suffixes: /tree/main, /tree/master, /blob/main/file, etc.
    s = re.sub(r"/(tree|blob)/[^/]+(/.*)?$", "", s)
    return s


def validate_repo_url(url: str) -> str:
    """Return a normalized http(s) clone URL, or raise RepoCloneError for anything else."""
    s = _normalize_repo_url(str(url))
    parsed = urlparse(s)
    path_parts = [p for p in parsed.path.split("/") if p]
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or len(path_parts) < 2
        or any(c.isspace() for c in s)
    ):
        raise RepoCloneError(INVALID_URL_MESSAGE, status_code=400)
    return s


def _classify_git_error(stderr: str) -> RepoCloneError:
    text = stderr.lower()
    if any(m in text for m in _NOT_FOUND_MARKERS):
        return RepoCloneError(NOT_FOUND_MESSAGE, status_code=404)
    if any(m in text for m in _BAD_HOST_MARKERS):
        return RepoCloneError(
            "Couldn't reach that repository's host. Check the URL and try again.",
            status_code=400,
        )
    return RepoCloneError("Cloning the repository failed. Please try again later.", status_code=502)


def _remove_dir(path: Path) -> None:
    """Best-effort delete; git marks object files read-only, which blocks rmtree on Windows."""
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(root, name), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    shutil.rmtree(path, ignore_errors=True)


class GitRepositoryLoader(RepositoryLoader):
    def __init__(
        self,
        base_dir: str | None = None,
        timeout_seconds: float = 60,
        max_bytes: int = 100 * 1024 * 1024,
        max_files: int = 5000,
    ) -> None:
        self.base_dir = Path(base_dir or ".codeatlas/repos").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_files = max_files

    def load(self, repo_url: str) -> Repository:
        repo_url_str = str(repo_url)
        clone_url = validate_repo_url(repo_url_str)
        repo_id = str(uuid.uuid4())
        repo_dir = self.base_dir / repo_id
        self._clone(clone_url, repo_dir)
        name = clone_url.rstrip("/").removesuffix(".git").split("/")[-1]
        return Repository(
            repo_id=repo_id,
            name=name,
            url=repo_url_str,
            root_path=str(repo_dir),
            ingested_at=datetime.now(UTC),
        )

    def _clone(self, repo_url: str, repo_dir: Path) -> None:
        """Shallow-clone without checkout, check size/file limits from the tree, then check out.

        The whole sequence shares one time budget. On any failure the partial clone is deleted.
        """
        deadline = time.monotonic() + self.timeout_seconds
        try:
            self._git(
                # Empty credential.helper + no prompts: private repos fail fast instead of hanging.
                ["-c", "credential.helper=", "clone", "--depth", "1", "--no-checkout", "--", repo_url, str(repo_dir)],
                deadline,
            )
            listing = self._git(["-C", str(repo_dir), "ls-tree", "-r", "-l", "-z", "HEAD"], deadline)
            self._check_limits(listing)
            self._git(["-C", str(repo_dir), "checkout", "-q", "HEAD"], deadline)
        except BaseException:
            _remove_dir(repo_dir)
            raise

    def _git(self, args: list[str], deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise self._timeout_error()
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=remaining,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise self._timeout_error()
        except FileNotFoundError:
            raise RepoCloneError("The server can't clone repositories right now (git is not installed).", 500)
        if result.returncode != 0:
            raise _classify_git_error(result.stderr or "")
        return result.stdout

    def _timeout_error(self) -> RepoCloneError:
        return RepoCloneError(
            f"Cloning took longer than {self.timeout_seconds:g} seconds and was stopped. "
            "The repository may be too large; try a smaller one.",
            status_code=504,
        )

    def _check_limits(self, ls_tree_output: str) -> None:
        # Each -z entry: "<mode> <type> <sha> <size>\t<path>"; submodules are type "commit" with size "-".
        file_count = 0
        total_bytes = 0
        for entry in ls_tree_output.split("\0"):
            meta, _, _path = entry.partition("\t")
            fields = meta.split()
            if len(fields) != 4 or fields[1] != "blob":
                continue
            file_count += 1
            total_bytes += int(fields[3]) if fields[3].isdigit() else 0
        if file_count > self.max_files:
            raise RepoCloneError(
                f"This repository has {file_count:,} files; the limit is {self.max_files:,}. Try a smaller repository.",
                status_code=413,
            )
        if total_bytes > self.max_bytes:
            raise RepoCloneError(
                f"This repository is {total_bytes / (1024 * 1024):.1f} MB; the limit is "
                f"{self.max_bytes / (1024 * 1024):g} MB. Try a smaller repository.",
                status_code=413,
            )
