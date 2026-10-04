import contextlib
import os
import shutil
import signal
import stat
import subprocess
import tempfile
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


ALLOWED_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")
UNSUPPORTED_HOST_MESSAGE = "Only repositories on github.com, gitlab.com or bitbucket.org are supported."

# Web-UI path segments that follow owner/repo: GitHub /tree, /blob; GitLab /-/tree; Bitbucket /src.
_WEB_PATH_MARKERS = {"tree", "blob", "src", "-"}


def _normalize_repo_url(url: str) -> str:
    """Convert web URLs to git-cloneable URLs (strip /tree/branch, /-/blob/..., /src/..., etc)."""
    s = url.strip().rstrip("/")
    head, sep, path = s.partition("://")
    if not sep:
        return s
    host, _, path = path.partition("/")
    parts = path.split("/")
    # Only look past owner/repo, so an owner or repo literally named "src" or "tree" is kept.
    for i in range(2, len(parts)):
        if parts[i] in _WEB_PATH_MARKERS:
            parts = parts[:i]
            break
    return f"{head}://{host}/{'/'.join(parts)}".rstrip("/")


def validate_repo_url(url: str) -> str:
    """Return a normalized clone URL on an allowed host, or raise RepoCloneError."""
    s = _normalize_repo_url(str(url))
    parsed = urlparse(s)
    path_parts = [p for p in parsed.path.split("/") if p]
    try:
        port = parsed.port
    except ValueError:
        port = -1
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or port is not None
        or len(path_parts) < 2
        or any(c.isspace() for c in s)
    ):
        raise RepoCloneError(INVALID_URL_MESSAGE, status_code=400)
    if parsed.hostname not in ALLOWED_HOSTS:
        raise RepoCloneError(UNSUPPORTED_HOST_MESSAGE, status_code=400)
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
    for _ in range(5):
        if not path.exists():
            return
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(root, name), stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            # A just-killed git child can hold file handles open for a moment on Windows.
            time.sleep(0.2)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            # os.stat, not DirEntry.stat: on Windows the cached listing can lag a file still being written.
            with contextlib.suppress(OSError):
                total += os.stat(os.path.join(root, name)).st_size
    return total


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill git and its helpers (git-remote-https, index-pack), which would otherwise keep downloading."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    proc.wait()


class GitRepositoryLoader(RepositoryLoader):
    POLL_SECONDS = 0.2

    def __init__(
        self,
        base_dir: str | None = None,
        timeout_seconds: float = 60,
        max_bytes: int = 100 * 1024 * 1024,
        max_files: int = 5000,
        max_download_bytes: int | None = None,
    ) -> None:
        self.base_dir = Path(base_dir or ".codeatlas/repos").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.max_download_bytes = max_download_bytes if max_download_bytes is not None else 2 * max_bytes

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
        """Clone in stages so oversized repos are rejected as early as possible.

        1. Partial clone (--filter=blob:none): commits and trees only, no file contents.
           Count files with ls-tree *without* -l (sizes need the blobs and would trigger
           lazy fetches) and apply the file-count limit.
        2. Full shallow clone without checkout; sum blob sizes with ls-tree -l and apply
           the size limit before anything is written to the working tree.
        3. Check out.

        Fallback: a host that doesn't support filtering ignores the filter (git only warns),
        so step 1 is already a full clone and step 2 reuses it instead of downloading again.

        Every clone runs under the download cap (size of the clone folder), and all steps
        share one time budget. On any failure the clone folder is deleted.
        """
        deadline = time.monotonic() + self.timeout_seconds
        try:
            self._git_clone(["--filter=blob:none"], repo_url, repo_dir, deadline)
            self._check_file_count(self._git(["-C", str(repo_dir), "ls-tree", "-r", "-z", "HEAD"], deadline))
            if self._has_missing_objects(repo_dir, deadline):
                _remove_dir(repo_dir)
                self._git_clone([], repo_url, repo_dir, deadline)
            listing = self._git(["-C", str(repo_dir), "ls-tree", "-r", "-l", "-z", "HEAD"], deadline)
            self._check_limits(listing)
            self._git(["-C", str(repo_dir), "checkout", "-q", "HEAD"], deadline)
        except BaseException:
            _remove_dir(repo_dir)
            raise

    def _git_clone(self, extra: list[str], repo_url: str, repo_dir: Path, deadline: float) -> None:
        # Empty credential.helper + no prompts: private repos fail fast instead of hanging.
        args = ["-c", "credential.helper=", "clone", "--depth", "1", "--no-checkout", *extra, "--", repo_url]
        self._git([*args, str(repo_dir)], deadline, watch_dir=repo_dir)

    def _has_missing_objects(self, repo_dir: Path, deadline: float) -> bool:
        """True if the partial clone really omitted blobs; False if the host ignored the filter.

        --missing=print lists absent objects with a "?" prefix instead of fetching them. (git
        still sets remote.origin.promisor when the filter is ignored, so that config can't be used.)
        """
        out = self._git(["-C", str(repo_dir), "rev-list", "--objects", "--missing=print", "HEAD"], deadline)
        return any(line.startswith("?") for line in out.splitlines())

    def _git(self, args: list[str], deadline: float, watch_dir: Path | None = None) -> str:
        """Run git, killing it if the deadline passes or watch_dir grows past the download cap."""
        if deadline - time.monotonic() <= 0:
            raise self._timeout_error()
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
        # Temp files, not pipes: a full pipe would block git while we poll.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(
                    ["git", *args],
                    stdout=out,
                    stderr=err,
                    env=env,
                    start_new_session=os.name != "nt",
                )
            except FileNotFoundError:
                raise RepoCloneError("The server can't clone repositories right now (git is not installed).", 500)
            try:
                while True:
                    try:
                        proc.wait(timeout=self.POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if time.monotonic() >= deadline:
                        raise self._timeout_error()
                    if watch_dir is not None and _dir_size(watch_dir) > self.max_download_bytes:
                        raise self._download_cap_error()
            finally:
                _kill_tree(proc)
            # Check once more after exit: a fast clone can finish between polls.
            if watch_dir is not None and _dir_size(watch_dir) > self.max_download_bytes:
                raise self._download_cap_error()
            out.seek(0)
            err.seek(0)
            stdout = out.read().decode("utf-8", errors="replace")
            stderr = err.read().decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise _classify_git_error(stderr)
        return stdout

    def _timeout_error(self) -> RepoCloneError:
        return RepoCloneError(
            f"Cloning took longer than {self.timeout_seconds:g} seconds and was stopped. "
            "The repository may be too large; try a smaller one.",
            status_code=504,
        )

    def _download_cap_error(self) -> RepoCloneError:
        return RepoCloneError(
            f"This repository is too large to download (over {self.max_download_bytes / (1024 * 1024):g} MB). "
            "Try a smaller repository.",
            status_code=413,
        )

    def _too_many_files_error(self, file_count: int) -> RepoCloneError:
        return RepoCloneError(
            f"This repository has {file_count:,} files; the limit is {self.max_files:,}. Try a smaller repository.",
            status_code=413,
        )

    def _check_file_count(self, ls_tree_output: str) -> None:
        # Each -z entry (no -l): "<mode> <type> <sha>\t<path>"; submodules are type "commit".
        file_count = sum(1 for entry in ls_tree_output.split("\0") if entry.partition("\t")[0].split()[1:2] == ["blob"])
        if file_count > self.max_files:
            raise self._too_many_files_error(file_count)

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
            raise self._too_many_files_error(file_count)
        if total_bytes > self.max_bytes:
            raise RepoCloneError(
                f"This repository is {total_bytes / (1024 * 1024):.1f} MB; the limit is "
                f"{self.max_bytes / (1024 * 1024):g} MB. Try a smaller repository.",
                status_code=413,
            )
