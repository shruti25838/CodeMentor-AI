# Results

Every number or claim here comes from a command that can be re-run.

### Fix: default embedding provider is hash
- Value: a fresh setup no longer crashes on first index
- Command: `python -m pytest tests/test_config.py`
- Dataset/repo: none (config defaults with env vars unset)
- Date: 2026-10-03
- Commit: e4ba06e
- Notes: Render also sets CODEATLAS_EMBEDDING_PROVIDER=hash explicitly

### Fix: /files/content only returns files inside the indexed repo
- Value: absolute paths, ../ traversal and missing files all return 404; real in-repo files return 200
- Command: `python -m pytest tests/test_safe_join.py`
- Dataset/repo: pytest temp directories, plus a manual check on a small public repo via /docs
- Date: 2026-10-03
- Commit: 219b062
- Notes: symlink test skips on Windows (needs privileges) and runs on Linux CI; project needs Python 3.11+

### Fix: repo cloning has time, size and file-count limits with friendly errors
- Value: invalid URLs get 400, missing/private repos 404, repos over the size or file limit 413, and slow clones 504. Each returns a plain-English `detail` string, and any partial clone is deleted
- Command: `python -m pytest tests/test_git_loader.py`
- Dataset/repo: small local git repos created in pytest temp directories, plus a manual check against a nonexistent public GitHub URL and an unresolvable host
- Date: 2026-10-03
- Commit: c52e941
- Notes: limits default to 60 s / 100 MB / 5000 files (CODEATLAS_CLONE_TIMEOUT_SECONDS, CODEATLAS_MAX_REPO_MB, CODEATLAS_MAX_REPO_FILES). Size is checked from git ls-tree before checkout; the download itself is bounded only by the time limit

### Fix: only github.com, gitlab.com and bitbucket.org repository URLs are accepted
- Value: other hosts, lookalike hosts (github.com.evil.example, evilgithub.com, gist.github.com), localhost, raw IPs and non-default ports get a 400 with a plain-English message before any clone starts
- Command: `python -m pytest tests/test_git_loader.py`
- Dataset/repo: URL strings only (no network)
- Date: 2026-10-03
- Commit: fc26951
- Notes: hostnames must match exactly. GitLab `/-/tree/...` and Bitbucket `/src/...` web URLs are normalized to the clone URL

### Fix: file count is checked before file contents are downloaded, and the clone download is capped
- Value: a repo over the file-count limit is rejected after a partial clone that holds only commits and the file list (cpython at a 1,000-file limit: rejected in 1.0 s with only that partial clone run). A download that grows past the cap is stopped partway and deleted (cpython at a 20 MB cap: stopped after 7.2 s, folder removed)
- Command: `python -m pytest tests/test_git_loader.py` (local repos), plus manual `GitRepositoryLoader(...).load("https://github.com/python/cpython")` runs with low limits
- Dataset/repo: local git repos in pytest temp directories, with and without partial-clone support; public github.com/python/cpython for the manual runs
- Date: 2026-10-03
- Commit: 5956bd6
- Notes: what is limited: file count (before any file contents are downloaded, on hosts that support partial clone), bytes downloaded (CODEATLAS_MAX_DOWNLOAD_MB, default 2x CODEATLAS_MAX_REPO_MB, checked about every 0.2 s), total file size (CODEATLAS_MAX_REPO_MB, checked before anything is written to the working tree), and total time (CODEATLAS_CLONE_TIMEOUT_SECONDS). What is not limited: the size check still needs the contents downloaded first, so a repo under the download cap but over the size limit is downloaded, then deleted. The cap can be overshot by up to 0.2 s of transfer. On hosts without partial clone, the file count is only known after the full download, which is still bounded by the download cap and time limit. The cap counts the whole clone folder, including git's ~20 KB of template files
