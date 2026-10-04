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

### Fix: per-client and global request limits on clone and LLM endpoints
- Value: over a limit, /analyze-repo, /ask, /ask/stream, /explain and /generate-code return 429 with a plain-English `detail` ("…Please try again in N seconds.") and a `Retry-After` header that browser code can read through CORS. Browsing endpoints (/repos, /files, /search, …) are not limited
- Command: `python -m pytest tests/test_rate_limit.py`, plus a manual run of `uvicorn codeatlas.app.main:app --no-proxy-headers` with CODEATLAS_LLM_PER_CLIENT_PER_MINUTE=1 (second POST /ask returned 429, `retry-after: 60`, `access-control-expose-headers: Retry-After`)
- Dataset/repo: none (in-process TestClient and a local server)
- Date: 2026-10-03
- Commit: 04ad86f
- Notes: defaults per minute: clones 10 per client / 30 global / 3 at once; LLM 60 per client / 300 global; at most 10,000 clients tracked. The client is the direct connection address unless CODEATLAS_TRUSTED_PROXY_HOPS=N, which takes the entry N from the right of X-Forwarded-For. The startup log states the mode without addresses. Limits live in one process: they reset on restart and multiply with the number of uvicorn workers. The frontend needed no change: chat and indexing errors already show `detail`

### Fix: admin/debug endpoints require an API key; the website's endpoints stay open
- Value: without a valid `X-API-Key`, every admin/debug endpoint returns 401. With no key configured on the server, they return 503 (fail closed). Every endpoint the website calls answers without a key
- Command: `python -m pytest tests/test_auth.py`, plus a manual run of `uvicorn codeatlas.app.main:app` with a throwaway key: anonymous GET /repos and /eval/stats returned 200; anonymous GET /metrics, /docs, /openapi.json and POST /search, /dependencies returned 401; the same GETs with the key returned 200; the CORS preflight for /ask returned 200
- Dataset/repo: none (in-process TestClient and a local server)
- Date: 2026-10-03
- Commit: 144ae11
- Notes: an anonymous visitor **can**: index a repo (POST /analyze-repo), chat (POST /ask, /ask/stream), list repos (GET /repos), browse files and read file contents (POST /files, /files/content), view the overview (POST /repo-overview), view the dependency graph (POST /dependencies/graph), and view the eval page stats (GET /eval/stats). Clone and LLM limits from the previous entry still apply. An anonymous visitor **cannot**: call /explain, /search, /generate-code or POST /dependencies; read /metrics; or open /docs, /docs/oauth2-redirect, /redoc or /openapi.json. The key is read from CODEATLAS_API_KEY and is never sent by the frontend; `codementor-ui` is unchanged, and a test checks that `lib/api.ts` only calls the open endpoints and sends no key. CODEATLAS_AUTH_ENABLED now defaults to true; setting it to false opens everything (local development only). docker-compose.yml and docker-compose.monitoring.yml still set it to false, so Prometheus can scrape /metrics without a key there. The Streamlit tool already has an API key field, so it needs the key for its admin calls

### Fix: landing page says plainly what the app does
- Value: the landing subtitle, page `<title>` and meta description now read "Paste a public GitHub repository and ask questions about its code, with answers that cite the files they came from." The filler lines "Agentic codebase intelligence for master developers" (meta description) and "Professional AI Developer Environment" (footer) are gone
- Command: `python -m pytest`, `ruff check codeatlas tests`, `ruff format --check codeatlas tests`, and in `codementor-ui`: `npm run lint` (0 errors; the same 28 warnings as before the change) and `npm run build`
- Dataset/repo: none (copy change)
- Date: 2026-10-03
- Commit: 5ea7c51
- Notes: the copy claims only what the live chat does today. The website streams answers through /ask/stream, which runs retrieval then the mentor agent, not the full planner/validator pipeline. Citations come from retrieval, so an answer with no matching code has no citations. The workspace welcome modal still describes a planner and "multi-agent AI"; that is outside this change
