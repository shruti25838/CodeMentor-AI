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

### Fix: unknown repository ids get a 404 before any language model call
- Value: /ask, /ask/stream, /explain and /generate-code return 404 with "This repository is not indexed on the server. It may have been removed when the server restarted. Go back to the home page and index it again." The model is never called. /ask/stream returns this as a JSON 404 before the stream opens, not as a 200 with an error event. Requests without a repo id (general chat on /ask and /ask/stream) are unchanged
- Command: `python -m pytest tests/test_unknown_repo.py` (fails 5 of 9 without the fix), full `python -m pytest`, `ruff check`, `ruff format --check`, and in `codementor-ui`: `npm run lint` (0 errors, same 28 warnings as before) and `npm run build`. Manual: local `uvicorn` with an empty state folder and fake LLM keys returned 404 for all four endpoints with an unknown id, so no model call was attempted
- Dataset/repo: none (in-process TestClient with stand-ins that record model calls, and a local server)
- Date: 2026-10-03
- Commit: 990ef78
- Notes: "exists" means the repo is in the repo state store, which /analyze-repo fills right after cloning. The LLM client object is still built during dependency setup before the check; building it makes no model call, but on a server with no LLM key configured that setup fails first with a 500, as before this change. The frontend needed no change: the chat already shows a non-200 response's `detail` as "Error: …"

### Feature: "Try an example" button on the landing page
- Value: one click indexes https://github.com/pallets/itsdangerous through the existing POST /analyze-repo flow, then opens the workspace with three suggested questions ("How does Signer create and check a signature?", "What is the difference between Serializer and URLSafeTimedSerializer?", "How does the code detect that a signed value has expired?"). Clicking a suggestion fills the chat input. On a local server the repo indexed in about 3 s: 15 parsed files, 63 dependency edges
- Command: `python -m pytest`, `ruff check codeatlas tests`, `ruff format --check codeatlas tests`, and in `codementor-ui`: `npm run lint` (0 errors, same 28 warnings as before) and `npm run build`. Manual: `curl -X POST localhost:8765/analyze-repo -d '{"repo_url":"https://github.com/pallets/itsdangerous"}'` against a local `uvicorn`, then GET /repos
- Dataset/repo: https://github.com/pallets/itsdangerous (public, BSD-3-Clause, maintained by Pallets)
- Date: 2026-10-03
- Commit: b6b0702
- Notes: no new endpoint. Each click is a normal clone, so it counts against the clone rate limit; /repos returns no URL and repo ids are random, so an earlier copy cannot be reused reliably. Suggestions are saved in localStorage together with the repo id, and the chat shows them only while that repo is the current one and before the first message. The button still uses the current landing flow, which opens the workspace after a fixed animation rather than when indexing finishes (item 4), and the first click after the server sleeps can hit a cold start (item 3). Answer quality for the suggested questions was not checked here because no LLM key is configured locally

### Feature: cold-start message for a sleeping backend; only safe reads are retried
- Value: a request with no answer after 4 s shows "Waking up the server, this can take up to a minute" in a banner at the top of every page, on the landing page's indexing spinner (including the "Try an example" request) and in the chat's thinking spinner. Read-only requests (GET /repos, GET /eval/stats, POST /files, /files/content, /repo-overview, /dependencies/graph) retry network errors and 502/503/504 with backoff (1, 2, 4, then every 8 s) for up to 75 s. POST /analyze-repo, /ask and /ask/stream are sent exactly once and never retried; any HTTP response they get, such as the backend's own 504 for a slow clone, reaches the page unchanged. If the server cannot be reached, the banner and the caller show "Could not reach the server. It may still be starting up. Please wait a minute and try again."
- Command: in `codementor-ui`: `npm test` (8 passed: retry and recovery, gateway retries, deadline, no retry for send-once requests on network and gateway errors, waking state on a slow send-once request, error text, and a check that api.ts sends indexing and chat once), `npm run lint` (0 errors, same 28 warnings as before), `npm run build`; plus `python -m pytest`, `ruff check codeatlas tests`, `ruff format --check codeatlas tests`
- Dataset/repo: none (mocked fetch and timers)
- Date: 2026-10-03
- Commit: 572c561
- Notes: no new dependencies. `npm test` uses Node's built-in test runner with TypeScript type stripping, which needs Node 22.6 or newer; frontend CI pins Node 20 and does not run it. Test files are excluded from the Next.js type check. The POST read endpoints change nothing on the server, so repeating them is safe. Indexing and chat have no client time limit, because the server may need a minute to wake plus up to the clone time limit

### Fix: /analyze-repo answers only when indexing is done; honest progress on the landing page
- Value: POST /analyze-repo now clones, parses, builds the dependency graph and builds the search index before answering, and returns `indexing_status: "ready"`. The repo is saved, and appears in /repos, only after all steps succeed. Indexing has a time limit (CODEATLAS_INDEX_TIMEOUT_SECONDS, default 120), checked between files and between embedding batches of 64. On timeout it returns 504 "Indexing this repository took too long, so it was stopped and nothing was saved. Try a smaller repository."; any other failure returns 500 "Indexing this repository failed, so nothing was saved. Please try again later." In both cases the clone folder and any partial index are deleted. The landing page shows a spinner with "Cloning, parsing and indexing <repo>" instead of the timed progress bar, and opens the workspace only after a success; on failure it stays on the landing page with the error. The "Waking up the server" message now shows only until the server has answered any request in the page session, so normal long indexing shows the indexing text
- Command: `python -m pytest` (130 passed; `tests/test_analyze_indexing.py` covers success and response shape, indexing failure, parse failure, timeout during embedding, timeout after the last batch, no repo visible in the state store or /repos while indexing, FAISS removal, and the config default), `ruff check`, `ruff format --check`, and in `codementor-ui`: `npm test` (10 passed), `npm run lint` (0 errors, same 28 warnings as before), `npm run build`. Manual: local `uvicorn`; https://github.com/pallets/itsdangerous returned 200 `"indexing_status":"ready"` in 2.7 s; https://github.com/pallets/flask with CODEATLAS_INDEX_TIMEOUT_SECONDS=0 returned the 504, and afterwards no clone folder, index files or state for it remained
- Dataset/repo: small Python project written by the test's fake loader; https://github.com/pallets/itsdangerous and https://github.com/pallets/flask for the manual runs
- Date: 2026-10-03
- Commit: e249883
- Notes: the response keeps the same four fields (`repository_id`, `file_count`, `dependency_edges`, `indexing_status`); only the status value changed from "queued" to "ready". The time limit covers the indexing step; cloning has its own limits, and parsing and graph building have none but are bounded by the file-count and size limits. The request holds a clone slot until indexing finishes (measured locally: about 0.1 s of indexing for itsdangerous, 2.5 s for flask, with the default hash embedder). `CodeRetriever` gains an abstract `remove`. "Previously indexed", "Open Coding Workspace" and the welcome popup are unchanged

### Feature: citations open the file preview at the cited lines
- Value: each citation in a chat answer (still inside the "Citations" section) is a button that opens the existing file preview. For a function citation such as `src/itsdangerous/url_safe.py (lines 72-76)`, the preview highlights lines 72-76, scrolls them into view and shows "cited: lines 72-76" in the header. A file citation opens the whole file. Clicking a file in the context panel's retrieved list now also opens at its cited lines (before, the line range was dropped)
- Command: in `codementor-ui`: `npm test` (16 passed, including 6 for the citation parser), `npx tsc --noEmit`, `npm run lint` (0 errors, same 28 warnings as before), `npm run build`; plus `python -m pytest`, `ruff check`, `ruff format --check`. Manual: retrieved the citations for two of the example's suggested questions from a local index of pallets/itsdangerous, parsed them with the same rules as `lib/citations.ts`, and posted each path to /files/content on a local `uvicorn`: 10 of 10 returned 200, with every cited range inside the file's line count
- Dataset/repo: https://github.com/pallets/itsdangerous
- Date: 2026-10-03
- Commit: 6aa916f
- Notes: no backend change and no new endpoint; the preview uses POST /files/content, which the website already called. Citation format parsed: "path (lines A-B) | snippet", "path | snippet" or "path". A cited path that is not relative to the repo root (only if the clone folder is outside `.codeatlas/repos`) gets the preview's existing "File not found" error
