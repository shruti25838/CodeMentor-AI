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

### Feature: per-stage timing for indexing and chat
- Value: medians in ms on this Windows machine; cold = first request in a fresh process (3 processes), warm = later requests in the same process (6 runs). The model was **not timed**: no GROQ_API_KEY or OPENAI_API_KEY is set here, so the built-in stand-in answered instantly and `llm` below is only LangChain overhead

  | Indexing | clone | parse | graph | read_files | embed | index_write | save_state | total |
  |---|---|---|---|---|---|---|---|---|
  | itsdangerous cold | 1792 | 106 | 23 | 28 | 27 | 7 | 3 | 2013 |
  | itsdangerous warm | 1691 | 97 | 22 | 24 | 25 | 11 | 3 | 1893 |
  | flask cold | 2285 | 809 | 258 | 376 | 263 | 61 | 15 | 4034 |
  | flask warm | 2147 | 820 | 225 | 376 | 260 | 61 | 19 | 3895 |

  | Chat question (model excluded) | embed_query (2 calls) | search (2) | rerank (2) | llm stand-in (3 calls) | first token sent | total |
  |---|---|---|---|---|---|---|
  | itsdangerous cold | 0.4 | 3.5 | 5.7 | 11.0 | 27.0 | 37.0 |
  | itsdangerous warm | 0.3 | 0.4 | 4.7 | 3.1 | 12.2 | 25.5 |
  | flask cold | 0.3 | 4.1 | 7.1 | 7.9 | 24.9 | 34.4 |
  | flask warm | 0.3 | 0.6 | 6.5 | 3.0 | 15.5 | 32.8 |

  Clone is 89% of indexing time for itsdangerous and 55 to 57% for flask. Without the model, a chat question takes under 40 ms. App import and startup took 2.6 to 2.9 s per process
- Command: `python scripts/time_stages.py https://github.com/pallets/itsdangerous --runs 3` and `python scripts/time_stages.py https://github.com/pallets/flask --runs 3`, each run 3 times; plus `python -m pytest` (137 passed; `tests/test_timing.py` covers summing repeated stages, call counts, the no-timer case, thread-pool handoff, every indexing stage in Server-Timing, chat stages in the done event and no question text in logs), `ruff check`, `ruff format --check`, and in `codementor-ui`: `npm test` (16 passed), `npm run lint` (0 errors, same 28 warnings as before), `npm run build`
- Dataset/repo: https://github.com/pallets/itsdangerous (15 files, 63 edges) and https://github.com/pallets/flask (83 files, 432 edges), question "How does Signer create and check a signature?"
- Date: 2026-10-04
- Commit: 64c7ae6
- Notes: /analyze-repo and /ask return a `Server-Timing` header; /ask/stream adds `timings_ms` to its done event; every request logs one `timing <endpoint> stage=ms ...` line with no question, answer or code. `embed` covers the indexing embeddings (hash embedder); `read_files` is reading file contents for them. The streamed chat path calls the model three times per question and none of those calls is streamed from the model: retrieval summarises the snippets, the mentor re-runs retrieval (which summarises again), then the mentor answers. So the time to the first token the user sees is the sum of three full model calls; `first_token_sent` measures that once a key is set, and `first_model_token` is recorded only in general (no-repo) mode, where the model really streams. The answer is then sent 4 words every 8 ms, which adds about 0.8 s for a 400-word answer. Numbers come from an in-process client, so they exclude network time between browser and server, and the clone time depends on this machine's connection to GitHub

### Feature: index cache keyed by repository URL and commit (the only cache built)
- Value: indexing a repository that is already indexed at the same commit takes 687 ms instead of 1,907 ms for itsdangerous (−64%) and 776 ms instead of 4,638 ms for flask (−83%). A repository indexed for the first time pays no extra cost: the lookup is 0.0–0.2 ms and makes no network call. Medians in ms, 3 fresh processes × 3 runs per repo

  | | cache off, warm runs (n=6) | cache on, repeat of an indexed repo (n=6) | cache on, first index (n=3) |
  |---|---|---|---|
  | itsdangerous | 1907 (clone 1706) | 687 (`git ls-remote` 687, lookup 0.2) | 2552 (clone 2324) |
  | flask | 4638 (clone 2347) | 776 (`git ls-remote` 775, lookup 0.2) | 4939 (clone 2475) |

  The cache-on first index is slower than cache-off only because clone times varied between runs (clone 2324 vs 1846 ms cold); it adds no stage beyond the 0.0 ms lookup
- Command: `$env:CODEATLAS_INDEX_CACHE="false"; python scripts/time_stages.py https://github.com/pallets/itsdangerous --runs 3` and the same with `"true"`, each 3 times, and the same for https://github.com/pallets/flask; plus `python -m pytest` (152 passed; `tests/test_index_cache.py` covers a hit, other spellings of the URL, a new commit, a different repo, an unknown remote commit, a missing clone folder or index, the off switch, a restart with the disk kept, an invalid URL, old state files without a commit, no `ls-remote` for a first index, and a real local clone whose recorded commit matches `ls-remote`), `ruff check`, `ruff format --check`, and in `codementor-ui`: `npm test` (16 passed), `npm run lint` (0 errors, same 28 warnings as before), `npm run build`
- Dataset/repo: https://github.com/pallets/itsdangerous and https://github.com/pallets/flask
- Date: 2026-10-04
- Commit: 4e6c097
- Notes: how a hit is decided: same URL after normalising (letter case, trailing `.git`, web paths such as `/tree/main`), the earlier index and clone folder still present, and the commit from `git ls-remote <url> HEAD` equal to the commit checked out when it was indexed. On a hit the response has the same four fields and the earlier `repository_id`, so everyone who indexes the same public repository shares one index; chat history is never stored with it. `ls-remote` takes most of a hit's time; it gets at most 15 s, and any failure falls back to a full index. A hit still counts against the clone rate limit, since it still contacts the host. Two simultaneous first requests for the same repository both index it. CODEATLAS_INDEX_CACHE=false turns the cache off.
  **On a Render restart:** the in-memory indexes are rebuilt from CODEATLAS_INDEX_DIR and CODEATLAS_STATE_DIR if those are on a persistent disk, but clone folders are always written to `.codeatlas/repos` under the working directory. If that folder is not on the same disk, every entry is missed after a restart. The fallback is a normal full index, which is the old behaviour. I could not check how the Render service's disk is set up.
  **Caches skipped:**
  (1) **Query embedding cache**: `embed_query` takes 0.3 ms for 2 calls, about 1% of a 25–37 ms model-free chat question. There is nothing to gain with the hash embedder. It would matter only with `sentence-transformers`, which is not installed.
  (2) **Exact-match answer cache with expiry**: it would skip the model calls, but those could not be timed here because there is no model key. The rest of the question takes 25–37 ms. So the measured gain is about 30 ms, and the rule says skip. Measure the `llm` stage on a server with a key first. If it is built later, it should key on repository id, commit and question, only for a question with no conversation history, and with expiry. It needs no new dependency: an in-process dict is enough, and it is lost on restart.
  (3) **Pre-indexed example repo**: with the index cache, every "Try an example" click after the first already takes about 0.7 s. Indexing at startup would save about 1.9 s on only the first click after each restart. That restart already takes up to a minute to wake on Render. It would cost a clone on every boot, even when nobody visits.
  (4) **Not considered**: a shared cache across server processes or instances, such as Redis, would need a new dependency and an external store.

### Feature: conversation memory for the website chat
- Value: in the streamed chat, the last 4 turns of the same conversation now reach the prompt, capped at about 1,500 tokens (oldest dropped first). A follow-up that leans on earlier turns, such as "what about URLSafeTimedSerializer?", is rewritten into a standalone question for code search. That costs one extra model call; other questions make none. Without the model, the rewrite takes 1.5–1.8 ms (18 follow-ups, 3 fresh processes). The model call itself could not be timed (no key). Its input is the history (at most about 1,500 tokens) plus about 60 tokens of instructions, and its output is one short question. It shows as the `rewrite` stage in `timings_ms` and the log line
- Command: `python scripts/time_stages.py https://github.com/pallets/itsdangerous --runs 3 --follow-up "What about URLSafeTimedSerializer?"` (3 times); `python -m pytest` (181 passed; `tests/test_conversation_memory.py`, 29 tests, covers the turn limit, the token cap and shortening of a single oversized turn, idle expiry, the session-count cap, isolation between sessions, between repositories and between repo and general chat, hashed session keys, when a rewrite happens and when it doesn't, rewrite fallback when the model fails, the rewritten question reaching search while the mentor gets the question as asked plus the history, no memory without a session id, 422 for malformed ids, failed answers not kept, and no conversation text in logs at DEBUG level or in any file written under the test folder); `ruff check`, `ruff format --check`; in `codementor-ui`: `npm test` (19 passed, 3 new for the session id), `npx tsc --noEmit`, `npm run lint` (0 errors, same 28 warnings as before), `npm run build`. Manual: local `uvicorn` with no model key: two questions with the same session id. The second one's reasoning steps said "Used 1 earlier turn(s)" and showed the rewrite. A session id `x` returned 422, and a marker word from the questions appeared 0 times in the server log
- Dataset/repo: https://github.com/pallets/itsdangerous; small Python project written by the tests
- Date: 2026-10-04
- Commit: 089a638
- Notes: the chat window makes a random id (`crypto.randomUUID`, or 16 random bytes in hex outside a secure context). It keeps the id in React state for that conversation and sends it as `session_id` on /ask/stream. The id is not stored in localStorage or sessionStorage, so a page reload starts a new chat on screen and a new memory. The server accepts ids matching `^[A-Za-z0-9_-]{16,128}$`. It keys memory by sha256 of id plus repository id, holds it in process memory only, and drops a session after 30 idle minutes or when 1,000 sessions are exceeded (least recently written first). No endpoint returns stored history: a session's turns only go into that session's own prompts. Reading another session's history would require guessing its 122-bit random id. Tokens are estimated as characters ÷ 4 (no tokenizer dependency). The rewrite is used only in repo mode, because only search needs it; general mode passes history as chat messages. Rewriting is decided by a word rule: 3 words or fewer, opening with and/but/so/or/then, starting with "what about"/"how about", or a referring word such as it/that/they/those (but not "this repo"). It errs toward rewriting: a false positive costs one model call, a false negative a weaker search. Failed answers are not kept. Settings: CODEATLAS_CHAT_HISTORY_TURNS, CODEATLAS_CHAT_HISTORY_TOKENS, CODEATLAS_CHAT_SESSION_TTL_SECONDS, CODEATLAS_CHAT_MAX_SESSIONS. **On a Render restart** all memory is lost and the next question starts fresh; there is no fallback store by design. With more than one uvicorn worker, a session's turns stay only in the worker that received them. **Not changed, but conflicts with "no cross-user content":** the existing analytics tracker keeps every question in memory, and the public GET /eval/stats returns the last 20 (first 120 characters) to anyone, so the eval page shows other visitors' questions. /ask (non-streamed, not used by the website) ignores `session_id`. With no model key, the stand-in model's fixed reply becomes the rewritten follow-up, so follow-up citations are poor in that local-only mode
