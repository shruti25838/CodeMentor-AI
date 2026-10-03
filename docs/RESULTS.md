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
