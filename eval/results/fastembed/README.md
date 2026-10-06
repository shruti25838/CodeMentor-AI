Retrieval eval results for the opt-in `fastembed` provider.

They live in a subfolder on purpose. `load_eval_results` globs `*.json` in `eval/results`
itself and labels each gauge by the file's `split`, so a second file with the same split next
to `dev.json` would overwrite the gauge and `/metrics` would report fastembed numbers while the
server runs `hash`. The glob is not recursive, so these files are the record of the comparison
and nothing else reads them.

Regenerate with:

    CODEATLAS_EMBEDDING_PROVIDER=fastembed python scripts/eval_retrieval.py --split dev  --json eval/results/fastembed/dev.json
    CODEATLAS_EMBEDDING_PROVIDER=fastembed python scripts/eval_retrieval.py --split test --json eval/results/fastembed/test.json

which needs `pip install -r requirements-fastembed.txt` first.
