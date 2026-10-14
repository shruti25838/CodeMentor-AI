# The legacy hash baseline

These two files are the hash embedder on **whole records with BM25 off** — the configuration
every number recorded in `docs/RESULTS.md` up to and including the holdout3 entries was measured
with, and the one the server shipped before the default changed.

They exist so that the old baseline stays **reproducible**, not merely remembered. `eval/results/dev.json`
and `eval/results/test.json` are regenerated whenever the default changes and describe whatever
the default is now; these do not move.

## Reproducing them

```powershell
$env:CODEATLAS_HASH_RECORD_FORMAT = "whole"
$env:CODEATLAS_HASH_BM25 = "false"
python scripts/eval_retrieval.py --split test --json eval/results/legacy/test.json
python scripts/eval_retrieval.py --split dev  --json eval/results/legacy/dev.json
Remove-Item Env:\CODEATLAS_HASH_RECORD_FORMAT
Remove-Item Env:\CODEATLAS_HASH_BM25
```

The two global variables `CODEATLAS_RECORD_FORMAT=whole` and `CODEATLAS_BM25=false` do the same
thing for a server running the hash provider; the per-kind pair above is used here because it
survives a change of provider.

## What they say

| split | questions | hit@5 | MRR@10 | recall@20 |
|---|---|---|---|---|
| test (held out) | 20 | **0.750** | **0.645** | 0.800 |
| dev | 20 | 0.750 | 0.6625 | 0.850 |

`tests/test_legacy_baseline.py` pins the held-out pair and asserts that the configuration these
files name is the one the variables above actually produce. The numbers are byte-identical to
the `eval/results/{dev,test}.json` recorded at `016c655`, which is what makes them a baseline
rather than a second measurement.
