Call-graph evaluation against Python's own `ast` module, written by

    python scripts/structural_eval.py --json eval/results/structural/all.json

It lives in a subfolder for the same reason the fastembed results do: `load_eval_results`
globs `*.json` in `eval/results` itself and labels each gauge by the file's `split`, and this
file has no single split. The glob is not recursive, so nothing but the benchmarks summary
reads it.

The script needs the two pinned repositories cloned under `.codeatlas/eval-repos/` (ignored by
git), which `scripts/build_benchmarks.py` and its test cannot assume. That is why the figures
are committed here rather than recomputed: the summary reads this file, and this file is
rewritten only by re-running the command above. No model and no API is involved either way.
