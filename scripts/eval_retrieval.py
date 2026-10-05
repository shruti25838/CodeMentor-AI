"""Score retrieval on the fixed question sets: hit rate@k, precision@k (k = 1, 3, 5) and MRR@10.

    python scripts/eval_retrieval.py --split dev
    python scripts/eval_retrieval.py --split test --json eval/results/test.json

Needs no model key: only retrieval runs. The embedder is the one the server would use
(CODEATLAS_EMBEDDING_PROVIDER, default hash) and its name is printed with the results.
Each repository is fetched at the commit pinned in its question file into --cache-dir
(default .codeatlas/eval-repos, ignored by git) and reused on later runs.

--min-hit-rate K=VALUE (repeatable) exits with status 1 if the overall hit rate at K is below VALUE.

The tuning flags (--candidates and below) change one retrieval or indexing setting for this run
only; without them the run uses the server's settings. They exist for the experiments in
docs/RESULTS.md.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.app.di import get_config, get_embedder, get_retrieval_settings  # noqa: E402
from codeatlas.services.eval.retrieval_eval import RANKED_DEPTH, SPLITS, EvalSettings, IndexSettings, run  # noqa: E402
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService  # noqa: E402

DEFAULT_SETS = [ROOT / "eval" / "questions" / "itsdangerous.json", ROOT / "eval" / "questions" / "flask.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--set", dest="sets", action="append", type=Path, help="question file (default: all)")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".codeatlas" / "eval-repos")
    parser.add_argument("--json", type=Path, help="also write the full results here")
    parser.add_argument("--min-hit-rate", action="append", default=[], metavar="K=VALUE")
    tuning = parser.add_argument_group("tuning (one setting per experiment)")
    tuning.add_argument("--candidates", type=int, help="records taken from vector search before reranking")
    tuning.add_argument("--rerank-weight", type=float, help="1.0 keyword overlap only, 0.0 vector similarity only")
    tuning.add_argument("--rerank-subtokens", choices=["on", "off"])
    tuning.add_argument("--drop-stopwords", choices=["on", "off"])
    tuning.add_argument("--skip-tests", choices=["on", "off"])
    tuning.add_argument("--embed-max-chars", type=int, help="0 = no limit")
    tuning.add_argument("--prefix-metadata", choices=["on", "off"])
    tuning.add_argument("--hash-lowercase", choices=["on", "off"])
    tuning.add_argument("--hash-subtokens", choices=["on", "off"])
    args = parser.parse_args()

    settings, embedder = _settings(args)
    result = run(args.sets or DEFAULT_SETS, embedder, args.split, args.cache_dir, settings=settings)
    result = {"codementor_commit": _commit(), **result}

    print(f"codementor commit: {result['codementor_commit']}")
    print(f"embedder: {result['embedder']}")
    print(f"settings: {json.dumps(result['settings'], separators=(',', ':'))}")
    print(f"split: {result['split']}")
    header = f"{'repo':<14}{'n':>4}  {'hit@1':>6}{'hit@3':>7}{'hit@5':>7}  {'P@1':>6}{'P@3':>7}{'P@5':>7}  {'MRR@' + str(RANKED_DEPTH):>7}"
    print(header)
    rows = [(name, r) for name, r in result["repos"].items()] + [("overall", result["overall"])]
    for name, m in rows:
        h, p = m["hit_rate"], m["precision"]
        print(
            f"{name:<14}{m['questions']:>4}  {h['@1']:>6.3f}{h['@3']:>7.3f}{h['@5']:>7.3f}"
            f"  {p['@1']:>6.3f}{p['@3']:>7.3f}{p['@5']:>7.3f}  {m[f'mrr@{RANKED_DEPTH}']:>7.3f}"
        )
    for name, r in result["repos"].items():
        print(f"{name}: pinned commit {r['commit'][:12]}, {r['files_indexed']} files, {r['records_indexed']} records")
    print(f"seconds: {result['seconds']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    failed = False
    for spec in args.min_hit_rate:
        k, value = spec.split("=")
        actual = result["overall"]["hit_rate"][f"@{int(k)}"]
        if actual < float(value):
            print(f"FAIL: overall hit@{k} {actual:.4f} is below the minimum {float(value):.4f}")
            failed = True
        else:
            print(f"ok: overall hit@{k} {actual:.4f} >= {float(value):.4f}")
    return 1 if failed else 0


def _settings(args) -> tuple[EvalSettings, object]:
    """The server's settings, with any tuning flags applied on top."""
    config = get_config()
    changes = {
        "candidates": args.candidates,
        "rerank_weight": args.rerank_weight,
        "rerank_subtokens": _on(args.rerank_subtokens),
        "drop_stopwords": _on(args.drop_stopwords),
        "skip_tests": _on(args.skip_tests),
    }
    retrieval = replace(get_retrieval_settings(), **{k: v for k, v in changes.items() if v is not None})
    max_chars = config.embed_max_chars if args.embed_max_chars is None else (args.embed_max_chars or None)
    prefix = config.embed_prefix_metadata if args.prefix_metadata is None else _on(args.prefix_metadata)
    embedder = get_embedder()
    if isinstance(embedder, HashEmbeddingService) and (args.hash_lowercase or args.hash_subtokens):
        lowercase = embedder._lowercase if args.hash_lowercase is None else _on(args.hash_lowercase)
        subtokens = embedder._subtokens if args.hash_subtokens is None else _on(args.hash_subtokens)
        embedder = HashEmbeddingService(lowercase=lowercase, subtokens=subtokens)
    return EvalSettings(retrieval=retrieval, index=IndexSettings(max_chars=max_chars, prefix_metadata=prefix)), embedder


def _on(value: str | None) -> bool | None:
    return None if value is None else value == "on"


def _commit() -> str:
    """Short hash of this checkout, with -dirty if tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


if __name__ == "__main__":
    raise SystemExit(main())
