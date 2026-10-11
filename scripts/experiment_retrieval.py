"""Score the five retrieval arrangements on one split, for one embedder, and write the result.

    python scripts/experiment_retrieval.py --embedder hash --split dev \
        --json eval/results/experiments/bm25-dev-hash.json
    python scripts/experiment_retrieval.py --embedder cloudflare --split dev --json ...

**Neither embedder calls an API.** `hash` is local arithmetic. `cloudflare` reads vectors from
the evaluation cache and **refuses to call the provider**: a text that is not already paid for
stops the run and says how many are missing, never which. That is the whole reason this script
exists separately from `eval_cloudflare.py`, which is allowed to spend.

Results go to `eval/results/experiments/`, a folder nothing else reads - not the `/metrics`
gauges, which glob `eval/results/*.json` and are not recursive, and not the Benchmarks summary,
which names the files it reads. So an experiment cannot move a published number.

The arrangements are in `services/eval/arrangements.py`. `current` is the server's own settings,
so it must reproduce the committed numbers for the same split; `--expect-current` checks that
and exits 1 if it has drifted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.app.di import cloudflare_settings, get_config, get_embedder, get_retrieval_settings  # noqa: E402
from codeatlas.models.repository import Repository  # noqa: E402
from codeatlas.services.eval.arrangements import (  # noqa: E402
    ARRANGEMENTS,
    DESCRIPTIONS,
    Arrangements,
    arrange,
)
from codeatlas.services.eval.basic_eval import RECALL_DEPTH, hit_at_k, ranking_metrics  # noqa: E402
from codeatlas.services.eval.retrieval_eval import (  # noqa: E402
    RANKED_DEPTH,
    SPLITS,
    checkout,
    describe_embedder,
    load_question_set,
)
from codeatlas.services.eval.vector_cache import OfflineVectorCache, cache_key  # noqa: E402
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser  # noqa: E402
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever  # noqa: E402
from codeatlas.services.retrieval.indexing import CodeIndexService  # noqa: E402
from codeatlas.services.retrieval.record_format import (  # noqa: E402
    RECORD_FORMATS,
    WHOLE,
    with_record_format,
)

DEFAULT_SETS = [ROOT / "eval" / "questions" / "itsdangerous.json", ROOT / "eval" / "questions" / "flask.json"]
HOLDOUT2_SETS = [
    ROOT / "eval" / "questions" / "holdout2" / "itsdangerous.json",
    ROOT / "eval" / "questions" / "holdout2" / "flask.json",
]
REPO_CACHE = ROOT / ".codeatlas" / "eval-repos"
VECTOR_CACHE = ROOT / ".codeatlas" / "eval-vectors"


def offline_cloudflare(app_config, cache_dir: Path) -> tuple[OfflineVectorCache, int]:
    """The cached Cloudflare vectors, served without a client and without credentials.

    The signature is built from the settings rather than from a provider object, so nothing
    here constructs something that would want a credential.
    """
    settings = cloudflare_settings(app_config)
    key = cache_key(settings.model, settings.pooling, settings.max_chars)
    path = cache_dir / f"{__import__('hashlib').sha256(key.encode('utf-8')).hexdigest()[:16]}.json"
    signature = f"cloudflare:{settings.model}:pooling={settings.pooling}:max_chars={settings.max_chars}"
    return OfflineVectorCache(path, signature), settings.max_chars


def evaluate(
    question_files,
    embedder,
    split: str,
    max_chars: int | None,
    arrangements,
    label: str,
    record_format: str,
    window_chars: int,
    window_overlap: int,
) -> dict:
    retrieval = get_retrieval_settings()
    config = get_config()
    depth = max(RANKED_DEPTH, RECALL_DEPTH, retrieval.candidates)

    repos: dict[str, dict] = {}
    pairs: dict[str, list] = {name: [] for name in arrangements}
    pools: dict[str, list] = {name: [] for name in arrangements}

    for path in question_files:
        question_set = load_question_set(path)
        root = checkout(question_set, REPO_CACHE)
        repository = Repository(
            repo_id=f"eval-{question_set.name}",
            name=question_set.name,
            url=question_set.repo_url,
            root_path=str(root.resolve()),
            ingested_at=datetime.now(UTC),
            commit=question_set.commit,
        )
        parsed = TreeSitterAstParser().parse_repository(repository)
        retriever = FaissCodeRetriever()
        CodeIndexService(
            embedder=embedder,
            retriever=retriever,
            max_chars=max_chars,
            prefix_metadata=config.embed_prefix_metadata,
            batch_size=64,
            record_format=record_format,
            window_chars=window_chars,
            window_overlap=window_overlap,
        ).index_repository(repository, parsed)
        built = Arrangements.build(retriever, embedder, retrieval, repository.repo_id)

        repo_pairs: dict[str, list] = {name: [] for name in arrangements}
        repo_pools: dict[str, list] = {name: [] for name in arrangements}
        questions = question_set.select(split)
        for q in questions:
            expected = set(q.expected_files)
            for name in arrangements:
                records = arrange(name, built, repository.repo_id, q.question, depth)
                ranked = [_relative(r.metadata.get("path", ""), root) for r in records]
                repo_pairs[name].append((ranked[:RANKED_DEPTH], expected))
                repo_pools[name].append((ranked, expected))
                pairs[name].append((ranked[:RANKED_DEPTH], expected))
                pools[name].append((ranked, expected))

        repos[question_set.name] = {
            "commit": question_set.commit,
            "files_indexed": len(parsed.files),
            "records_indexed": len(parsed.files) + len(parsed.functions),
            "questions": len(questions),
            "arrangements": {n: _metrics(repo_pairs[n], repo_pools[n], retrieval.candidates) for n in arrangements},
        }

    return {
        "embedder": label,
        "split": split,
        "record_format": record_format,
        "candidates": retrieval.candidates,
        "repos": repos,
        "overall": {n: _metrics(pairs[n], pools[n], retrieval.candidates) for n in arrangements},
    }


def _metrics(pairs, pools, candidates: int) -> dict:
    m = ranking_metrics(pairs, (1, 3, 5), candidates=pools)
    return {
        "questions": m.questions,
        "hit@1": round(m.hit_rate[1], 4),
        "hit@3": round(m.hit_rate[3], 4),
        "hit@5": round(m.hit_rate[5], 4),
        f"mrr@{RANKED_DEPTH}": round(m.mrr, 4),
        f"recall@{RECALL_DEPTH}": round(m.recall, 4),
        # Recall at the pool depth the server actually reranks from. At the default candidates
        # of 20 this is the same number as recall@20; it is printed separately so that a run
        # with a different depth does not quietly relabel it.
        f"recall@candidates({candidates})": round(
            sum(hit_at_k(r, e, candidates) for r, e in pools) / len(pools) if pools else 0.0, 4
        ),
    }


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def print_results(result: dict) -> None:
    print(f"\nembedder: {result['embedder']}")
    print(f"split: {result['split']}, records {result['record_format']}, candidate pool {result['candidates']}")
    recall_key = f"recall@candidates({result['candidates']})"
    header = (
        f"  {'arrangement':<14}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}"
        f"{'MRR@10':>9}{'R@20':>8}{'R@pool':>8}"
    )
    for label, block in [("overall", result["overall"]), *[(n, r["arrangements"]) for n, r in result["repos"].items()]]:
        print(f"\n{label}")
        print(header)
        for name, m in block.items():
            print(
                f"  {name:<14}{m['questions']:>4}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}"
                f"{m['mrr@10']:>9.3f}{m['recall@20']:>8.3f}{m[recall_key]:>8.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["hash", "cloudflare"], default="hash")
    parser.add_argument("--record-format", choices=RECORD_FORMATS, default=WHOLE)
    parser.add_argument("--window-chars", type=int, help="default: the clip, or 2000 when there is none")
    parser.add_argument("--window-overlap", type=int, default=200)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--questions", choices=["original", "holdout2"], default="original")
    parser.add_argument("--json", type=Path, help="write the result here")
    parser.add_argument("--vector-cache", type=Path, default=VECTOR_CACHE)
    parser.add_argument(
        "--arrangement", dest="arrangements", action="append", choices=ARRANGEMENTS, help="default: all five"
    )
    parser.add_argument(
        "--expect-current",
        metavar="METRIC=VALUE",
        action="append",
        default=[],
        help="fail if the 'current' arrangement's overall metric is not this (e.g. mrr@10=0.6625)",
    )
    args = parser.parse_args()

    app_config = get_config()
    if args.embedder == "cloudflare":
        embedder, max_chars = offline_cloudflare(app_config, args.vector_cache)
        label = f"cloudflare ({embedder.signature()}, served from the evaluation cache, no API call)"
        print(f"offline cloudflare cache: {embedder.path.name}, {len(embedder)} vectors, clip {max_chars}")
    else:
        embedder, max_chars = get_embedder(), app_config.embed_max_chars
        label = describe_embedder(embedder)
        print(f"embedder: {label}")

    files = HOLDOUT2_SETS if args.questions == "holdout2" else DEFAULT_SETS
    arrangements = tuple(args.arrangements or ARRANGEMENTS)
    started = time.perf_counter()
    embedder = with_record_format(embedder, args.record_format)
    window_chars = args.window_chars or max_chars or 2000
    print(f"record format: {args.record_format} (window {window_chars}, overlap {args.window_overlap})")
    result = evaluate(
        files,
        embedder,
        args.split,
        max_chars,
        arrangements,
        label,
        args.record_format,
        window_chars,
        args.window_overlap,
    )
    result = {
        "codementor_commit": _commit(),
        "questions": args.questions,
        "arrangements": {name: DESCRIPTIONS[name] for name in arrangements},
        **result,
        "seconds": round(time.perf_counter() - started, 1),
    }
    if args.embedder == "cloudflare":
        result["cache"] = {"file": embedder.path.name, "vectors": len(embedder), "hits": embedder.hits, "embedded": 0}

    print_results(result)
    print(f"\nseconds: {result['seconds']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    failed = False
    for spec in args.expect_current:
        metric, value = spec.split("=")
        actual = result["overall"]["current"][metric]
        if abs(actual - float(value)) > 1e-9:
            print(f"FAIL: current {metric} is {actual}, not the recorded {value}")
            failed = True
        else:
            print(f"ok: current {metric} {actual} matches the recorded value")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
