"""Score the five retrieval arrangements on one split, for one embedder, and write the result.

    python scripts/experiment_retrieval.py --embedder hash --split dev \
        --json eval/results/experiments/bm25-dev-hash.json
    python scripts/experiment_retrieval.py --embedder cloudflare --split dev --json ...

**By default neither embedder calls an API.** `hash` is local arithmetic. `cloudflare` reads
vectors from the evaluation cache and **refuses to call the provider**: a text that is not
already paid for stops the run and says how many are missing, never which.

A record format other than `whole` changes every document, so its vectors are not in the cache
and cannot be. `--allow-spend` is the deliberate way to pay for them, and it:

* prints the estimate and today's ledger **before anything is sent**, counting only the texts
  that are not already cached;
* refuses to start if the estimate would take today's total past the budget, so a run that
  cannot finish does not spend half a budget discovering that;
* writes every vector it pays for into the same cache, so a rerun is free.

`--plan-only` prints the estimate and sends nothing.

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
from codeatlas.services.eval.vector_cache import CachingEmbeddingService, OfflineVectorCache, cache_key  # noqa: E402
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser  # noqa: E402
from codeatlas.services.qa.answer_service import _without_stopwords  # noqa: E402
from codeatlas.services.retrieval import neuron_ledger  # noqa: E402
from codeatlas.services.retrieval.cloudflare_embedder import CloudflareEmbeddingService  # noqa: E402
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever  # noqa: E402
from codeatlas.services.retrieval.indexing import CodeIndexService  # noqa: E402
from codeatlas.services.retrieval.record_format import (  # noqa: E402
    RECORD_FORMATS,
    WHOLE,
    with_record_format,
)
from codeatlas.utils.config import load_cloudflare_credentials  # noqa: E402

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
    # The same string the real provider reports, so the two cloudflare result files describe
    # their embedder identically and nobody has to work out whether they are the same vectors.
    signature = (
        f"cloudflare:{settings.model}:pooling={settings.pooling}:max_chars={settings.max_chars}"
        f":prefix={int(settings.prefix_metadata)}:stopwords={int(settings.drop_stopwords)}"
    )
    return OfflineVectorCache(path, signature), settings.max_chars


def spending_cloudflare(app_config, cache_dir: Path) -> CachingEmbeddingService:
    """The cached embedder that *is* allowed to call the provider when a text is not cached."""
    settings = cloudflare_settings(app_config)
    return CachingEmbeddingService(
        inner=CloudflareEmbeddingService(credentials=load_cloudflare_credentials(), settings=settings),
        key=cache_key(settings.model, settings.pooling, settings.max_chars),
        cache_dir=cache_dir,
    )


def plan(question_files, embedder, max_chars, record_format, window_chars, window_overlap, split) -> dict:
    """What this run would cost: the texts it needs that are not already paid for.

    Counts documents and questions exactly as the run will send them, and batches them the way
    the run will, because a batch is billed at the length of its longest member.
    """
    config = get_config()
    batch_size = cloudflare_settings(config).batch_size
    unpaid: list[str] = []
    documents = 0
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
        indexer = CodeIndexService(
            embedder=embedder,
            retriever=FaissCodeRetriever(),
            max_chars=max_chars,
            prefix_metadata=config.embed_prefix_metadata,
            batch_size=batch_size,
            record_format=record_format,
            window_chars=window_chars,
            window_overlap=window_overlap,
        )
        texts, _ = indexer.documents(repository, parsed)
        questions = [
            _without_stopwords(q.question) if config.drop_stopwords else q.question
            for q in question_set.select(split)
        ]
        both = texts + questions
        unpaid.extend(text for text, known in zip(both, embedder.known(both)) if not known)
        documents += len(texts)
    return {
        "documents": documents,
        "uncached_texts": len(unpaid),
        "estimated_neurons": round(neuron_ledger.estimated_neurons(unpaid, batch_size), 2),
    }


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
    parser.add_argument(
        "--allow-spend",
        action="store_true",
        help="let the cloudflare run pay for vectors it does not have (prints the estimate first)",
    )
    parser.add_argument("--plan-only", action="store_true", help="print the estimate and send nothing")
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
    budget = app_config.cloudflare_daily_neurons
    if args.embedder == "cloudflare" and args.allow_spend:
        if load_cloudflare_credentials() is None:
            print("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must both be set (.env is read).")
            return 1
        embedder = spending_cloudflare(app_config, args.vector_cache)
        max_chars = cloudflare_settings(app_config).max_chars
        label = f"cloudflare ({embedder.signature()}, allowed to spend)"
        print(f"cloudflare cache: {embedder.path.name}, clip {max_chars}")
    elif args.embedder == "cloudflare":
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

    if args.embedder == "cloudflare" and (args.allow_spend or args.plan_only):
        estimate = plan(
            files, embedder, max_chars, args.record_format, window_chars, args.window_overlap, args.split
        )
        spent = neuron_ledger.spent_today()
        print(
            f"\nplan: {estimate['documents']} documents, {estimate['uncached_texts']} not yet cached, "
            f"{estimate['estimated_neurons']:.2f} estimated neurons"
        )
        print(
            f"ledger: {spent.estimated_neurons:.2f} spent today (UTC {neuron_ledger.today()}), "
            f"budget {budget:g}, remaining {max(budget - spent.estimated_neurons, 0.0):.2f}"
        )
        if args.plan_only:
            print("plan only: nothing was sent")
            return 0
        if not neuron_ledger.affordable(estimate["estimated_neurons"], budget):
            print("refusing to start: this run would take today's total past the budget.")
            return 1
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
        inner = embedder.inner if hasattr(embedder, "inner") else embedder
        result["cache"] = {
            "file": inner.path.name,
            "hits": inner.hits,
            "embedded": getattr(inner, "misses", 0),
        }
        if args.allow_spend:
            after = neuron_ledger.spent_today()
            result["ledger_after"] = {
                "estimated_neurons": round(after.estimated_neurons, 4),
                "calls": after.calls,
                "day": neuron_ledger.today(),
            }
            print(
                f"\nledger after this run: {after.estimated_neurons:.2f} estimated neurons, "
                f"{after.calls} calls; remaining {max(budget - after.estimated_neurons, 0.0):.2f}"
            )

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
