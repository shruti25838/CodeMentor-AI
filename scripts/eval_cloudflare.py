"""Compare Cloudflare embedding configurations, and orderings within each, on a fixed budget.

    python scripts/eval_cloudflare.py --split dev --plan-only   # what it would cost, calls nothing
    python scripts/eval_cloudflare.py --split dev               # runs it
    python scripts/eval_cloudflare.py --split test --config mean-2000 --config cls-2000

A configuration is a model, a pooling and a per-document clip: the three things that change
what a vector is. Each one indexes both repositories **once**, and the three orderings -
vector similarity, keyword overlap, and reciprocal rank fusion of the two - are scored from
that one index, so comparing orderings costs nothing at all.

**Spending is the hard part, not the scoring.** The free allowance is 10,000 neurons a day and
resets at 00:00 UTC. So:

* Every vector is cached on disk by model, pooling, clip and a hash of the text
  (services/eval/vector_cache.py), which is why a held-out run after a dev run is nearly free
  and why a rerun costs nothing at all.
* The estimate printed before a run counts **only what is not already cached**, so it says what
  this run will really cost rather than what the corpus is worth.
* A run that would take today's total past the budget refuses to start, naming both figures.
* `--plan-only` prints the estimate and stops without sending anything.

**The 600-document cap does not apply here.** That cap protects a visitor's repository from
spending the day's allowance on one analysis; an experiment is a deliberate, measured spend,
and clipping the corpus would make the numbers describe something other than the repositories.

Nothing here prints a document, a question, a URL or a credential: configurations, counts,
metrics and neuron estimates only.
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.app.di import cloudflare_settings, get_config, get_retrieval_settings  # noqa: E402
from codeatlas.models.repository import Repository  # noqa: E402
from codeatlas.services.eval.basic_eval import (  # noqa: E402
    RECALL_DEPTH,
    hit_at_k,
    ranking_metrics,
    reciprocal_rank,
)
from codeatlas.services.eval.orderings import ORDERINGS, Rankers, order  # noqa: E402
from codeatlas.services.eval.retrieval_eval import (  # noqa: E402
    RANKED_DEPTH,
    SPLITS,
    checkout,
    load_question_set,
)
from codeatlas.services.eval.vector_cache import CachingEmbeddingService, cache_key  # noqa: E402
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser  # noqa: E402
from codeatlas.services.retrieval import neuron_ledger  # noqa: E402
from codeatlas.services.retrieval.cloudflare_embedder import CloudflareEmbeddingService  # noqa: E402
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever  # noqa: E402
from codeatlas.services.retrieval.indexing import CodeIndexService  # noqa: E402
from codeatlas.utils.config import load_cloudflare_credentials  # noqa: E402

DEFAULT_SETS = [ROOT / "eval" / "questions" / "itsdangerous.json", ROOT / "eval" / "questions" / "flask.json"]
DEFAULT_CACHE_DIR = ROOT / ".codeatlas" / "eval-repos"
VECTOR_CACHE_DIR = ROOT / ".codeatlas" / "eval-vectors"


@dataclass(frozen=True)
class Config:
    """One set of settings that changes what a vector is, and therefore needs its own index."""

    name: str
    pooling: str
    max_chars: int


# The three the batch set out to compare: the default, a tighter clip, and the other pooling.
CONFIGS = {
    "mean-2000": Config("mean-2000", "mean", 2000),
    "mean-1500": Config("mean-1500", "mean", 1500),
    "cls-2000": Config("cls-2000", "cls", 2000),
}


def build_embedder(config: Config, app_config, cache_dir: Path) -> CachingEmbeddingService:
    settings = replace(
        cloudflare_settings(app_config),
        pooling=config.pooling,
        max_chars=config.max_chars,
    )
    inner = CloudflareEmbeddingService(credentials=load_cloudflare_credentials(), settings=settings)
    return CachingEmbeddingService(
        inner=inner,
        key=cache_key(settings.model, settings.pooling, settings.max_chars),
        cache_dir=cache_dir,
    )


def repository_for(question_set, root: Path) -> Repository:
    return Repository(
        repo_id=f"eval-{question_set.name}",
        name=question_set.name,
        url=question_set.repo_url,
        root_path=str(root.resolve()),
        ingested_at=datetime.now(UTC),
        commit=question_set.commit,
    )


def plan_for(config: Config, question_files, split: str, app_config, cache_dir: Path, repo_cache: Path) -> dict:
    """What this configuration would cost, counting only the texts that are not already cached."""
    embedder = build_embedder(config, app_config, cache_dir)
    index_settings = replace(cloudflare_settings(app_config), pooling=config.pooling, max_chars=config.max_chars)
    unpaid: list[str] = []
    documents_total = 0
    per_repo = {}
    for path in question_files:
        question_set = load_question_set(path)
        root = checkout(question_set, repo_cache)
        repository = repository_for(question_set, root)
        parsed = TreeSitterAstParser().parse_repository(repository)
        indexer = CodeIndexService(
            embedder=embedder,
            retriever=FaissCodeRetriever(),
            max_chars=index_settings.max_chars,
            prefix_metadata=app_config.embed_prefix_metadata,
            batch_size=index_settings.batch_size,
        )
        documents, _ = indexer.documents(repository, parsed)
        questions = [q.question for q in question_set.select(split)]
        texts = documents + questions
        unpaid.extend(text for text, known in zip(texts, embedder.known(texts)) if not known)
        documents_total += len(documents)
        per_repo[question_set.name] = {
            "documents": len(documents),
            "questions": len(questions),
            "clipped": indexer.clipped_counts(parsed),
        }
    return {
        "config": config.name,
        "documents": documents_total,
        "uncached_texts": len(unpaid),
        # Batched as the run will send them, since a batch is billed by its longest member.
        "estimated_neurons": round(neuron_ledger.estimated_neurons(unpaid, index_settings.batch_size), 2),
        "repos": per_repo,
        "cache_file": embedder.path.name,
    }


def evaluate(config: Config, question_files, split: str, app_config, cache_dir: Path, repo_cache: Path) -> dict:
    """Index each repository once for this configuration, then score every ordering from it."""
    embedder = build_embedder(config, app_config, cache_dir)
    index_settings = replace(cloudflare_settings(app_config), pooling=config.pooling, max_chars=config.max_chars)
    retrieval = get_retrieval_settings()
    depth = max(RANKED_DEPTH, RECALL_DEPTH, retrieval.candidates)

    repos: dict[str, dict] = {}
    pairs: dict[str, list] = {ordering: [] for ordering in ORDERINGS}
    pools: dict[str, list] = {ordering: [] for ordering in ORDERINGS}
    per_question: dict[str, dict] = {}

    for path in question_files:
        question_set = load_question_set(path)
        root = checkout(question_set, repo_cache)
        repository = repository_for(question_set, root)
        parsed = TreeSitterAstParser().parse_repository(repository)
        retriever = FaissCodeRetriever()
        indexer = CodeIndexService(
            embedder=embedder,
            retriever=retriever,
            max_chars=index_settings.max_chars,
            prefix_metadata=app_config.embed_prefix_metadata,
            batch_size=index_settings.batch_size,
        )
        indexer.index_repository(repository, parsed)
        rankers = Rankers.build(retriever, embedder, retrieval)

        repo_pairs: dict[str, list] = {ordering: [] for ordering in ORDERINGS}
        repo_pools: dict[str, list] = {ordering: [] for ordering in ORDERINGS}
        questions = question_set.select(split)
        for q in questions:
            expected = set(q.expected_files)
            for ordering in ORDERINGS:
                records = order(ordering, rankers, repository.repo_id, q.question, depth)
                ranked = [_relative(record.metadata.get("path", ""), root) for record in records]
                repo_pairs[ordering].append((ranked[:RANKED_DEPTH], expected))
                repo_pools[ordering].append((ranked, expected))
                pairs[ordering].append((ranked[:RANKED_DEPTH], expected))
                pools[ordering].append((ranked, expected))
                per_question.setdefault(ordering, {})[q.id] = {
                    "hit@5": hit_at_k(ranked[:RANKED_DEPTH], expected, 5),
                    "rr": round(reciprocal_rank(ranked[:RANKED_DEPTH], expected), 4),
                    f"recall@{RECALL_DEPTH}": hit_at_k(ranked, expected, RECALL_DEPTH),
                }

        repos[question_set.name] = {
            "commit": question_set.commit,
            "files_indexed": len(parsed.files),
            "records_indexed": len(parsed.files) + len(parsed.functions),
            "questions": len(questions),
            "clipped": indexer.clipped_counts(parsed),
            "orderings": {ordering: _metrics(repo_pairs[ordering], repo_pools[ordering]) for ordering in ORDERINGS},
        }

    return {
        "config": config.name,
        "pooling": config.pooling,
        "max_chars": config.max_chars,
        "split": split,
        "repos": repos,
        "overall": {ordering: _metrics(pairs[ordering], pools[ordering]) for ordering in ORDERINGS},
        "per_question": per_question,
        # The cache file by name only: the full path is a developer's home directory, which has
        # no place in a results file that gets committed.
        "cache": {"hits": embedder.hits, "misses": embedder.misses, "file": embedder.path.name},
    }


def _metrics(pairs, pools) -> dict:
    m = ranking_metrics(pairs, (1, 3, 5), candidates=pools)
    return {
        "questions": m.questions,
        "hit@1": round(m.hit_rate[1], 4),
        "hit@3": round(m.hit_rate[3], 4),
        "hit@5": round(m.hit_rate[5], 4),
        f"mrr@{RANKED_DEPTH}": round(m.mrr, 4),
        f"recall@{RECALL_DEPTH}": round(m.recall, 4),
    }


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def print_plan(plans: list[dict], budget: float) -> float:
    spent = neuron_ledger.spent_today().estimated_neurons
    total = sum(plan["estimated_neurons"] for plan in plans)
    print(f"ledger: {spent:.2f} estimated neurons spent today (UTC {neuron_ledger.today()})")
    print(f"budget: {budget:g}, remaining {max(budget - spent, 0.0):.2f}")
    for plan in plans:
        print(
            f"  {plan['config']:<12}{plan['documents']:>6} documents"
            f"{plan['uncached_texts']:>7} not yet cached"
            f"{plan['estimated_neurons']:>10.2f} estimated neurons"
        )
    print(f"  {'total':<12}{'':>6}{'':>7}{'':>7}{total:>10.2f} estimated neurons")
    return total


def print_results(result: dict) -> None:
    print(
        f"\nconfiguration {result['config']} (pooling {result['pooling']}, clip {result['max_chars']}), "
        f"split {result['split']}"
    )
    header = f"  {'ordering':<10}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR@10':>9}{'recall@20':>11}"
    print(header)
    for ordering, m in result["overall"].items():
        print(
            f"  {ordering:<10}{m['questions']:>4}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}"
            f"{m['mrr@10']:>9.3f}{m['recall@20']:>11.3f}"
        )
    for name, repo in result["repos"].items():
        print(f"  {name}: {repo['files_indexed']} files, {repo['records_indexed']} records")
        for scope, counts in sorted(repo["clipped"].items()):
            share = counts["clipped"] / counts["records"] if counts["records"] else 0.0
            kept = counts["chars_kept"] / counts["chars"] if counts["chars"] else 1.0
            print(
                f"    {scope + ' records':<20}{counts['records']:>5}  over the clip: {counts['clipped']:>4}"
                f" ({share:.1%})  characters kept: {kept:.1%}"
            )
        for ordering, m in repo["orderings"].items():
            print(
                f"    {ordering:<10}{m['questions']:>4}{m['hit@1']:>8.3f}{m['hit@3']:>8.3f}{m['hit@5']:>8.3f}"
                f"{m['mrr@10']:>9.3f}{m['recall@20']:>11.3f}"
            )
    print(f"  cache: {result['cache']['hits']} hits, {result['cache']['misses']} embedded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--config", dest="configs", action="append", choices=sorted(CONFIGS), help="default: all")
    parser.add_argument("--set", dest="sets", action="append", type=Path, help="question file (default: all)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where repositories are checked out")
    parser.add_argument("--vector-cache", type=Path, default=VECTOR_CACHE_DIR)
    parser.add_argument("--json", type=Path, help="also write the full results here")
    parser.add_argument("--plan-only", action="store_true", help="print the estimate and send nothing")
    args = parser.parse_args()

    app_config = get_config()
    budget = app_config.cloudflare_daily_neurons
    question_files = args.sets or DEFAULT_SETS
    configs = [CONFIGS[name] for name in (args.configs or sorted(CONFIGS))]

    if load_cloudflare_credentials() is None:
        print("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must both be set (.env is read).")
        return 1

    print(f"split: {args.split}")
    print(f"configurations: {', '.join(c.name for c in configs)}")
    plans = [plan_for(c, question_files, args.split, app_config, args.vector_cache, args.cache_dir) for c in configs]
    total = print_plan(plans, budget)

    if args.plan_only:
        print("\nplan only: nothing was sent")
        return 0

    spent = neuron_ledger.spent_today().estimated_neurons
    # The same check the provider itself makes before every batch, asked once for the whole run
    # so a run that cannot finish does not start and spend half a budget finding out.
    if not neuron_ledger.affordable(total, budget):
        print(
            f"\nrefusing to start: {total:.2f} estimated neurons on top of today's {spent:.2f} "
            f"would pass the budget of {budget:g}."
        )
        return 1

    results = [evaluate(c, question_files, args.split, app_config, args.vector_cache, args.cache_dir) for c in configs]
    for result in results:
        print_results(result)

    after = neuron_ledger.spent_today()
    print(f"\nledger after this run: {after.estimated_neurons:.2f} estimated neurons, {after.calls} calls")
    print(f"remaining in today's budget: {max(budget - after.estimated_neurons, 0.0):.2f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "configs": [r["config"] for r in results],
            "results": results,
            "ledger_after": {"estimated_neurons": after.estimated_neurons, "calls": after.calls, "day": after.day},
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
