"""Run the chosen retrieval configuration through the **server's own path**, and cost it.

    python scripts/adoption_check.py
    python scripts/adoption_check.py --json eval/results/experiments/adoption-hash.json

`scripts/experiment_retrieval.py` scores five arrangements by building them itself: it forces
`rerank_weight` and `bm25_enabled` on an `AnswerService` of its own and passes the record format
in by hand. That is the right shape for an experiment and the wrong shape for the question this
script asks, which is **whether a server configured from the environment retrieves the same way**.

So nothing here is told what to do. The embedder is `di.get_embedder()`, the index is built with
`di.index_options()`, the retrieval settings are `di.get_retrieval_settings()`, and the search is
`AnswerService.retrieve` - the same four calls `get_retrieval_service()` makes. Set the
environment and this reproduces the recorded numbers, or it does not and says so.

Three things are checked, and the exit status is 1 if any of them fails:

* **the scores**, against the committed experiment files, on all three splits;
* **the cost of a record format**, measured both ways per repository: records, bytes of index
  on disk, and the seconds indexing took;
* **the index guard**, which must refuse an index whose record format is not the one configured
  rather than searching it.

No default is changed and nothing is written outside `--json`. `hash` only: this script must not
reach a provider, and it refuses to run under one that would.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.app.di import (  # noqa: E402
    build_embedder,
    get_config,
    get_embedder,
    get_retrieval_settings,
    index_options,
)
from codeatlas.models.repository import Repository  # noqa: E402
from codeatlas.services.eval.basic_eval import RECALL_DEPTH, ranking_metrics  # noqa: E402
from codeatlas.services.eval.retrieval_eval import (  # noqa: E402
    RANKED_DEPTH,
    checkout,
    describe_embedder,
    load_question_set,
)
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser  # noqa: E402
from codeatlas.services.qa.answer_service import AnswerService  # noqa: E402
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever  # noqa: E402
from codeatlas.services.retrieval.indexing import CodeIndexService  # noqa: E402
from codeatlas.services.retrieval.record_format import STRUCTURED, WHOLE  # noqa: E402
from codeatlas.utils.config import every_index_kind_on  # noqa: E402

REPO_CACHE = ROOT / ".codeatlas" / "eval-repos"
EXPERIMENTS = ROOT / "eval" / "results" / "experiments"
QUESTIONS = ROOT / "eval" / "questions"

ORIGINAL_SETS = [QUESTIONS / "itsdangerous.json", QUESTIONS / "flask.json"]
HOLDOUT2_SETS = [QUESTIONS / "holdout2" / "itsdangerous.json", QUESTIONS / "holdout2" / "flask.json"]

# Which committed experiment file each split's numbers have to match, and the arrangement inside
# it. The arrangement name is what the experiment called this configuration; the server has no
# such name, which is exactly why the two agreeing is worth checking.
EXPECTED = {
    "dev": ("structured-dev-hash.json", "rrf+keyword", ORIGINAL_SETS),
    "test": ("chosen-heldout-hash.json", "rrf+keyword", ORIGINAL_SETS),
    "holdout2": ("holdout2-chosen-hash.json", "rrf+keyword", HOLDOUT2_SETS),
}

# The environment variables that decide what is being checked, printed with their effective
# values so a reader can reproduce the run without reading this file. These are settings, not
# credentials, and none of them is secret.
REPORTED_ENV = (
    "CODEATLAS_EMBEDDING_PROVIDER",
    "CODEATLAS_RECORD_FORMAT",
    "CODEATLAS_HASH_RECORD_FORMAT",
    "CODEATLAS_BM25",
    "CODEATLAS_HASH_BM25",
    "CODEATLAS_RECORD_WINDOW_CHARS",
    "CODEATLAS_RECORD_WINDOW_OVERLAP",
    "CODEATLAS_RETRIEVAL_CANDIDATES",
    "CODEATLAS_RERANK_WEIGHT",
    "CODEATLAS_BM25_K1",
    "CODEATLAS_BM25_B",
    "CODEATLAS_BM25_RRF_K",
    "CODEATLAS_BM25_DEPTH",
    "CODEATLAS_DROP_STOPWORDS",
    "CODEATLAS_EMBED_PREFIX_METADATA",
    "CODEATLAS_EMBED_MAX_CHARS",
)

METRICS = ("hit@1", "hit@3", "hit@5", f"mrr@{RANKED_DEPTH}", f"recall@{RECALL_DEPTH}")


def describe(embedder) -> str:
    """The embedder's description, with the record-format wrapper named rather than hiding it.

    `describe_embedder` matches on the concrete provider class, and under a non-default record
    format the provider is wrapped (see record_format.with_record_format), so it would otherwise
    report the wrapper's class name and say nothing about which embedder is inside it.
    """
    inner = getattr(embedder, "inner", None)
    if inner is None:
        return describe_embedder(embedder)
    return f"{describe_embedder(inner)}, records {embedder.signature().rsplit('records=', 1)[-1]}"


def effective_env(config) -> list[dict]:
    """Each reported setting: what the environment said, and what the config made of it."""
    values = {
        "CODEATLAS_EMBEDDING_PROVIDER": config.embedding_provider,
        "CODEATLAS_RECORD_FORMAT": config.record_format,
        # The effective column is what the *hash index* is built and searched as, which is the
        # only index this script builds, rather than the raw per-kind value: an unset per-kind
        # variable is not a setting of its own, it is the global one one line above.
        "CODEATLAS_HASH_RECORD_FORMAT": config.record_format_for("hash"),
        "CODEATLAS_BM25": str(config.bm25_enabled).lower(),
        "CODEATLAS_HASH_BM25": str(config.bm25_enabled_for("hash")).lower(),
        "CODEATLAS_RECORD_WINDOW_CHARS": config.record_window_chars,
        "CODEATLAS_RECORD_WINDOW_OVERLAP": config.record_window_overlap,
        "CODEATLAS_RETRIEVAL_CANDIDATES": config.retrieval_candidates,
        "CODEATLAS_RERANK_WEIGHT": config.rerank_weight,
        "CODEATLAS_BM25_K1": config.bm25_k1,
        "CODEATLAS_BM25_B": config.bm25_b,
        "CODEATLAS_BM25_RRF_K": config.bm25_rrf_k,
        "CODEATLAS_BM25_DEPTH": config.bm25_depth,
        "CODEATLAS_DROP_STOPWORDS": str(config.drop_stopwords).lower(),
        "CODEATLAS_EMBED_PREFIX_METADATA": str(config.embed_prefix_metadata).lower(),
        "CODEATLAS_EMBED_MAX_CHARS": 0 if config.embed_max_chars is None else config.embed_max_chars,
    }
    return [
        {"name": name, "set": os.getenv(name), "effective": values[name], "from_default": os.getenv(name) is None}
        for name in REPORTED_ENV
    ]


def repository_of(question_set, root: Path) -> Repository:
    return Repository(
        repo_id=f"eval-{question_set.name}",
        name=question_set.name,
        url=question_set.repo_url,
        root_path=str(root.resolve()),
        ingested_at=datetime.now(UTC),
        commit=question_set.commit,
    )


def server_index(config, embedder, retriever) -> CodeIndexService:
    """The index service `di.get_index_service()` builds, minus the Cloudflare fallback half."""
    return CodeIndexService(
        embedder=embedder, retriever=retriever, name=config.embedding_provider, **index_options(config)
    )


def score(config, embedder, settings, question_files, split: str) -> dict:
    """Every question in the split through `AnswerService.retrieve`, scored per repo and overall."""
    depth = max(RANKED_DEPTH, RECALL_DEPTH, settings.candidates)
    repos: dict[str, dict] = {}
    pairs: list[tuple[list[str], set[str]]] = []
    pools: list[tuple[list[str], set[str]]] = []

    for path in question_files:
        question_set = load_question_set(path)
        root = checkout(question_set, REPO_CACHE)
        repository = repository_of(question_set, root)
        parsed = TreeSitterAstParser().parse_repository(repository)
        retriever = FaissCodeRetriever()
        server_index(config, embedder, retriever).index_repository(repository, parsed)
        # The retrieval half of the service the server answers with: same retriever, same
        # embedder, same settings, no model. See di.get_retrieval_service.
        answers = AnswerService(retriever=retriever, embedder=embedder, llm=None, settings=settings)

        repo_pairs: list[tuple[list[str], set[str]]] = []
        repo_pools: list[tuple[list[str], set[str]]] = []
        for q in question_set.select(split):
            records = answers.retrieve(repository.repo_id, q.question, top_k=depth)
            ranked = [_relative(r.metadata.get("path", ""), root) for r in records]
            expected = set(q.expected_files)
            repo_pairs.append((ranked[:RANKED_DEPTH], expected))
            repo_pools.append((ranked, expected))
        pairs.extend(repo_pairs)
        pools.extend(repo_pools)
        repos[question_set.name] = _metrics(repo_pairs, repo_pools)

    return {"split": split, "repos": repos, "overall": _metrics(pairs, pools)}


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


def compare(measured: dict, expected: dict) -> list[dict]:
    """One line per metric that differs. An empty list is the whole of "it matches"."""
    return [
        {"metric": name, "measured": measured.get(name), "expected": expected.get(name)}
        for name in METRICS
        if measured.get(name) != expected.get(name)
    ]


def index_cost(config, question_files) -> list[dict]:
    """Records, bytes of index on disk and indexing seconds, per repository and record format.

    The index is written to a temporary directory so that the size is the real one - a FAISS
    index and its record sidecar - and so that nothing here touches the server's index folder.
    """
    rows: list[dict] = []
    for path in question_files:
        question_set = load_question_set(path)
        root = checkout(question_set, REPO_CACHE)
        repository = repository_of(question_set, root)
        parsed = TreeSitterAstParser().parse_repository(repository)
        for record_format in (WHOLE, STRUCTURED):
            formatted = every_index_kind_on(config, record_format)
            embedder = build_embedder(formatted)
            with tempfile.TemporaryDirectory(prefix="codeatlas-adoption-") as tmp:
                retriever = FaissCodeRetriever(base_dir=tmp)
                indexer = server_index(formatted, embedder, retriever)
                started = time.perf_counter()
                indexer.index_repository(repository, parsed)
                seconds = time.perf_counter() - started
                size = sum(f.stat().st_size for f in Path(tmp).glob(f"{repository.repo_id}.*"))
                records = retriever.records(repository.repo_id)
                rows.append(
                    {
                        "repo": question_set.name,
                        "record_format": record_format,
                        "signature": embedder.signature(),
                        "source_files": len(parsed.files),
                        "parsed_functions": len(parsed.functions),
                        "records": len(records),
                        "file_records": sum(1 for r in records if r.scope == "file"),
                        "function_records": sum(1 for r in records if r.scope != "file"),
                        "index_bytes": size,
                        "index_seconds": round(seconds, 3),
                    }
                )
    return rows


def guard_check(config, question_file: Path) -> dict:
    """An index built under one record format, searched by a server running the other.

    Both directions, because the guard has to refuse in both: a `structured` server must not
    read a `whole` index, and a `whole` server must not read a `structured` one. "Refused"
    means no records and no search, not a worse ranking.
    """
    question_set = load_question_set(question_file)
    root = checkout(question_set, REPO_CACHE)
    repository = repository_of(question_set, root)
    parsed = TreeSitterAstParser().parse_repository(repository)
    question = question_set.questions[0].question
    settings = get_retrieval_settings()

    built: dict[str, tuple[FaissCodeRetriever, object]] = {}
    for record_format in (WHOLE, STRUCTURED):
        formatted = every_index_kind_on(config, record_format)
        embedder = build_embedder(formatted)
        retriever = FaissCodeRetriever()
        server_index(formatted, embedder, retriever).index_repository(repository, parsed)
        built[record_format] = (retriever, embedder)

    cases = []
    for index_format in (WHOLE, STRUCTURED):
        retriever, _ = built[index_format]
        for server_format in (WHOLE, STRUCTURED):
            _, embedder = built[server_format]
            answers = AnswerService(retriever=retriever, embedder=embedder, llm=None, settings=settings)
            found = len(answers.retrieve(repository.repo_id, question, top_k=5))
            service = server_index(every_index_kind_on(config, server_format), embedder, retriever)
            cases.append(
                {
                    "index_built_as": index_format,
                    "server_configured_as": server_format,
                    "stored_signature": retriever.stored_embedder(repository.repo_id),
                    "server_signature": embedder.signature(),
                    "records_returned": found,
                    "index_cache_reuses_it": service.has_index(repository.repo_id),
                    "refused": found == 0,
                }
            )
    matched = [c for c in cases if c["index_built_as"] == c["server_configured_as"]]
    mixed = [c for c in cases if c["index_built_as"] != c["server_configured_as"]]
    return {
        "repo": question_set.name,
        "question_id": question_set.questions[0].id,
        "cases": cases,
        "passes": all(c["records_returned"] > 0 for c in matched) and all(c["refused"] for c in mixed),
    }


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def print_environment(env: list[dict]) -> None:
    print("\nenvironment")
    print(f"  {'variable':<34}{'set to':>12}{'effective':>14}  source")
    for row in env:
        shown = "-" if row["set"] is None else str(row["set"])
        source = "default" if row["from_default"] else "environment"
        print(f"  {row['name']:<34}{shown:>12}{str(row['effective']):>14}  {source}")


def print_costs(costs: list[dict]) -> None:
    print("\nwhat a record format costs, per repository")
    header = f"  {'repo':<14}{'format':<12}{'files':>7}{'records':>9}{'file':>7}{'function':>10}{'index KB':>10}{'seconds':>9}"
    print(header)
    for row in costs:
        print(
            f"  {row['repo']:<14}{row['record_format']:<12}{row['source_files']:>7}{row['records']:>9}"
            f"{row['file_records']:>7}{row['function_records']:>10}{row['index_bytes'] / 1024:>10.1f}"
            f"{row['index_seconds']:>9.2f}"
        )


def print_guard(guard: dict) -> None:
    print("\nthe index guard, on a mixed format")
    print(f"  {'index built as':<18}{'server runs':<14}{'records':>9}{'cache reuse':>13}  verdict")
    for case in guard["cases"]:
        mixed = case["index_built_as"] != case["server_configured_as"]
        if mixed:
            verdict = "refused" if case["refused"] else "SEARCHED IT"
        else:
            verdict = "searched" if case["records_returned"] else "FOUND NOTHING"
        print(
            f"  {case['index_built_as']:<18}{case['server_configured_as']:<14}{case['records_returned']:>9}"
            f"{str(case['index_cache_reuses_it']):>13}  {verdict}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    config = get_config()
    if config.embedding_provider != "hash":
        print(
            f"This check runs on the hash embedder only; CODEATLAS_EMBEDDING_PROVIDER is "
            f"{config.embedding_provider!r}. It must not reach a provider."
        )
        return 1

    embedder = get_embedder()
    settings = get_retrieval_settings()
    started = time.perf_counter()

    print(f"codementor commit: {_commit()}")
    print(f"embedder: {describe(embedder)}")
    print(f"signature: {embedder.signature()}")
    env = effective_env(config)
    print_environment(env)

    print("\nscores through AnswerService, against the committed experiment files")
    print(f"  {'split':<10}{'scope':<14}{'n':>4}" + "".join(f"{m:>12}" for m in METRICS) + "   against the file")
    splits: list[dict] = []
    ok = True
    for split, (filename, arrangement, question_files) in EXPECTED.items():
        recorded = json.loads((EXPERIMENTS / filename).read_text(encoding="utf-8"))
        measured = score(config, embedder, settings, question_files, split)
        rows = []
        for scope in (*measured["repos"], "overall"):
            mine = measured["overall"] if scope == "overall" else measured["repos"][scope]
            theirs = (
                recorded["overall"][arrangement]
                if scope == "overall"
                else recorded["repos"][scope]["arrangements"][arrangement]
            )
            differences = compare(mine, theirs)
            ok = ok and not differences
            rows.append({"scope": scope, "measured": mine, "expected": theirs, "differences": differences})
            cells = "".join(f"{mine[m]:>12.4f}" for m in METRICS)
            verdict = "same" if not differences else "DIFFERS"
            print(f"  {split:<10}{scope:<14}{mine['questions']:>4}{cells}   {verdict}")
        splits.append(
            {
                "split": split,
                "expected_source": f"eval/results/experiments/{filename}",
                "arrangement": arrangement,
                "rows": rows,
            }
        )

    costs = index_cost(config, ORIGINAL_SETS)
    print_costs(costs)

    guard = guard_check(config, ORIGINAL_SETS[0])
    print_guard(guard)
    ok = ok and guard["passes"]

    result = {
        "codementor_commit": _commit(),
        "embedder": describe(embedder),
        "signature": embedder.signature(),
        "environment": env,
        "retrieval_settings": vars(settings),
        "splits": splits,
        "index_cost": costs,
        "index_guard": guard,
        "matches_experiments": ok,
        "seconds": round(time.perf_counter() - started, 1),
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    print(f"\nseconds: {result['seconds']}")
    if ok:
        print("MATCH: the server path reproduces every recorded number, and the guard refuses a mixed index.")
        return 0
    print("MISMATCH: see the lines marked above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
