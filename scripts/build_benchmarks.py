"""Build the one committed summary of recorded results that the website shows.

    python scripts/build_benchmarks.py                  # write eval/benchmarks.json
    python scripts/build_benchmarks.py --check          # fail if the committed file is stale

Everything in the summary is either read from a committed result file, computed from this
repository's own app, or quoted from a `docs/RESULTS.md` entry named in PROVENANCE below.
**No model and no API is called here, and nothing is measured at request time.** The retrieval,
structural and endpoint figures are recomputed on every build, so a result file that changes or a
route that is added makes `tests/test_build_benchmarks.py` fail until the summary is rebuilt.

The figures that are quoted rather than read are the ones whose only record is prose: the
model-free chat timings, the call-graph coverage, and the answer smoke check. Each carries the
heading of the entry it came from, and a test asserts that heading is still in `docs/RESULTS.md`.

Deliberately absent, and all four are requirements rather than oversights:
  * end-to-end latency - it is the model provider's, not this system's, and is not benchmarked;
  * any question text, from the eval sets or from a visitor;
  * any key, token or account id;
  * any budget or neuron figure.
There is also no "generated at" timestamp: the file is a pure function of its inputs, which is
what lets the test compare a fresh build against the committed copy byte for byte.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.services.retrieval.cloudflare_embedder import (  # noqa: E402
    DEFAULT_CLOUDFLARE_MAX_DOCUMENTS,
    DEFAULT_CLOUDFLARE_MODEL,
)

SUMMARY = ROOT / "eval" / "benchmarks.json"
RESULTS = ROOT / "eval" / "results"

SCHEMA = 1

HEADING_NOTE = "Recorded results on small fixed question sets, not live accuracy."
END_TO_END_NOTE = (
    "End-to-end latency is not shown. It depends on the model provider and is not benchmarked here."
)

# Pinned repositories. The two of them are the whole of every question set below.
REPOS = {
    "itsdangerous": {"url": "https://github.com/pallets/itsdangerous", "commit": "672971d66a2e"},
    "flask": {"url": "https://github.com/pallets/flask", "commit": "d73fa1cdcbd8"},
}

# When each block was recorded, and the docs/RESULTS.md entry that records it. A result file says
# which commit of this repository produced it but not when, and a figure quoted from prose says
# neither, so both live here. tests/test_build_benchmarks.py checks every heading still exists.
PROVENANCE = {
    "retrieval.hash": {
        "commit": "016c655",
        "date": "2026-10-11",
        "results_entry": "Change: the committed hash retrieval results carry recall@20",
    },
    "retrieval.cloudflare.dev": {
        "commit": "149dac1",
        "date": "2026-10-09",
        "results_entry": "Check: three Cloudflare configurations on the dev split, three orderings each",
    },
    "retrieval.cloudflare.held-out": {
        "commit": "b1bbfd4",
        "date": "2026-10-10",
        "results_entry": "Check: the held-out split, run once, for the default configuration",
    },
    "structural": {
        "commit": "28fb0e2",
        "date": "2026-10-11",
        "results_entry": "Change: the structural eval can write a result file, and that file is committed",
    },
    "endpoints": {
        "commit": "7831330",
        "date": "2026-10-11",
        "results_entry": "Feature: a read-only GET /eval/benchmarks that serves recorded results",
    },
    "call_graph": {
        "commit": "8336875",
        "date": "2026-10-05",
        "results_entry": "Feature: deterministic call graph for Python, and an analyst that answers from it",
    },
    "timing.stages": {
        "commit": "64c7ae6",
        "date": "2026-10-04",
        "results_entry": "Feature: per-stage timing for indexing and chat",
    },
    "timing.tuned": {
        "commit": "513641c",
        "date": "2026-10-05",
        "results_entry": "Measurement: retrieval tuning experiments (dev set), confirmed on the held-out test set",
    },
    "smoke.fast": {
        "commit": "f67b5d4",
        "date": "2026-10-06",
        "results_entry": "Check: deep-mode answer smoke test rerun, with the quota damage removed",
    },
    "smoke.deep": {
        "commit": "d01ecba",
        "date": "2026-10-06",
        "results_entry": "Check: deep-mode smoke test after the mentor-first fix — **the score did not improve**",
    },
    "smoke.partial": {
        "commit": "32a9aba",
        "date": "2026-10-06",
        "results_entry": "Check: third deep smoke pass",
    },
    "document_cap": {
        "commit": "6933310",
        "date": "2026-10-11",
        "results_entry": "Check: what a full 1,705-document flask index would cost, with the cap left at 600",
    },
}

METRICS = ["hit@1", "hit@3", "hit@5", "p@3", "mrr@10", "recall@20"]

SPLIT_LABELS = {"dev": "development", "test": "held-out"}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(scope: str, metrics: dict) -> dict:
    """One line of a retrieval table. A metric the run did not compute is null, never 0."""
    return {"scope": scope, "questions": metrics["questions"], **{m: metrics.get(m) for m in METRICS}}


# ---------------------------------------------------------------- retrieval


def hash_retrieval() -> dict:
    """scripts/eval_retrieval.py with the default (hash) embedder, both splits."""
    splits = []
    for name in ("dev", "test"):
        data = _read(RESULTS / f"{name}.json")
        rows = []
        for repo in REPOS:
            m = data["repos"][repo]
            rows.append(
                _row(
                    repo,
                    {
                        "questions": m["questions"],
                        "hit@1": m["hit_rate"]["@1"],
                        "hit@3": m["hit_rate"]["@3"],
                        "hit@5": m["hit_rate"]["@5"],
                        "p@3": m["precision"]["@3"],
                        "mrr@10": m["mrr@10"],
                        "recall@20": m["recall@20"],
                    },
                )
            )
        o = data["overall"]
        rows.append(
            _row(
                "overall",
                {
                    "questions": o["questions"],
                    "hit@1": o["hit_rate"]["@1"],
                    "hit@3": o["hit_rate"]["@3"],
                    "hit@5": o["hit_rate"]["@5"],
                    "p@3": o["precision"]["@3"],
                    "mrr@10": o["mrr@10"],
                    "recall@20": o["recall@20"],
                },
            )
        )
        splits.append(
            {
                "id": SPLIT_LABELS[name],
                "questions": o["questions"],
                "source": f"eval/results/{name}.json",
                "recorded": PROVENANCE["retrieval.hash"],
                "result_file_commit": data["codementor_commit"],
                "rows": rows,
            }
        )

    return {
        "id": "hash",
        "label": "hash",
        "default": True,
        "embedder": _read(RESULTS / "dev.json")["embedder"],
        "note": "The server's default. A hashed bag of words over identifier subtokens, not a semantic embedding.",
        "splits": splits,
    }


def cloudflare_retrieval() -> dict:
    """scripts/eval_cloudflare.py, the default configuration and the ordering the server uses."""
    config, ordering = "mean-2000", "keyword"
    splits, settings = [], {}
    for name, key in (("dev", "retrieval.cloudflare.dev"), ("test", "retrieval.cloudflare.held-out")):
        data = _read(RESULTS / "cloudflare" / f"{name}.json")
        chosen = next(r for r in data["results"] if r["config"] == config)
        settings = {"pooling": chosen["pooling"], "max_chars": chosen["max_chars"], "ordering": ordering}
        rows = [_row(repo, chosen["repos"][repo]["orderings"][ordering]) for repo in REPOS]
        rows.append(_row("overall", chosen["overall"][ordering]))
        splits.append(
            {
                "id": SPLIT_LABELS[name],
                "questions": chosen["overall"][ordering]["questions"],
                "source": f"eval/results/cloudflare/{name}.json",
                "recorded": PROVENANCE[key],
                "rows": rows,
            }
        )

    return {
        "id": "cloudflare",
        "label": "cloudflare (opt-in)",
        "default": False,
        "embedder": (
            f"cloudflare ({DEFAULT_CLOUDFLARE_MODEL}, {settings['pooling']} pooling, "
            f"clip {settings['max_chars']:,} characters)"
        ),
        "settings": settings,
        "note": (
            "Off unless CODEATLAS_EMBEDDING_PROVIDER=cloudflare. Ordered by keyword overlap, as the server "
            f"orders. Every record longer than {settings['max_chars']:,} characters was embedded by its opening "
            "alone, so this is the server running this provider rather than the embedder in isolation. "
            "P@3 was not computed on the development run."
        ),
        "splits": splits,
    }


def retrieval() -> dict:
    return {
        "title": "Retrieval",
        "question_set": "eval/questions/*.json — 40 questions, 20 per repository, fixed into two splits before any run",
        "scored": (
            "The retrieval step of the chat: embed the question, take the nearest records, rerank them by "
            "keyword overlap. A record counts as relevant when its file is one of the question's expected files."
        ),
        "metrics": METRICS,
        "providers": [hash_retrieval(), cloudflare_retrieval()],
    }


# ---------------------------------------------------------------- structural


def structural() -> dict:
    """scripts/structural_eval.py: the call graph against Python's own ast module."""
    data = _read(RESULTS / "structural" / "all.json")
    return {
        "title": "Structural analysis against Python's ast",
        "language": "python",
        "note": (
            "Python only. `ast` is a different parser by a different author, so the two agree or they do not. "
            "Precision of 1.000 is partly structural: the graph only claims a target it resolved through "
            "imports and scope, and pays for that quiet in recall."
        ),
        "source": "eval/results/structural/all.json",
        "recorded": PROVENANCE["structural"],
        "result_file_commit": data["codementor_commit"],
        "questions": data["overall"]["questions"],
        "overall": data["overall"],
        "by_split": data["by_split"],
        "by_repo": data["by_repo"],
        "by_kind": data["by_kind"],
        "disagreements": data["disagreements"],
        "disagreements_note": (
            "All of them are misses: no disagreement has the graph claiming a call that `ast` did not find. "
            "Two of the four are `ast` over-counting, because it matches a call by its trailing name and is "
            "not type-aware."
        ),
    }


# ---------------------------------------------------------------- system facts


def _count_endpoints() -> dict:
    """scripts/count_endpoints.py, imported by path the way its own test imports it."""
    spec = importlib.util.spec_from_file_location("count_endpoints", ROOT / "scripts" / "count_endpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count()


def system() -> dict:
    counted = _count_endpoints()
    return {
        "title": "System facts",
        "endpoints": {
            "routers": counted["routers"],
            "routes": counted["routes"],
            "open": counted["routes_open"],
            "need_a_key": counted["routes_needing_key"],
            "rate_limited": counted["routes_rate_limited"],
            "source": "scripts/count_endpoints.py, over this repository's own app",
            "note": "Counted from create_app(). Nothing is started and no request is made.",
            "recorded": PROVENANCE["endpoints"],
        },
        "call_graph": {
            "recorded": PROVENANCE["call_graph"],
            "source": "docs/RESULTS.md, CallGraph.build() over the two pinned checkouts",
            "repos": {
                "itsdangerous": {
                    "modules": 15,
                    "definitions": 135,
                    "call_sites": 378,
                    "resolved": 91,
                    "resolved_share": 0.241,
                    "import_cycle_groups": 0,
                    "build_seconds": 0.03,
                },
                "flask": {
                    "modules": 83,
                    "definitions": 1580,
                    "call_sites": 3906,
                    "resolved": 742,
                    "resolved_share": 0.190,
                    "import_cycle_groups": 1,
                    "build_seconds": 0.17,
                },
            },
            "note": (
                "The resolved share is not an accuracy figure. Most call sites in any repository target the "
                "standard library or a dependency, and those are correctly outside the graph."
            ),
        },
    }


# ---------------------------------------------------------------- timing


def timing() -> dict:
    """Chat stages with the language model excluded. Never end-to-end latency."""
    return {
        "title": "Model-free chat timing",
        "label": "excludes the language model",
        "units": "milliseconds, median",
        "excludes_the_model": True,
        "end_to_end_note": END_TO_END_NOTE,
        "current": {
            "recorded": PROVENANCE["timing.tuned"],
            "source": "docs/RESULTS.md, scripts/time_stages.py",
            "note": (
                "At the settings the server ships (search depth 20), one warm itsdangerous chat question "
                "takes 45 ms with the model excluded, of which the rerank is 19 ms."
            ),
            "warm_total_ms": 45.0,
            "rerank_ms": 19.0,
            "repo": "itsdangerous",
        },
        "stages": {
            "recorded": PROVENANCE["timing.stages"],
            "source": "docs/RESULTS.md, scripts/time_stages.py, 3 processes cold / 6 runs warm",
            "note": (
                "Measured before search depth rose from 10 to 20, so the totals here are lower than the "
                "shipped settings give; the breakdown is what the stages cost relative to one another. "
                "The model was not timed: no provider key was set, so a stand-in answered instantly and "
                "the llm figure is LangChain overhead only."
            ),
            "columns": ["embed_query", "search", "rerank", "llm_stand_in", "first_token_sent", "total"],
            "rows": [
                {"case": "itsdangerous cold", "values": [0.4, 3.5, 5.7, 11.0, 27.0, 37.0]},
                {"case": "itsdangerous warm", "values": [0.3, 0.4, 4.7, 3.1, 12.2, 25.5]},
                {"case": "flask cold", "values": [0.3, 4.1, 7.1, 7.9, 24.9, 34.4]},
                {"case": "flask warm", "values": [0.3, 0.6, 6.5, 3.0, 15.5, 32.8]},
            ],
        },
    }


# ---------------------------------------------------------------- limitations


def limitations() -> dict:
    return {
        "title": "Limitations and weak checks",
        "collapsed": True,
        "smoke_check": {
            "title": "Answer smoke check",
            "weak": True,
            "what_it_is": (
                "Two string tests per answer: is the expected file among the citations, and does the answer "
                "contain one short hand-written word. There is no judge model and nothing reads whether the "
                "explanation is correct, complete or well reasoned. A wrong answer can pass; a good one fails "
                "on a synonym."
            ),
            "questions": 15,
            "model": "openai/gpt-oss-20b",
            "modes": [
                {
                    "mode": "fast",
                    "passed": 8,
                    "of": 15,
                    "expected_file_cited": 9,
                    "fact_present": 13,
                    "right_fact_wrong_file": 5,
                    "recorded": PROVENANCE["smoke.fast"],
                },
                {
                    "mode": "deep",
                    "passed": 3,
                    "of": 15,
                    "expected_file_cited": 7,
                    "fact_present": 9,
                    "right_fact_wrong_file": 6,
                    "recorded": PROVENANCE["smoke.deep"],
                },
            ],
            "note": (
                "The last complete pass of each mode. A third deep pass reached only 8 of the 15 questions "
                "before the provider's daily limit stopped it, and is not comparable with either. "
                "Deep mode is worse than fast mode on this check and the reason is not settled."
            ),
            "partial_pass": PROVENANCE["smoke.partial"],
        },
        "sample_sizes": [
            {"what": "retrieval", "questions": 40, "detail": "20 per repository, 10 development and 10 held out each"},
            {"what": "structural", "questions": 49, "detail": "24 itsdangerous and 25 flask, 26 development / 23 held out"},
            {"what": "answer smoke check", "questions": 15, "detail": "run in each of the two chat modes"},
        ],
        "sample_size_note": (
            "On a split of 20 questions one question is 0.05 of a hit rate, so a difference of two or three "
            "questions is not a result. The subjects were chosen by hand, which is a bias a held-out split "
            "narrows but does not remove."
        ),
        "repos": [
            {"name": name, "url": repo["url"], "commit": repo["commit"]} for name, repo in REPOS.items()
        ],
        "repos_note": (
            "Two Python libraries by the same authors. Nothing here says how the system behaves on another "
            "language, a larger repository, or code written in a different style."
        ),
        "document_cap": {
            "documents": DEFAULT_CLOUDFLARE_MAX_DOCUMENTS,
            "applies_to": "the semantic (cloudflare) provider",
            "recorded": PROVENANCE["document_cap"],
            "note": (
                f"A capped index keeps the first {DEFAULT_CLOUDFLARE_MAX_DOCUMENTS} documents in collection order - every file record, then "
                "functions - which is not a random sample. Every retrieval figure above was measured on the "
                "uncapped corpus; what a capped index would score has not been measured."
            ),
        },
    }


# ---------------------------------------------------------------- the summary


def build() -> dict:
    return {
        "schema": SCHEMA,
        "generated_by": "scripts/build_benchmarks.py",
        "heading_note": HEADING_NOTE,
        "end_to_end_note": END_TO_END_NOTE,
        "repos": REPOS,
        "retrieval": retrieval(),
        "structural": structural(),
        "system": system(),
        "timing": timing(),
        "limitations": limitations(),
    }


def rendered(summary: dict) -> str:
    return json.dumps(summary, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SUMMARY)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file on disk is not what a build gives")
    args = parser.parse_args()

    text = rendered(build())
    if args.check:
        on_disk = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if on_disk != text:
            print(f"{args.out} is stale or hand-edited; rebuild it with: python scripts/build_benchmarks.py")
            return 1
        print(f"{args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
