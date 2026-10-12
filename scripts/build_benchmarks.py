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
EXPERIMENTS = RESULTS / "experiments"

# One dev run per (embedder, record format), each holding all five arrangements.
DEV_SOURCES = {
    ("hash", "whole"): "arrangements-dev-hash.json",
    ("hash", "structured"): "structured-dev-hash.json",
    ("cloudflare", "whole"): "arrangements-dev-cloudflare.json",
    ("cloudflare", "structured"): "structured-dev-cloudflare.json",
}
# The original held-out split, scored once for the chosen configuration only.
HELD_OUT_SOURCES = {"hash": "chosen-heldout-hash.json", "cloudflare": "chosen-heldout-cloudflare.json"}
# holdout2, scored once for today's default and once for the chosen configuration.
HOLDOUT2_SOURCES = {
    ("hash", "default"): "holdout2-default-hash.json",
    ("hash", "chosen"): "holdout2-chosen-hash.json",
    ("cloudflare", "default"): "holdout2-default-cloudflare.json",
    ("cloudflare", "chosen"): "holdout2-chosen-cloudflare.json",
}
# holdout3, the set that had never been scored at all until the entry that spent it. Four
# candidates, one run each, and one more scored for information that the rule barred from being
# selected - carried here too, because leaving it out would hide where the hash gain comes from.
HOLDOUT3_SOURCES = {
    ("hash", "default"): "holdout3-default-hash.json",
    ("hash", "chosen"): "holdout3-chosen-hash.json",
    ("cloudflare", "default"): "holdout3-default-cloudflare.json",
    ("cloudflare", "chosen"): "holdout3-chosen-cloudflare.json",
}
HOLDOUT3_INFORMATION = ("hash", "whole", "rrf+keyword", "holdout3-whole-bm25-hash.json")
# The metric each embedder's candidate must not go backwards on, fixed in docs/RESULTS.md before
# holdout3 was scored. Two different metrics on purpose: each embedder is guarded on the one this
# repository had already recorded the candidate costing it.
HOLDOUT3_GUARD = {"hash": "hit@5", "cloudflare": "recall@20"}
HOLDOUT3_METRICS = ["hit@1", "hit@3", "hit@5", "mrr@10", "recall@20"]

# The chat-mode comparison: the full run and **only** the full run. A two-question trial was made
# first, while the token accounting was still being fixed, and `chat-modes-trial.json` is kept as
# the record of it - but it is deliberately not a source here. Falling back to it would put a
# two-question, four-run block on the page under the same heading, the same badge and the same
# column names as the fifteen-question, thirty-run comparison, and the only thing distinguishing
# them would be a number a reader has to notice. If the full run is missing the panel shows no
# chat-mode block at all, which is the honest outcome.
CHAT_MODE_SOURCES = ("chat-modes.json",)
# How many questions and runs the full comparison is, so the block can say so rather than relying
# on a reader to multiply. A file that does not match these is not the full run.
CHAT_MODE_QUESTIONS = 15
CHAT_MODE_RUNS = 30

# What the server runs today, and what the decision entry chose, per embedder.
BASELINE = ("whole", "current")
CHOSEN = {"hash": ("structured", "rrf+keyword"), "cloudflare": ("structured", "current")}
# Arrangements the experiment can score and the application has no setting for. BM25 was built
# as a second retriever fused with the first, so "BM25 alone" is not a mode the server has.
NOT_CONFIGURABLE = ("bm25", "vector", "rrf")

SCHEMA = 1

HEADING_NOTE = "Recorded results on small fixed question sets, not live accuracy."
END_TO_END_NOTE = "End-to-end latency is not shown. It depends on the model provider and is not benchmarked here."

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
    "experiments": {
        "commit": "5edf06e",
        "date": "2026-10-11",
        "results_entry": "Check: the chosen configurations on the original held-out split and on holdout2",
    },
    "holdout3": {
        "commit": "ecf3744",
        "date": "2026-10-12",
        "results_entry": "Check: holdout3, scored once for each candidate, and the rule applied",
    },
    "chat_modes": {
        "commit": "075a613",
        "date": "2026-10-11",
        "results_entry": "Feature: a grounded chat mode",
    },
    "document_cap": {
        "commit": "6933310",
        "date": "2026-10-11",
        "results_entry": "Check: what a full 1,705-document flask index would cost, with the cap left at 600",
    },
}

METRICS = ["hit@1", "hit@3", "hit@5", "p@3", "mrr@10", "recall@20"]

SPLIT_LABELS = {"dev": "development", "test": "held-out"}

# Said once, here, and carried by every panel that shows a held-out number. The original held-out
# split was written at the same time as dev and has been scored in entry after entry since, so a
# configuration doing well on it is not being tested on code the process has never seen. holdout2
# was written after those runs and scored once, so it is the fresher of the two estimates - which
# is why it is the lower one.
REUSED_SPLIT_NOTE = (
    "The original held-out split was reused: it has been scored in at least six recorded entries "
    "since it was written, so it has informed choices before and is a weak confirmation rather "
    "than a clean one."
)
FRESH_SPLIT_NOTE = (
    "holdout2 and holdout3 were each written after those runs and each scored once. They are the "
    "fresher estimates, and they do not agree with one another: holdout2 is the lowest of the four "
    "sets and holdout3 lands between it and the reused split."
)
BOTH_SPLITS_NOTE = (
    "All three are shown side by side wherever a held-out number appears. The two fresh sets differ "
    "from each other by up to 0.217 of MRR@10 on the same code, so no single set of twenty questions "
    "settles a comparison on its own."
)


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
                "reused": name == "test",
                "note": REUSED_SPLIT_NOTE if name == "test" else "",
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
                "reused": name == "test",
                "note": REUSED_SPLIT_NOTE if name == "test" else "",
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
        "splits_note": f"{REUSED_SPLIT_NOTE} {FRESH_SPLIT_NOTE} All three are compared in the Experiments panel.",
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


# ---------------------------------------------------------------- experiments


EXPERIMENT_METRICS = ["hit@5", "mrr@10", "recall@20"]


def _scores(metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    return {
        "questions": metrics["questions"],
        **{name: metrics.get(name) for name in EXPERIMENT_METRICS},
    }


def _verdict(embedder: str, records: str, arrangement: str, dev: dict, baseline: dict) -> str:
    """The keep rule, applied: kept only if dev MRR rises and dev hit@5 does not fall."""
    if (records, arrangement) == BASELINE:
        return "baseline"
    if (records, arrangement) == CHOSEN[embedder]:
        return "chosen"
    met = dev["mrr@10"] > baseline["mrr@10"] and dev["hit@5"] >= baseline["hit@5"]
    if not met:
        return "rejected"
    if arrangement in NOT_CONFIGURABLE:
        return "met the rule; the server has no mode for it"
    return "met the rule; not chosen"


def experiments() -> dict:
    """Every retrieval experiment this batch ran, kept and rejected, from the files it wrote."""
    dev_runs = {key: _read(EXPERIMENTS / name) for key, name in DEV_SOURCES.items()}
    held_out = {key: _read(EXPERIMENTS / name) for key, name in HELD_OUT_SOURCES.items()}
    holdout2 = {key: _read(EXPERIMENTS / name) for key, name in HOLDOUT2_SOURCES.items()}
    holdout3_runs = {key: _read(EXPERIMENTS / name) for key, name in HOLDOUT3_SOURCES.items()}

    rows = []
    for embedder in ("hash", "cloudflare"):
        baseline = dev_runs[(embedder, "whole")]["overall"]["current"]
        for records in ("whole", "structured"):
            run = dev_runs[(embedder, records)]
            for arrangement, description in run["arrangements"].items():
                dev = run["overall"][arrangement]
                verdict = _verdict(embedder, records, arrangement, dev, baseline)
                row = {
                    "embedder": embedder,
                    "records": records,
                    "arrangement": arrangement,
                    "change": f"{records} records, {description}",
                    "verdict": verdict,
                    "dev": _scores(dev),
                    "held_out": None,
                    "holdout2": None,
                    "holdout3": None,
                    "recorded": PROVENANCE["experiments"],
                    "result_file_commit": run["codementor_commit"],
                }
                if verdict == "chosen":
                    row["held_out"] = _scores(held_out[embedder]["overall"][arrangement])
                    row["holdout2"] = _scores(holdout2[(embedder, "chosen")]["overall"][arrangement])
                    row["holdout3"] = _scores(holdout3_runs[(embedder, "chosen")]["overall"][arrangement])
                elif verdict == "baseline":
                    row["holdout2"] = _scores(holdout2[(embedder, "default")]["overall"][arrangement])
                    row["holdout3"] = _scores(holdout3_runs[(embedder, "default")]["overall"][arrangement])
                rows.append(row)

    return {
        "title": "Experiments",
        "note": (
            "Every retrieval arrangement this batch scored, kept and rejected. Two held-out sets "
            "are shown, and they are not equally good evidence."
        ),
        "keep_rule": "Kept only if dev MRR@10 rises and dev hit@5 does not fall.",
        "metrics": EXPERIMENT_METRICS,
        "splits": {
            "dev": "development - what the choice was made from",
            "held_out": "held out - reused, scored many times before",
            "holdout2": "fresh - written later, scored once",
            "holdout3": "never used - written after holdout2 was scored, scored once here",
        },
        "reused_split_note": REUSED_SPLIT_NOTE,
        "fresh_split_note": FRESH_SPLIT_NOTE,
        "both_splits_note": BOTH_SPLITS_NOTE,
        "source": "eval/results/experiments/",
        "rows": rows,
        "splits_compared": splits_compared(dev_runs, held_out, holdout2, holdout3_runs),
        "holdout3": holdout3(holdout3_runs),
        "chat_modes": chat_modes(),
    }


def splits_compared(dev_runs: dict, held_out: dict, holdout2: dict, holdout3_runs: dict) -> dict:
    """The three held-out sets beside each other, for the four configurations scored on all of them.

    The chosen rows carry their own numbers. A baseline's held-out figure does not live in the
    experiment files - it is the recorded retrieval run in the panel above - so it is read from
    there and the source of each number is named, rather than one of them being quietly filled
    in from somewhere a reader cannot check.
    """
    hash_held_out = _read(RESULTS / "test.json")["overall"]["mrr@10"]
    cloudflare_held_out = next(
        r for r in _read(RESULTS / "cloudflare" / "test.json")["results"] if r["config"] == "mean-2000"
    )["overall"]["keyword"]["mrr@10"]
    baseline_held_out = {
        "hash": (hash_held_out, "eval/results/test.json"),
        "cloudflare": (cloudflare_held_out, "eval/results/cloudflare/test.json"),
    }

    rows = []
    for embedder in ("hash", "cloudflare"):
        records, arrangement = CHOSEN[embedder]
        reused, source = baseline_held_out[embedder]
        fresh = holdout2[(embedder, "default")]["overall"][BASELINE[1]]["mrr@10"]
        rows.append(
            {
                "embedder": embedder,
                "configuration": "today's default",
                "change": f"{BASELINE[0]} records, {BASELINE[1]}",
                "verdict": "baseline",
                "dev": dev_runs[(embedder, BASELINE[0])]["overall"][BASELINE[1]]["mrr@10"],
                "held_out": reused,
                "holdout2": fresh,
                "holdout3": holdout3_runs[(embedder, "default")]["overall"][BASELINE[1]]["mrr@10"],
                "difference": round(fresh - reused, 4),
                "held_out_source": source,
                "holdout2_source": f"eval/results/experiments/{HOLDOUT2_SOURCES[(embedder, 'default')]}",
                "holdout3_source": f"eval/results/experiments/{HOLDOUT3_SOURCES[(embedder, 'default')]}",
            }
        )
        reused = held_out[embedder]["overall"][arrangement]["mrr@10"]
        fresh = holdout2[(embedder, "chosen")]["overall"][arrangement]["mrr@10"]
        rows.append(
            {
                "embedder": embedder,
                "configuration": "chosen",
                "change": f"{records} records, {arrangement}",
                "verdict": "chosen",
                "dev": dev_runs[(embedder, records)]["overall"][arrangement]["mrr@10"],
                "held_out": reused,
                "holdout2": fresh,
                "holdout3": holdout3_runs[(embedder, "chosen")]["overall"][arrangement]["mrr@10"],
                "difference": round(fresh - reused, 4),
                "held_out_source": f"eval/results/experiments/{HELD_OUT_SOURCES[embedder]}",
                "holdout2_source": f"eval/results/experiments/{HOLDOUT2_SOURCES[(embedder, 'chosen')]}",
                "holdout3_source": f"eval/results/experiments/{HOLDOUT3_SOURCES[(embedder, 'chosen')]}",
            }
        )

    return {
        "title": "The three held-out sets, side by side",
        "metric": "mrr@10",
        "note": f"{REUSED_SPLIT_NOTE} {FRESH_SPLIT_NOTE} {BOTH_SPLITS_NOTE}",
        # The difference is holdout2 against the reused split, which is the gap the wording above
        # is about; the heading names which two, because there are now three numbers beside it.
        "columns": ["dev", "held out (reused)", "holdout2 (fresh)", "holdout3 (never used)", "holdout2 - held out"],
        "rows": rows,
    }


def holdout3(runs: dict) -> dict:
    """The set that had never been scored, spent once, with the rule it was spent under applied.

    Every verdict here is **computed from the committed result files** against the rule as it was
    written down before any of them existed (docs/RESULTS.md, "Decision: the rule holdout3 will be
    read by"). Nothing in this block is a sentence somebody typed beside a number: if a file
    changed, the verdict would change with it rather than going quietly stale.
    """
    information = _read(EXPERIMENTS / HOLDOUT3_INFORMATION[3])
    rows = []
    for embedder in ("hash", "cloudflare"):
        guard = HOLDOUT3_GUARD[embedder]
        incumbent = runs[(embedder, "default")]["overall"][BASELINE[1]]
        records, arrangement = CHOSEN[embedder]
        candidate = runs[(embedder, "chosen")]["overall"][arrangement]
        rows.append(
            _holdout3_row(
                embedder,
                "today's default",
                f"{BASELINE[0]} records, {BASELINE[1]}",
                incumbent,
                eligible=True,
                verdict="the incumbent",
                source=HOLDOUT3_SOURCES[(embedder, "default")],
            )
        )
        rows.append(
            _holdout3_row(
                embedder,
                "candidate",
                f"{records} records, {arrangement}",
                candidate,
                eligible=True,
                verdict=_holdout3_verdict(candidate, incumbent, guard),
                source=HOLDOUT3_SOURCES[(embedder, "chosen")],
            )
        )
        if embedder == HOLDOUT3_INFORMATION[0]:
            rows.append(
                _holdout3_row(
                    embedder,
                    "information only",
                    f"{HOLDOUT3_INFORMATION[1]} records, {HOLDOUT3_INFORMATION[2]}",
                    information["overall"][HOLDOUT3_INFORMATION[2]],
                    eligible=False,
                    verdict="not eligible - the rule barred it from selection before it was scored",
                    source=HOLDOUT3_INFORMATION[3],
                )
            )

    return {
        "title": "holdout3: a question set that had never been scored, spent once",
        "label": "scored once, and not retuned on",
        "scored_once": True,
        "questions": rows[0]["questions"],
        "metrics": HOLDOUT3_METRICS,
        "rule": (
            "Written down before holdout3 was scored. The hash candidate replaces the default only if "
            "its holdout3 MRR@10 is higher and its hit@5 is not lower; cloudflare structured replaces "
            "cloudflare whole only if its MRR@10 is higher and its recall@20 is not lower."
        ),
        "guards": HOLDOUT3_GUARD,
        "note": (
            "One run per candidate, each made once, with no setting changed afterwards and nothing "
            "re-scored. The verdicts below are computed from the committed result files against that "
            "rule. Passing it makes a default change proposable, not applied: the server still runs "
            "whole records with BM25 off."
        ),
        "source": "eval/results/experiments/",
        "recorded": PROVENANCE["holdout3"],
        "rows": rows,
    }


def _holdout3_row(embedder: str, configuration: str, change: str, metrics: dict, **rest) -> dict:
    return {
        "embedder": embedder,
        "configuration": configuration,
        "change": change,
        "questions": metrics["questions"],
        **{name: metrics.get(name) for name in HOLDOUT3_METRICS},
        "eligible": rest["eligible"],
        "verdict": rest["verdict"],
        "source": f"eval/results/experiments/{rest['source']}",
    }


def _holdout3_verdict(candidate: dict, incumbent: dict, guard: str) -> str:
    """The rule, applied: a higher MRR@10 and a guard metric that has not fallen."""
    better = candidate["mrr@10"] > incumbent["mrr@10"]
    held = candidate[guard] >= incumbent[guard]
    if better and held:
        return f"passes: MRR@10 higher, {guard} not lower"
    if not better:
        return "fails: MRR@10 is not higher"
    return f"fails: {guard} is lower"


def chat_modes() -> dict | None:
    """The legacy-against-grounded comparison, labelled for exactly what it is."""
    for name in CHAT_MODE_SOURCES:
        path = EXPERIMENTS / name
        if path.exists():
            data = _read(path)
            break
    else:
        return None

    summary = data["summary"]
    runs = sum(block["runs"] for block in summary.values())
    if data["questions"] != CHAT_MODE_QUESTIONS or runs != CHAT_MODE_RUNS:
        # Not the full comparison. Rather than drawing a smaller run under the same heading and
        # trusting a reader to spot the question count, the block is left out entirely.
        return None
    return {
        "title": "Chat mode: legacy against grounded",
        "label": "one local run, one model, one question set - not a benchmark",
        "model": data["model"],
        "questions": data["questions"],
        "runs": runs,
        "scale_note": (
            f"{data['questions']} questions, {runs} runs - each mode answered every question once. "
            "A shorter trial run exists in the result folder and is deliberately not shown here."
        ),
        "weak_check": data["weak_check"],
        "stopped_early": data.get("stopped_early", ""),
        "source": f"eval/results/experiments/{path.name}",
        "recorded": PROVENANCE["chat_modes"],
        "note": (
            "Measured on this machine against one provider on one day. The timings are the "
            "client's own, over a local socket, and say nothing about a deployed server."
        ),
        "modes": [
            {
                "mode": mode,
                "runs": block["runs"],
                "passed": block["passed"],
                "expected_file_cited": block["expected_file_cited"],
                "fact_present": block["fact_present"],
                "no_answer": block["no_answer"],
                "first_token_seconds_median": block["first_token_seconds_median"],
                "total_seconds_median": block["total_seconds_median"],
                "model_calls_total": block["model_calls_total"],
                "tokens_total": block["tokens_total"],
                "cited_files_in_prompt": block["cited_files_in_prompt"],
                "cited_files_total": block["cited_files_total"],
            }
            for mode, block in summary.items()
        ],
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
            {
                "what": "structural",
                "questions": 49,
                "detail": "24 itsdangerous and 25 flask, 26 development / 23 held out",
            },
            {"what": "answer smoke check", "questions": 15, "detail": "run in each of the two chat modes"},
        ],
        "sample_size_note": (
            "On a split of 20 questions one question is 0.05 of a hit rate, so a difference of two or three "
            "questions is not a result. The subjects were chosen by hand, which is a bias a held-out split "
            "narrows but does not remove."
        ),
        "repos": [{"name": name, "url": repo["url"], "commit": repo["commit"]} for name, repo in REPOS.items()],
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
        "experiments": experiments(),
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
