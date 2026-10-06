"""Prometheus metrics, served at GET /metrics (needs the API key).

Labels hold route templates, stage names and status codes only: never raw URLs, repository
names, questions or answers, so label values stay few and nothing a visitor typed is exposed.
"""

import json
import logging
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# From a 1 ms search to the 120 s indexing limit; clone and model calls sit in the upper half.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120)

REQUEST_COUNT = Counter(
    "codeatlas_request_total",
    "HTTP requests by route template, method and status code",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "codeatlas_request_latency_seconds",
    "Time until the response started (for a stream: until it opened, not until it closed)",
    ["path"],
    buckets=LATENCY_BUCKETS,
)

ERROR_COUNT = Counter(
    "codeatlas_errors_total",
    "Failed requests: 5xx responses, unhandled exceptions, and errors sent inside a 200 event stream",
    ["path", "kind"],
)

STAGE_LATENCY = Histogram(
    "codeatlas_stage_duration_seconds",
    "Time per request spent in each stage (a stage that runs several times in a request is summed)",
    ["endpoint", "stage"],
    buckets=LATENCY_BUCKETS,
)

RETRIEVAL_HIT_RATE = Gauge(
    "codeatlas_retrieval_hit_rate",
    "Share of eval questions with a relevant file in the top k results (from eval/results, not live traffic)",
    ["split", "k"],
)

RETRIEVAL_MRR = Gauge(
    "codeatlas_retrieval_mrr",
    "Mean reciprocal rank at 10 on the eval questions (from eval/results, not live traffic)",
    ["split"],
)

RETRIEVAL_EVAL_INFO = Gauge(
    "codeatlas_retrieval_eval_info",
    "Which eval run the retrieval gauges come from (value is always 1)",
    ["split", "commit", "embedder"],
)

# Timer stage names -> metric stage names. Both model calls of a chat question (follow-up rewrite and
# answer) count as "model"; anything not listed keeps its own name.
_STAGE_NAMES = {"llm": "model", "rewrite": "model"}


def observe_stages(endpoint: str, stages_ms: dict[str, float], total_ms: float) -> None:
    """Record one request's stage times (ms, as StageTimer keeps them) and its total."""
    seconds: dict[str, float] = {}
    for name, ms in stages_ms.items():
        stage = _STAGE_NAMES.get(name, name)
        seconds[stage] = seconds.get(stage, 0.0) + ms / 1000
    seconds["total"] = total_ms / 1000
    for stage, value in seconds.items():
        STAGE_LATENCY.labels(endpoint=endpoint, stage=stage).observe(value)


def load_eval_results(results_dir: str) -> int:
    """Set the retrieval gauges from the JSON files scripts/eval_retrieval.py --json wrote; returns files read."""
    loaded = 0
    for path in sorted(Path(results_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            split = data["split"]
            overall = data["overall"]
            for k, value in overall["hit_rate"].items():
                RETRIEVAL_HIT_RATE.labels(split=split, k=k.lstrip("@")).set(value)
            RETRIEVAL_MRR.labels(split=split).set(overall["mrr@10"])
            RETRIEVAL_EVAL_INFO.labels(
                split=split, commit=data.get("codementor_commit", ""), embedder=data.get("embedder", "").split(" ")[0]
            ).set(1)
            loaded += 1
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Skipped eval results file %s: %s", path.name, exc)
    return loaded
