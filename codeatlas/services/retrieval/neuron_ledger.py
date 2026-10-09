"""A running total of estimated neurons spent today, kept on this machine.

Cloudflare's free allowance is 10,000 neurons per day and resets at 00:00 UTC. Nothing in an
embeddings response says how much of it is left, or even what one call cost: the body carries
vectors and a shape, no usage at all. The only authoritative figure is the account dashboard,
which this project cannot read and should not try to.

So the count is kept here, the same way `observability/token_ledger.py` keeps the model's
token spend, and the provider stops calling when today's total passes the configured budget.
Three things follow from that, and all three matter:

* **The day is a UTC day**, not a local one, because that is when the allowance resets.
* **The figure is an estimate and is named one.** It is `ceil(chars / 3)` capped at 512 tokens
  per document, priced at Cloudflare's published rate - measured against the dashboard once,
  it over-counted by 1.39x (see docs/RESULTS.md). It is a ceiling on spend, not a measurement.
* **It counts what this machine spent.** Another machine, or the dashboard, will read higher.

Nothing here records what was embedded: a count of documents and a number of neurons, never a
path, a text or a question. Writing must never break indexing, so every read and write is
wrapped and a failure means the call goes uncounted.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Under .codeatlas/, which .gitignore already excludes in full.
DEFAULT_PATH = Path(".codeatlas") / "cloudflare-neurons.json"
ENV_PATH = "CODEATLAS_NEURON_LEDGER"

# Cloudflare's published price for @cf/baai/bge-small-en-v1.5, and the model's input window.
NEURONS_PER_MILLION_INPUT_TOKENS = 1841
MAX_TOKENS_PER_DOCUMENT = 512
# Deliberately pessimistic: 3 characters per token over-counts English and is about right for
# code, so the estimate errs towards stopping early.
CHARS_PER_TOKEN = 3

# Days kept in the file. Only today's total is ever read; the rest is there to be looked at.
HISTORY_DAYS = 30

_lock = threading.Lock()


@dataclass(frozen=True)
class Spend:
    day: str
    estimated_neurons: float
    documents: int
    calls: int


def ledger_path() -> Path:
    """Where the count lives. Relative to the working directory unless set outright."""
    return Path(os.getenv(ENV_PATH) or DEFAULT_PATH)


def today() -> str:
    """The UTC day, because that is the day Cloudflare's allowance resets on."""
    return datetime.now(UTC).date().isoformat()


def estimated_tokens(texts: list[str]) -> int:
    """ceil(chars / 3) per text, capped at the model's 512-token window."""
    return sum(min(math.ceil(len(text) / CHARS_PER_TOKEN), MAX_TOKENS_PER_DOCUMENT) for text in texts)


def estimated_neurons(texts: list[str]) -> float:
    return estimated_tokens(texts) * NEURONS_PER_MILLION_INPUT_TOKENS / 1_000_000


def record(neurons: float, documents: int, path: Path | None = None) -> None:
    """Add one call's estimated cost to today's total. Never raises."""
    if neurons <= 0:
        return
    target = path or ledger_path()
    try:
        with _lock:
            data = _read(target)
            days = data.setdefault("days", {})
            entry = days.setdefault(today(), {"estimated_neurons": 0.0, "documents": 0, "calls": 0})
            entry["estimated_neurons"] = round(float(entry.get("estimated_neurons", 0.0)) + neurons, 4)
            entry["documents"] = int(entry.get("documents", 0)) + max(documents, 0)
            entry["calls"] = int(entry.get("calls", 0)) + 1
            data["days"] = dict(sorted(days.items())[-HISTORY_DAYS:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        # An uncounted call is better than an index that fails over bookkeeping.
        logger.debug("Could not record estimated neuron spend: %s", exc)


def spent_today(path: Path | None = None) -> Spend:
    entry = (_read(path or ledger_path()).get("days") or {}).get(today()) or {}
    return Spend(
        day=today(),
        estimated_neurons=float(entry.get("estimated_neurons", 0.0)),
        documents=int(entry.get("documents", 0)),
        calls=int(entry.get("calls", 0)),
    )


def remaining(budget: float, path: Path | None = None) -> float:
    """Estimated neurons left in today's budget; 0.0 once it is spent."""
    return max(budget - spent_today(path).estimated_neurons, 0.0)


def affordable(estimate: float, budget: float, path: Path | None = None) -> bool:
    """Whether one more call of this estimated size still fits inside today's budget."""
    if budget <= 0:
        return False
    return spent_today(path).estimated_neurons + estimate <= budget


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
