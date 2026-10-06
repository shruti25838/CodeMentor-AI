"""A running total of tokens spent today, per model, kept on this machine.

The provider does not say how much of a daily token allowance is left. Its headers cover
only the per-minute window, and the daily figure appears solely inside a 429 once the budget
is already gone — by which time a long run has nothing left to stop for. Inferring it from
the request-count header was worse than useless: that counter refills every few minutes, so
it read 996 of 1000 while 199,400 of 200,000 tokens had in fact been spent.

So the count is kept here instead. Every model call adds its usage to a small JSON file
under `.codeatlas/`, which is gitignored, and a long run checks the total before each
question and stops when it passes a limit. The count is of what this machine has spent, so
it is a lower bound on the account's real usage — another machine, or the provider's web
console, would not be in it.

Writing must never break a request: every read and write is wrapped, and a failure means
the call simply goes uncounted.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path(".codeatlas") / "token-spend.json"
ENV_PATH = "CODEATLAS_TOKEN_LEDGER"

_lock = threading.Lock()


@dataclass(frozen=True)
class Spend:
    day: str
    model: str
    input_tokens: int
    output_tokens: int
    calls: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


def ledger_path() -> Path:
    """Where the count lives. Relative to the working directory unless set outright."""
    return Path(os.getenv(ENV_PATH) or DEFAULT_PATH)


def _today() -> str:
    return date.today().isoformat()


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def record(model: str, input_tokens: int, output_tokens: int, path: Path | None = None) -> None:
    """Add one call's usage to today's total. Never raises."""
    if input_tokens <= 0 and output_tokens <= 0:
        return
    target = path or ledger_path()
    try:
        with _lock:
            data = _read(target)
            # A new day starts the count again rather than accumulating for ever.
            if data.get("day") != _today():
                data = {"day": _today(), "models": {}}
            models = data.setdefault("models", {})
            entry = models.setdefault(model or "unknown", {"input": 0, "output": 0, "calls": 0})
            entry["input"] += max(input_tokens, 0)
            entry["output"] += max(output_tokens, 0)
            entry["calls"] += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        # A call that cannot be counted is better than a request that fails over counting.
        logger.debug("Could not record token spend: %s", exc)


def spent_today(model: str | None = None, path: Path | None = None) -> Spend:
    """Today's total for one model, or across all of them when model is None."""
    data = _read(path or ledger_path())
    day = data.get("day") or _today()
    models: dict = data.get("models") or {}
    if day != _today():
        # Yesterday's file; today nothing has been spent yet.
        return Spend(day=_today(), model=model or "all", input_tokens=0, output_tokens=0, calls=0)

    wanted = models if model is None else {model: models.get(model, {})}
    return Spend(
        day=day,
        model=model or "all",
        input_tokens=sum(int(e.get("input", 0)) for e in wanted.values()),
        output_tokens=sum(int(e.get("output", 0)) for e in wanted.values()),
        calls=sum(int(e.get("calls", 0)) for e in wanted.values()),
    )


def within_budget(limit: int, model: str | None = None, path: Path | None = None) -> tuple[bool, int]:
    """(whether spending may continue, tokens left) against a locally-kept daily limit."""
    if limit <= 0:
        return True, 0
    used = spent_today(model, path).total
    return used < limit, max(limit - used, 0)
