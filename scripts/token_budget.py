"""Report what the provider actually says about quota, and what this machine has spent.

    python scripts/token_budget.py
    python scripts/token_budget.py --max-daily-tokens 150000

One request of a single output token is made and its rate-limit headers are read.

What the provider tells us, and what it does not:

* **Per minute, exactly:** `x-ratelimit-limit-tokens` and `x-ratelimit-remaining-tokens`.
* **Requests, exactly, but not daily:** `x-ratelimit-limit-requests` and its remaining
  count. Its own `x-ratelimit-reset-requests` header reads a couple of minutes, so the
  counter refills continuously.
* **Tokens per day: nothing.** There is no header for it. The figure appears only inside
  the body of a 429 once it has run out.

An earlier version of this script estimated the daily figure from the request counter. That
was wrong, and badly: the counter read 996 of 1000 while 199,400 of the account's 200,000
daily tokens had in fact been spent, so the script reported about 199,200 left minutes
before the provider refused a call. **It no longer guesses.** The daily line says the figure
is unknown unless the provider has stated it, and the useful number beside it is the running
total this machine has spent today, kept in `.codeatlas/token-spend.json`.

Nothing here reads or prints the key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from codeatlas.observability import token_ledger  # noqa: E402

# Default ceiling for a long unattended run, well under a 200,000 daily allowance so there
# is room left for live testing afterwards.
DEFAULT_MAX_DAILY_TOKENS = 150_000

_TPD = re.compile(r"tokens per day \(TPD\): Limit (\d+), Used (\d+)")

DAILY_UNKNOWN = "unknown unless the provider states it"


@dataclass
class Budget:
    model: str
    tokens_per_minute_limit: int
    tokens_per_minute_remaining: int
    tokens_reset: str
    requests_limit: int
    requests_remaining: int
    requests_reset: str
    # Only ever set from the provider's own words, inside a 429. Never inferred.
    daily_tokens_remaining: int | None = None
    spent_today: int = 0
    calls_today: int = 0
    ledger: str = ""
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"model                     : {self.model}",
            f"tokens per minute         : {self.tokens_per_minute_remaining:,} of "
            f"{self.tokens_per_minute_limit:,} left (resets in {self.tokens_reset or '?'})",
            f"requests                  : {self.requests_remaining:,} of {self.requests_limit:,} left "
            f"(resets in {self.requests_reset or '?'}; not a daily counter)",
        ]
        if self.daily_tokens_remaining is None:
            lines.append(f"daily tokens              : {DAILY_UNKNOWN}")
        else:
            lines.append(f"daily tokens              : {self.daily_tokens_remaining:,} left, as stated by the provider")
        lines.append(
            f"spent today, this machine : {self.spent_today:,} tokens over {self.calls_today} call(s)  [{self.ledger}]"
        )
        if self.note:
            lines.append(f"note                      : {self.note}")
        for warning in self.warnings:
            lines.append(f"WARNING                   : {warning}")
        return "\n".join(lines)


def parse_tpd(text: str) -> tuple[int, int] | None:
    """(limit, used) from a 429 body that names the daily token limit."""
    match = _TPD.search(text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def probe(model: str | None = None) -> Budget:
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    import os

    from openai import OpenAI

    from codeatlas.utils.config import load_config

    config = load_config()
    model = model or config.llm_model
    spend = token_ledger.spent_today(model)
    ledger = str(token_ledger.ledger_path())

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
    try:
        headers = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        ).headers
    except Exception as exc:
        reported = parse_tpd(str(exc))
        if reported is None:
            raise
        limit, used = reported
        return Budget(
            model=model,
            tokens_per_minute_limit=0,
            tokens_per_minute_remaining=0,
            tokens_reset="",
            requests_limit=0,
            requests_remaining=0,
            requests_reset="",
            daily_tokens_remaining=max(limit - used, 0),
            spent_today=spend.total,
            calls_today=spend.calls,
            ledger=ledger,
            note="the provider refused the probe because the daily token limit is reached",
        )

    def as_int(name: str) -> int:
        try:
            return int(float(headers.get(name, 0)))
        except (TypeError, ValueError):
            return 0

    budget = Budget(
        model=model,
        tokens_per_minute_limit=as_int("x-ratelimit-limit-tokens"),
        tokens_per_minute_remaining=as_int("x-ratelimit-remaining-tokens"),
        tokens_reset=headers.get("x-ratelimit-reset-tokens", ""),
        requests_limit=as_int("x-ratelimit-limit-requests"),
        requests_remaining=as_int("x-ratelimit-remaining-requests"),
        requests_reset=headers.get("x-ratelimit-reset-requests", ""),
        spent_today=spend.total,
        calls_today=spend.calls,
        ledger=ledger,
    )
    if spend.calls == 0:
        budget.warnings.append("nothing recorded today on this machine; the account may still have been used elsewhere")
    return budget


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--max-daily-tokens",
        type=int,
        default=DEFAULT_MAX_DAILY_TOKENS,
        help="exit non-zero once this machine has spent this many tokens today",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    budget = probe(args.model or None)
    if args.json:
        print(json.dumps({k: v for k, v in budget.__dict__.items()}, indent=2))
    else:
        print(budget.render())

    allowed, left = token_ledger.within_budget(args.max_daily_tokens, args.model or budget.model)
    if not allowed:
        print(
            f"\nSTOP: this machine has spent {budget.spent_today:,} tokens today, "
            f"at or over the {args.max_daily_tokens:,} limit."
        )
        return 1
    if budget.daily_tokens_remaining is not None and budget.daily_tokens_remaining <= 0:
        print("\nSTOP: the provider reports no daily tokens left.")
        return 1
    print(f"\n{left:,} of the {args.max_daily_tokens:,} local daily limit left.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
