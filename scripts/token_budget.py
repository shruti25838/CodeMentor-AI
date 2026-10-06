"""Report how much of the provider's quota is left, before anything spends it.

    python scripts/token_budget.py
    python scripts/token_budget.py --require-daily 60000

One tiny request (one output token) is made and its rate-limit headers are read.

What the provider actually tells us, and what it does not:

* Per minute, exactly: `x-ratelimit-limit-tokens` and `x-ratelimit-remaining-tokens`.
* Per day, for requests, exactly: `x-ratelimit-limit-requests` and the remaining count.
* Per day, for tokens: **nothing**. Groq does not send a tokens-per-day header. The figure
  appears only inside the body of a 429 once it has run out, as
  `tokens per day (TPD): Limit 200000, Used 199248`.

So the daily token figure here is an estimate, and it says which it is. The reliable guard
is the 429 itself: a run that watches for a TPD 429 stops on the provider's own word rather
than on a guess. The daily *request* counter is the useful signal before that — when it is
close to its limit the day has not rolled over, and when it is barely touched it has.

Nothing here reads or prints the key.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Groq's published free-tier allowance for the chat models used here. Only used to turn the
# daily request counter into an estimate; the provider never sends this number.
ASSUMED_DAILY_TOKEN_LIMIT = 200_000

_TPD = re.compile(r"tokens per day \(TPD\): Limit (\d+), Used (\d+)")


@dataclass
class Budget:
    model: str
    tokens_per_minute_limit: int
    tokens_per_minute_remaining: int
    requests_per_day_limit: int
    requests_per_day_remaining: int
    daily_tokens_remaining: int | None  # None when the provider has not said
    daily_is_estimate: bool
    note: str = ""

    @property
    def requests_used_today(self) -> int:
        return max(self.requests_per_day_limit - self.requests_per_day_remaining, 0)

    def render(self) -> str:
        lines = [
            f"model                     : {self.model}",
            f"tokens per minute         : {self.tokens_per_minute_remaining} of {self.tokens_per_minute_limit} left",
            f"requests today            : {self.requests_used_today} of "
            f"{self.requests_per_day_limit} used, {self.requests_per_day_remaining} left",
        ]
        if self.daily_tokens_remaining is None:
            lines.append("daily tokens remaining    : unknown (the provider sends no header for it)")
        else:
            how = "ESTIMATE" if self.daily_is_estimate else "reported by the provider"
            lines.append(f"daily tokens remaining    : ~{self.daily_tokens_remaining:,} ({how})")
        if self.note:
            lines.append(f"note                      : {self.note}")
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
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])

    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        headers = raw.headers
    except Exception as exc:  # the probe itself can be refused, which is informative
        reported = parse_tpd(str(exc))
        if reported:
            limit, used = reported
            return Budget(
                model=model,
                tokens_per_minute_limit=0,
                tokens_per_minute_remaining=0,
                requests_per_day_limit=0,
                requests_per_day_remaining=0,
                daily_tokens_remaining=max(limit - used, 0),
                daily_is_estimate=False,
                note="the provider refused the probe because the daily token limit is reached",
            )
        raise

    def as_int(name: str) -> int:
        try:
            return int(float(headers.get(name, 0)))
        except (TypeError, ValueError):
            return 0

    requests_limit = as_int("x-ratelimit-limit-requests")
    requests_remaining = as_int("x-ratelimit-remaining-requests")

    # No daily token header exists, so infer from the daily request counter. If almost no
    # requests have been made today the daily token budget is close to full; if many have,
    # this estimate is weak and says so.
    daily_remaining: int | None = None
    estimate = True
    note = ""
    if requests_limit:
        share_used = (requests_limit - requests_remaining) / requests_limit
        daily_remaining = int(ASSUMED_DAILY_TOKEN_LIMIT * (1 - share_used))
        note = (
            f"no tokens-per-day header exists; estimated from the daily request counter "
            f"against an assumed {ASSUMED_DAILY_TOKEN_LIMIT:,} limit"
        )
    return Budget(
        model=model,
        tokens_per_minute_limit=as_int("x-ratelimit-limit-tokens"),
        tokens_per_minute_remaining=as_int("x-ratelimit-remaining-tokens"),
        requests_per_day_limit=requests_limit,
        requests_per_day_remaining=requests_remaining,
        daily_tokens_remaining=daily_remaining,
        daily_is_estimate=estimate,
        note=note,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--require-daily",
        type=int,
        default=0,
        help="exit non-zero if the estimated daily tokens left is below this",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    budget = probe(args.model or None)
    if args.json:
        print(json.dumps(budget.__dict__, indent=2))
    else:
        print(budget.render())

    short = (
        args.require_daily
        and budget.daily_tokens_remaining is not None
        and budget.daily_tokens_remaining < args.require_daily
    )
    if short:
        print(
            f"\nSTOP: about {budget.daily_tokens_remaining:,} daily tokens left, "
            f"below the {args.require_daily:,} needed."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
