"""Ask one session several questions against a running server, and print what each one cost.

    python scripts/chat_latency_probe.py --base-url http://127.0.0.1:8000 -n 8

Chat latency rises as a session asks more questions. Four things could be doing it: the
history growing the prompt, the follow-up rewrite spending an extra model call, the provider's
per-minute limit making the request wait, and the process itself getting slower. This asks a
fixed list of questions in **one** session and prints one row per question carrying the fields
that tell those apart -- `history_turns`, `rewrite_ran`, input tokens per model call, the retry
backoff the provider imposed, and how long each model call took.

**Against a real server over HTTP**, not an in-process client: the thing being measured is a
chat as a visitor has it, and an in-process client would hide the event loop, the executor and
the connection. Start one first, in the mode to be measured:

    $env:CODEATLAS_CHAT_MODE = "grounded"; python -m uvicorn codeatlas.app.main:app --port 8000

The mode is **not** taken from the command line. It is read back out of each answer's
`agents_used`, so a row cannot be labelled with a mode the server was not running.

**This spends real money and is the only script here that does.** Two guards, both of which
stop the run cleanly rather than letting it overrun:

* a **token cap** for the whole session, checked against the on-disk ledger before every
  question. The cap is applied inside the guard, against this session's own spend -- the
  ledger total when the run started is subtracted, so an earlier run's spend today neither
  counts against this cap nor hides it. The guard also **reserves what the next question is
  expected to cost**, so the cap holds after the last question as well as before it.
* a **pacer** that keeps the tokens spent in any trailing 60 seconds under the provider's
  per-minute allowance, estimating the next question from the most expensive one so far.

A refusal -- a 429, a daily quota, a payload rejection -- ends the run with a line saying so.
No question text, answer text or key is printed: only the numbers the server reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.observability import token_ledger  # noqa: E402
from codeatlas.services.llm.quota import MESSAGES  # noqa: E402

DEFAULT_REPO = "https://github.com/pallets/itsdangerous"

# Eight questions in one conversation, fixed so two runs are comparable. Four stand on their
# own and four lean on the turn before them ("it", "that", "them", an "and" opener), which is
# what makes the follow-up rewrite run on some questions and not others -- the difference the
# rows have to show for `rewrite_ran` to be worth reading.
QUESTIONS = [
    "How does Signer create a signature?",
    "How is a signature checked when unsigning?",
    "Why does it derive a key first?",
    "What does the timestamp signer add on top of that?",
    "Which exceptions can unsigning raise?",
    "What is the difference between them?",
    "How does the URL safe serializer encode its payload?",
    "And how does it decode it again?",
]

# How a refusal reads once the app has turned it into a sentence for the visitor. The app
# catches the provider's 429 and answers with one of these, so a run that only watched for an
# error event would sail past a rate limit and record the refusals as answers.
REFUSALS = tuple(MESSAGES.values())
ERROR_PREFIX = "Error generating answer"


@dataclass
class Row:
    """One question, as the server reported it. Every field is a number or a short label."""

    index: int
    mode: str
    history_turns: int | None
    rewrite_ran: int | None
    llm_calls: int
    in_tokens: float
    out_tokens: float
    llm_ms: float
    llm_each: list[float]
    retry_s: float
    first_token_ms: float | None
    total_ms: float
    wall_ms: float
    spent: int
    refused: str = ""


@dataclass
class Pacer:
    """Keeps the tokens spent in any trailing 60 seconds under the provider's allowance.

    The provider's limit is a sliding window, so this keeps one: each question's spend with the
    time it finished, everything older than the window dropped. Before the next question it
    waits until the window has room for a question as expensive as the worst one so far, which
    is the only estimate available that cannot be optimistic about what has already happened.

    The clock and the sleep are injected so a test can drive the waiting without waiting, and
    so neither the answers nor the decisions depend on how finely the host's clock resolves.
    """

    limit: int
    window_seconds: float = 60.0
    first_estimate: int = 5_000
    spends: list[tuple[float, int]] = field(default_factory=list)
    poll_seconds: float = 1.0
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], object] = time.sleep

    def record(self, tokens: int) -> None:
        self.spends.append((self.clock(), tokens))

    def estimate(self) -> int:
        return max((tokens for _, tokens in self.spends), default=self.first_estimate)

    def in_window(self) -> int:
        cutoff = self.clock() - self.window_seconds
        self.spends = [(at, tokens) for at, tokens in self.spends if at >= cutoff]
        return sum(tokens for _, tokens in self.spends)

    def _over_the_limit(self) -> bool:
        """Whether a question as expensive as the worst so far would not fit right now.

        The window has to hold something for this to be true: with nothing recorded there is
        nothing that can age out, so a standing estimate larger than the limit would otherwise
        spin for ever rather than let the one question that cannot fit through.
        """
        return bool(self.spends) and self.in_window() + self.estimate() > self.limit

    def wait(self) -> float:
        """Block until there is room, and return how long that took.

        **Exactly 0.0 when the next question already fits**, by returning the constant rather
        than a measurement. Returning `monotonic() - started` unconditionally made the no-wait
        answer the cost of reading the clock twice: 0.0 on Windows, where the counter is too
        coarse to resolve two adjacent readings, and a few microseconds on CI, where it is not.
        `wait() == 0.0` is the honest assertion -- five microseconds is not a wait -- so the
        difference is computed only once something has actually been slept through. Deciding
        whether there is room still reads the clock, to date the window against; what it never
        does on this path is subtract one reading from another.
        """
        if self.limit <= 0 or not self._over_the_limit():
            return 0.0
        started = self.clock()
        while self._over_the_limit():
            self.sleeper(self.poll_seconds)
        return self.clock() - started


def budget_guard(cap: int, baseline: int, ledger: Path | None, reserve: int = 0) -> tuple[bool, int, int]:
    """(may this session spend more, what it has spent, what is left of the cap).

    The cap lives in here rather than at the call site so that there is exactly one place that
    decides the run may continue. `baseline` is the ledger's total when the run started, so the
    spend measured is this session's and not today's.

    `reserve` is what the next question is expected to cost, and it is the difference between a
    cap and a tripwire. Asking only whether the spend so far is under the cap lets the last
    question push past it, because its cost is not known until it has been paid: the first run
    of this script overran 40,000 by 3,958 that way. With the next question's cost reserved,
    the run stops while there is still room for it, so the cap holds on the way out too.
    """
    spent = max(token_ledger.spent_today(path=ledger).total - baseline, 0)
    return spent < cap and spent + reserve <= cap, spent, max(cap - spent, 0)


@dataclass
class Asked:
    """What one question produced. `opening` exists only to recognise a refusal by."""

    event: dict
    wall_ms: float
    first_token_ms: float | None
    # The first characters of the answer, kept because the app turns a 429 into a sentence for
    # the visitor rather than into an error event, so a run watching only for an error event
    # would record refusals as answers. Matched against a fixed list, and never printed.
    opening: str


def ask(client, base_url: str, repo_id: str, session_id: str, question: str) -> Asked:
    """One question over the real stream, read as it arrives so the first token can be timed."""
    payload = {"question": question, "repo_id": repo_id, "session_id": session_id}
    started = time.perf_counter()
    first_token: float | None = None
    done: dict = {}
    opening = ""
    with client.stream("POST", f"{base_url}/ask/stream", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind == "token":
                if first_token is None:
                    first_token = (time.perf_counter() - started) * 1000
                if len(opening) < 300:
                    opening += str(event.get("content", ""))
            elif kind in ("done", "error"):
                done = event
    return Asked(done, (time.perf_counter() - started) * 1000, first_token, opening[:300])


def refusal_of(event: dict, answer: str) -> str:
    """The refusal this question hit, as a short label, or "" when it was answered."""
    if event.get("type") == "error":
        return "stream error"
    for kind, message in MESSAGES.items():
        if answer.startswith(message):
            return kind
    if answer.startswith(ERROR_PREFIX):
        return "model error"
    return ""


def mode_of(event: dict) -> str:
    """Which chat path answered, read off the agents the server says ran."""
    agents = event.get("agents_used") or []
    if agents == ["grounded"]:
        return "grounded"
    if "mentor" in agents:
        return "legacy"
    return ",".join(agents) or "?"


HEADER = "  q turns rw calls  in_tok out_tok  llm_ms first_ms total_ms retry_s per-call llm_ms"


def format_row(row: Row) -> str:
    each = "/".join(f"{ms:.0f}" for ms in row.llm_each) if len(row.llm_each) > 1 else ""
    return (
        f"{row.index:3d}"
        f"{_or_dash(row.history_turns):>6}"
        f"{_or_dash(row.rewrite_ran):>3}"
        f"{row.llm_calls:>6}"
        f"{row.in_tokens:>8.0f}"
        f"{row.out_tokens:>8.0f}"
        f"{row.llm_ms:>8.0f}"
        f"{_or_dash(None if row.first_token_ms is None else round(row.first_token_ms)):>9}"
        f"{row.total_ms:>9.0f}"
        f"{row.retry_s:>8.1f}"
        f"  {each}"
    ).rstrip()


def _or_dash(value) -> str:
    return "-" if value is None else str(value)


def analyze(client, base_url: str, repo_url: str) -> str:
    response = client.post(f"{base_url}/analyze-repo", json={"repo_url": repo_url}, timeout=300.0)
    response.raise_for_status()
    return response.json()["repository_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="a server that is already running")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--repo-id", help="skip the analyze step and ask about this repository")
    parser.add_argument("-n", "--questions", type=int, default=len(QUESTIONS))
    parser.add_argument("--token-cap", type=int, default=40_000, help="hard cap on this session's provider tokens")
    parser.add_argument("--tpm-limit", type=int, default=8_000, help="the provider's per-minute token allowance")
    parser.add_argument("--min-gap", type=float, default=0.0, help="seconds to wait between questions regardless")
    parser.add_argument("--ledger", help="the token ledger to count against (default: the app's own)")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    import httpx

    ledger = Path(args.ledger) if args.ledger else None
    baseline = token_ledger.spent_today(path=ledger).total
    pacer = Pacer(limit=args.tpm_limit)
    session_id = uuid.uuid4().hex
    questions = QUESTIONS[: args.questions]
    rows: list[Row] = []
    stopped = ""

    with httpx.Client(timeout=args.timeout) as client:
        repo_id = args.repo_id or analyze(client, args.base_url, args.repo_url)
        print(f"repo: {repo_id}  session: {session_id[:8]}  cap: {args.token_cap} tokens  tpm: {args.tpm_limit}")
        print(HEADER)

        for index, question in enumerate(questions, 1):
            # The worst question so far is the only estimate that cannot be optimistic about
            # what has already happened. Before the first one there is nothing to go on, so the
            # pacer's standing figure stands in and the cap has to be at least that large.
            reserve = pacer.estimate()
            may_continue, spent, left = budget_guard(args.token_cap, baseline, ledger, reserve)
            if not may_continue:
                stopped = (
                    f"token cap reached before question {index}: {spent} of {args.token_cap} spent, "
                    f"and the next question is expected to cost about {reserve}"
                )
                break
            if index > 1:
                waited = pacer.wait() + _sleep(args.min_gap)
                if waited:
                    print(f"    (paced {waited:.0f}s; {pacer.in_window()} tokens in the last 60s, {left} of cap left)")

            before = token_ledger.spent_today(path=ledger).total
            asked = ask(client, args.base_url, repo_id, session_id, question)
            after = token_ledger.spent_today(path=ledger).total
            pacer.record(after - before)

            event = asked.event
            timings = event.get("timings_ms") or {}
            row = Row(
                index=index,
                mode=mode_of(event),
                history_turns=timings.get("history_turns"),
                rewrite_ran=timings.get("rewrite_ran"),
                llm_calls=int(timings.get("llm_calls", 1 if "llm" in timings else 0)),
                in_tokens=timings.get("in_tokens", 0),
                out_tokens=timings.get("out_tokens", 0),
                llm_ms=timings.get("llm", 0.0),
                llm_each=timings.get("llm_each") or ([timings["llm"]] if "llm" in timings else []),
                retry_s=timings.get("retry_wait", 0.0) / 1000,
                first_token_ms=asked.first_token_ms,
                total_ms=timings.get("total", asked.wall_ms),
                wall_ms=asked.wall_ms,
                spent=after - before,
                refused=refusal_of(event, asked.opening),
            )
            rows.append(row)
            print(format_row(row))
            if row.refused:
                stopped = f"the provider refused question {index} ({row.refused}); stopping"
                break

    _summarise(rows, args, baseline, ledger, stopped)


def _sleep(seconds: float) -> float:
    if seconds > 0:
        time.sleep(seconds)
    return seconds


def _models(ledger: Path | None) -> str:
    """The model names today's ledger holds. A model name is not a credential."""
    path = ledger or token_ledger.ledger_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    return ", ".join(sorted(data.get("models") or {})) or "none"


def _summarise(rows: list[Row], args, baseline: int, ledger: Path | None, stopped: str) -> None:
    print()
    if not rows:
        print("no questions were asked")
        return
    modes = sorted({row.mode for row in rows})
    print(f"mode (from agents_used): {', '.join(modes)}")
    first, last = rows[0], rows[-1]
    print(f"total_ms first -> last: {first.total_ms:.0f} -> {last.total_ms:.0f}")
    print(f"in_tokens first -> last: {first.in_tokens:.0f} -> {last.in_tokens:.0f}")
    print(f"retry waits: {sum(row.retry_s for row in rows):.1f}s over {len(rows)} question(s)")
    spend = token_ledger.spent_today(path=ledger)
    _, spent_here, left = budget_guard(args.token_cap, baseline, ledger)
    print(f"session spend: {spent_here} tokens of the {args.token_cap} cap ({left} left)")
    print(f"ledger today: {spend.total} tokens over {spend.calls} call(s), model(s): {_models(ledger)}")
    if stopped:
        print(f"STOPPED: {stopped}")


if __name__ == "__main__":
    main()
