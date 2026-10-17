"""CODEATLAS_REASONING_EFFORT unset against low, in both chat modes, with a real model.

    python scripts/reasoning_effort_compare.py --limit 4 --out eval/results/experiments/reasoning-effort.json

Four arms over the same questions and the same warm index:

    legacy  / effort unset     the server's current default, nothing sent
    legacy  / effort low       reasoning_effort=low in the request body
    grounded/ effort unset
    grounded/ effort low

**The arms are interleaved by question, not run one after another**, so a run cut short by
the token cap leaves every question either fully compared across the four arms or not run at
all. A run that stopped half way through arm one would otherwise compare nothing.

What is measured, per arm: time to the first streamed token (median and range), total time,
model calls, tokens, the deterministic pass check `scripts/answer_smoke.py` uses, and how many
runs got no model answer. **The pass check is weak** - two string tests per answer, is the
expected file cited and is one hand-written word present, no judge model and nothing that
reads whether the explanation is correct. Runs of it vary a lot: the same legacy path scored
8/15 on one day and 12/15 on another.

Spending is bounded three ways. The pacer keeps under the provider's per-minute token
allowance; `--max-session-tokens` stops the run cleanly once *this run* has spent that many,
counted as a delta on the on-disk ledger rather than against the day's total, so the cap means
what it says; and the provider's own statement of a daily limit, which only ever appears in
the body of a 429, stops it at once. Nothing here prints a key, or anything from the
environment.

The answers themselves are written under `.codeatlas/`, which is gitignored, one file per arm
plus a side-by-side view - they are prose from a model and have no business in the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from codeatlas.observability import token_ledger  # noqa: E402
from codeatlas.services.qa.grounded_chat import GROUNDED, LEGACY  # noqa: E402
from scripts.answer_smoke import (  # noqa: E402
    QUESTIONS,
    REPOS,
    Pacer,
    free_megabytes,
    judge,
    seen_daily_remaining,
    token_total,
    watch_for_daily_limit,
)
from scripts.chat_mode_compare import ask, model_calls  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

UNSET = "unset"
LOW = "low"

# The four arms, in the order each question is run under them.
ARMS = [(LEGACY, UNSET), (LEGACY, LOW), (GROUNDED, UNSET), (GROUNDED, LOW)]

# What one question is expected to cost in each mode, used only for pacing and for deciding
# whether the remaining cap is enough for another run. Measured at 2a66017: legacy spent
# 52,084 tokens over 15 questions and grounded 24,495, so roughly 3,500 and 1,650. Rounded up,
# because a guess that is too low is the one that overshoots the cap.
ESTIMATE = {LEGACY: 4000, GROUNDED: 2000}

ANSWERS_DIR = PROJECT / ".codeatlas" / "reasoning-effort-answers"


def arm_name(mode: str, effort: str) -> str:
    return f"{mode}-{effort}"


def swap_model(orchestrator, answers, provider, model) -> None:
    """Point every path that calls the model at this one, without dropping any cached state.

    The alternative - clearing the DI caches between arms - would rebuild `AnswerService` and
    with it the in-process BM25 index, so the first question of each arm would pay to build a
    lexical index the others did not. That is a timing difference between arms caused purely by
    how the measurement was set up, and this is a measurement of timing.

    Each agent reads `self._llm` when it builds its chain, per call, so assigning it is enough.
    The grounded path and the follow-up rewriter both ask the provider for a model per request,
    so for those it is `get_chat_model` that has to change.
    """
    answers._llm = model
    orchestrator._planner._llm = model
    orchestrator._analyst_agent._llm = model
    orchestrator._mentor_agent._llm = model
    provider.get_chat_model = lambda: model


def complete_questions(rows: list[dict]) -> list[str]:
    """The question ids every arm managed to run, in the order they were first seen.

    A run stopped by the cap ends mid-question, so the last question has an answer under one
    or two arms and not the rest. Summarising over all the rows would then compare one arm's
    six questions against another's five, and the arm that happened to go first would carry an
    extra question nobody else answered. Only complete sets are summarised; the partial ones
    stay in `rows` so the file records what was actually spent.
    """
    seen: list[str] = []
    counts: dict[str, int] = {}
    for row in rows:
        if row["question_id"] not in seen:
            seen.append(row["question_id"])
        counts[row["question_id"]] = counts.get(row["question_id"], 0) + 1
    return [q for q in seen if counts[q] == len(ARMS)]


def summarise(rows: list[dict], mode: str, effort: str) -> dict:
    complete = set(complete_questions(rows))
    mine = [r for r in rows if r["chat_mode"] == mode and r["effort"] == effort and r["question_id"] in complete]
    answered = [r for r in mine if not r["no_answer"]]
    first = sorted(r["first_token_seconds"] for r in answered if r["first_token_seconds"] is not None)
    totals = [r["total_seconds"] for r in mine]
    return {
        "runs": len(mine),
        "answered": len(answered),
        "passed": sum(1 for r in mine if r["passed"]),
        "expected_file_cited": sum(1 for r in mine if r["file_ok"]),
        "fact_present": sum(1 for r in mine if r["fact_ok"]),
        "no_answer": sum(1 for r in mine if r["no_answer"]),
        "first_token_seconds_median": round(_median(first), 2) if first else None,
        "first_token_seconds_min": round(first[0], 2) if first else None,
        "first_token_seconds_max": round(first[-1], 2) if first else None,
        "total_seconds_median": round(_median(sorted(totals)), 2) if totals else None,
        "total_seconds_min": round(min(totals), 2) if totals else None,
        "total_seconds_max": round(max(totals), 2) if totals else None,
        "model_calls_total": sum(r["model_calls"] for r in mine),
        "tokens_total": sum(r["tokens"] for r in mine),
        "questions": sorted({r["question_id"] for r in mine}),
    }


def _median(ordered: list[float]) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def write_answers(answers_by_arm: dict[str, list[dict]]) -> list[Path]:
    """One file per arm, plus one that puts the four side by side for a single question.

    Under `.codeatlas/`, which is gitignored: these are a model's prose, they differ on every
    run, and committing them would be committing noise.
    """
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, rows in answers_by_arm.items():
        path = ANSWERS_DIR / f"{arm}.md"
        parts = [f"# {arm}\n"]
        for row in rows:
            parts.append(f"## {row['question_id']} — {row['repo']}\n")
            parts.append(f"**Question:** {row['question']}\n")
            parts.append(f"*first token {row['first_token_seconds']}s, total {row['total_seconds']}s, ")
            parts.append(f"{row['model_calls']} model call(s), {row['tokens']} tokens, ")
            parts.append(f"pass {row['passed']}*\n\n")
            parts.append((row["answer"] or "(no answer)") + "\n\n---\n\n")
        path.write_text("".join(parts), encoding="utf-8")
        written.append(path)

    side = ANSWERS_DIR / "side-by-side.md"
    ids: list[str] = []
    for rows in answers_by_arm.values():
        for row in rows:
            if row["question_id"] not in ids:
                ids.append(row["question_id"])
    parts = ["# The same question under each arm\n\n"]
    for question_id in ids:
        shown = None
        blocks = []
        for arm, rows in answers_by_arm.items():
            row = next((r for r in rows if r["question_id"] == question_id), None)
            if row is None:
                continue
            shown = shown or row
            blocks.append(
                f"### {arm}\n*first token {row['first_token_seconds']}s, {row['tokens']} tokens, "
                f"pass {row['passed']}*\n\n{row['answer'] or '(no answer)'}\n\n"
            )
        if shown is None:
            continue
        parts.append(f"## {question_id} — {shown['question']}\n\n")
        parts.extend(blocks)
        parts.append("---\n\n")
    side.write_text("".join(parts), encoding="utf-8")
    written.append(side)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=4, help="first N questions per repository")
    parser.add_argument(
        "--max-session-tokens",
        type=int,
        default=60_000,
        help="stop cleanly once THIS RUN has spent this many tokens, read from the ledger",
    )
    parser.add_argument("--min-free-mb", type=int, default=600)
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    numbered = {id(item): f"{item['repo']}-{n + 1}" for n, item in enumerate(items)}
    if args.limit:
        kept: dict[str, int] = {}
        chosen = []
        for item in items:
            if kept.get(item["repo"], 0) < args.limit:
                chosen.append(item)
                kept[item["repo"]] = kept.get(item["repo"], 0) + 1
        items = chosen

    import os
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="codeatlas-effort-"))
    os.environ.update(
        CODEATLAS_INDEX_DIR=str(work / "indexes"),
        CODEATLAS_STATE_DIR=str(work / "state"),
        # The run works in a temporary directory and the ledger path is relative by default,
        # so without this the day's spend would be written into the temp folder and thrown
        # away - and the cap would have nothing to read.
        CODEATLAS_TOKEN_LEDGER=str(PROJECT / ".codeatlas" / "token-spend.json"),
        CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_CLONE_GLOBAL_PER_MINUTE="1000",
        CODEATLAS_LLM_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_LLM_GLOBAL_PER_MINUTE="1000",
    )
    # Whatever is in the environment or .env must not decide what the "unset" arm sends.
    os.environ.pop("CODEATLAS_REASONING_EFFORT", None)
    os.chdir(work)
    started = time.perf_counter()

    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    # By name only. The value is never read into a variable, printed or logged.
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise SystemExit("no model key found in .env; this comparison needs a real model")

    import httpx

    from codeatlas.app import di
    from codeatlas.app.main import create_app
    from codeatlas.services.llm.provider import LlmProvider, reasoning_effort_kwargs
    from scripts.mode_compare import serve

    watch_for_daily_limit()
    base_config = di.get_config()

    # One client per effort, each built by the shipped code path - so what this measures is
    # exactly what a server with the variable set would send, not a parameter bolted on here.
    efforts = {UNSET: None, LOW: LOW}
    models = {
        name: LlmProvider(replace(base_config, llm_reasoning_effort=value)).get_chat_model()
        for name, value in efforts.items()
    }
    for name, value in efforts.items():
        sent = reasoning_effort_kwargs(replace(base_config, llm_reasoning_effort=value))
        print(f"arm effort {name:5s}: request body carries {sent or 'nothing extra'}")

    apps = {
        LEGACY: create_app(replace(base_config, chat_mode=LEGACY)),
        GROUNDED: create_app(replace(base_config, chat_mode=GROUNDED)),
    }
    clients = {}
    for mode, app in apps.items():
        base_url, _ = serve(app)
        clients[mode] = httpx.Client(base_url=base_url, timeout=600)

    print(f"model: {base_config.llm_provider} / {base_config.llm_model}")
    print(f"output token budget: {base_config.llm_max_tokens} (unchanged by this run)")
    print(f"session token cap: {args.max_session_tokens:,}")

    # Indexed once; the DI singletons are shared, so both apps see the same index.
    repo_ids: dict[str, str] = {}
    for name in sorted({i["repo"] for i in items}):
        body = clients[LEGACY].post("/analyze-repo", json={"repo_url": REPOS[name]}, timeout=600).json()
        repo_ids[name] = body["repository_id"]
        print(f"indexed {name}: {body['file_count']} files")

    orchestrator = di.get_agent_orchestrator()
    answer_service = di.get_answer_service()
    provider = di.get_llm_provider()

    # Read before the warm-ups, not after. The cap is on the whole session, and a warm-up is a
    # real model call that spends real tokens: reading the ledger after them would hide four
    # calls from the only thing that stops the run.
    ledger_at_start = token_ledger.spent_today().total

    # One warm-up per arm, not recorded. The first call on a client opens a connection and the
    # first question of a mode builds that mode's lexical index, and neither belongs in a
    # comparison of warm requests.
    warm = items[0]
    for mode, effort in ARMS:
        swap_model(orchestrator, answer_service, provider, models[effort])
        ask(clients[mode], repo_ids[warm["repo"]], warm["question"])
    warm_up_tokens = token_ledger.spent_today().total - ledger_at_start
    print(f"warmed all four arms: {warm_up_tokens:,} tokens, charged to the cap\n")

    pacer = Pacer()
    rows: list[dict] = []
    answers_by_arm: dict[str, list[dict]] = {arm_name(m, e): [] for m, e in ARMS}
    stopped_early = ""
    plan = [(mode, effort, item) for item in items for mode, effort in ARMS]

    for mode, effort, item in plan:
        free = free_megabytes()
        if 0 <= free < args.min_free_mb:
            stopped_early = f"only {free} MB of memory free, need {args.min_free_mb} MB"
            print(f"\nSTOPPING CLEANLY: {stopped_early}")
            break
        # A delta on the ledger, not the day's total: the cap is on this session.
        spent_here = token_ledger.spent_today().total - ledger_at_start
        left = args.max_session_tokens - spent_here
        if left < ESTIMATE[mode]:
            stopped_early = (
                f"this run has spent {spent_here:,} tokens, leaving {left:,} of the "
                f"{args.max_session_tokens:,} cap, too little for another {mode} run"
            )
            print(f"\nSTOPPING CLEANLY: {stopped_early}")
            break

        waited = pacer.wait_for(ESTIMATE[mode])
        swap_model(orchestrator, answer_service, provider, models[effort])
        tokens_before = token_total()
        calls_before = model_calls()
        result = ask(clients[mode], repo_ids[item["repo"]], item["question"])
        used = token_total() - tokens_before
        calls = model_calls() - calls_before
        pacer.record(used)
        verdict = judge(item, result)
        row = {
            "repo": item["repo"],
            "question_id": numbered[id(item)],
            "chat_mode": mode,
            "effort": effort,
            "passed": verdict["passed"],
            "file_ok": verdict["file_ok"],
            "fact_ok": verdict["fact_ok"],
            "first_token_seconds": round(result["first_token_seconds"], 3)
            if result["first_token_seconds"] is not None
            else None,
            "total_seconds": round(result["total_seconds"], 3),
            "model_calls": calls,
            "tokens": used,
            "no_answer": result["no_answer"],
            "agents_used": result["agents_used"],
        }
        rows.append(row)
        # The answers are kept out of the result file and out of the repository; see
        # write_answers. The question text is from the committed question set.
        answers_by_arm[arm_name(mode, effort)].append({**row, "question": item["question"], "answer": result["answer"]})
        # Both written after every run, so a run stopped by the cap, by the provider or from
        # outside keeps everything it has already paid for. Writing the answers only at the end
        # once cost a whole run's answers to a stop that the result file itself survived.
        write_answers(answers_by_arm)
        if args.out:
            _write(args.out, _payload(rows, items, base_config, stopped_early, started, args, warm_up_tokens))
        reported = seen_daily_remaining()
        if reported is not None and reported <= 0:
            stopped_early = f"the provider reports {reported:,} daily tokens left (daily-limit 429)"
            print(f"\nSTOPPING CLEANLY: {stopped_early}")
            break
        mark = "PASS" if verdict["passed"] else "FAIL"
        first = f"{result['first_token_seconds']:5.2f}s" if result["first_token_seconds"] is not None else "   -  "
        print(
            f"{mark} {arm_name(mode, effort):16s} {row['question_id']:16s} first {first} "
            f"total {result['total_seconds']:5.1f}s calls {calls} tokens {used:5d}"
            + (f"  waited {waited:.0f}s" if waited else "")
        )

    payload = _payload(rows, items, base_config, stopped_early, started, args, warm_up_tokens)
    payload["session_tokens_spent"] = token_ledger.spent_today().total - ledger_at_start

    print(f"\nmodel: {payload['model']}")
    header = (
        f"{'arm':<18}{'runs':>5}{'pass':>6}{'first median':>14}{'first range':>18}"
        f"{'total med':>11}{'calls':>7}{'tokens':>8}{'no ans':>7}"
    )
    print(header)
    for mode, effort in ARMS:
        s = payload["summary"][arm_name(mode, effort)]
        median = f"{s['first_token_seconds_median']:.2f}s" if s["first_token_seconds_median"] is not None else "-"
        if s["first_token_seconds_min"] is not None:
            spread = f"{s['first_token_seconds_min']:.2f}-{s['first_token_seconds_max']:.2f}s"
        else:
            spread = "-"
        total = f"{s['total_seconds_median']:.2f}s" if s["total_seconds_median"] is not None else "-"
        print(
            f"{arm_name(mode, effort):<18}{s['runs']:>5}{s['passed']:>6}{median:>14}{spread:>18}"
            f"{total:>11}{s['model_calls_total']:>7}{s['tokens_total']:>8}{s['no_answer']:>7}"
        )

    print(f"\nthis run spent {payload['session_tokens_spent']:,} tokens of the {args.max_session_tokens:,} cap")
    spend = token_ledger.spent_today()
    print(f"this machine has spent {spend.total:,} tokens today over {spend.calls} call(s)")
    if stopped_early:
        print(f"stopped early: {stopped_early}")
    print(
        "\nThis is a weak check: two string tests per answer, no judge model, no measure of "
        "quality. Runs vary a lot - the same legacy path scored 8/15 on one day and 12/15 on "
        "another - so a difference of one or two passes between arms is not a finding."
    )

    for path in write_answers(answers_by_arm):
        print(f"wrote {path}")
    if args.out:
        print(f"wrote {_write(args.out, payload)}")
    return 0


def _payload(rows, items, base_config, stopped_early: str, started: float, args, warm_up_tokens: int) -> dict:
    spent = token_ledger.spent_today()
    return {
        "recorded": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model": f"{base_config.llm_provider} / {base_config.llm_model}",
        "llm_max_tokens": base_config.llm_max_tokens,
        "questions": len(items),
        "arms": [arm_name(m, e) for m, e in ARMS],
        "session_token_cap": args.max_session_tokens,
        "warm_up_tokens": warm_up_tokens,
        "summary_note": (
            "Each arm is summarised over the questions EVERY arm completed, so the arms are "
            "compared on the same questions. A question the cap cut short is still in rows."
        ),
        "weak_check": (
            "Two string tests per answer: is the expected file cited, and is one hand-written word "
            "present. No judge model, and nothing reads whether the explanation is correct. Runs vary "
            "a lot: the same legacy path scored 8/15 on one day and 12/15 on another."
        ),
        "stopped_early": stopped_early,
        "summary": {arm_name(m, e): summarise(rows, m, e) for m, e in ARMS},
        "rows": rows,
        "tokens_spent_today": {"input": spent.input_tokens, "output": spent.output_tokens, "calls": spent.calls},
        "seconds": round(time.perf_counter() - started, 1),
    }


def _write(out: str, payload: dict) -> Path:
    path = Path(out)
    if not path.is_absolute():
        path = PROJECT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
