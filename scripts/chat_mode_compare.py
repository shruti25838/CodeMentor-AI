"""Legacy against grounded on the same smoke-check questions, with a real model.

    python scripts/chat_mode_compare.py --out eval/results/experiments/chat-modes.json

Both modes answer the same fifteen questions about the same two indexed repositories, warm,
one run each. What is measured:

* **time to the first streamed token** - the wait a visitor actually sits through. Legacy
  finishes three model calls before it sends anything; grounded forwards the model's own
  tokens, so this is the number the two shapes differ on by design.
* **total time**, **model calls** and **tokens**, so the first number can be read against
  what it cost.
* the **same deterministic pass check** `scripts/answer_smoke.py` uses - is the expected file
  cited, and is a hand-written fact present. **It is a weak check**: two string tests, no
  judge model, nothing that reads whether the explanation is correct. An answer can pass
  while being wrong about everything else and a good answer can fail on a synonym.
* the **share of cited files that were actually in the prompt**. For grounded this is true by
  construction and the measurement is a check on that claim; for legacy it is a real question,
  because the citations come from one retrieval and the answer from another.

Spending is bounded three ways: the pacer keeps the per-minute token allowance, the local
ledger stops the run before it passes `--max-session-tokens`, and the provider's own statement
of a daily limit stops it immediately. Nothing here prints a key, a question's answer text, or
anything from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from codeatlas.observability import token_ledger  # noqa: E402
from codeatlas.observability.agent_metrics import AgentUsageCallback  # noqa: E402
from scripts.answer_smoke import (  # noqa: E402
    QUESTIONS,
    REPOS,
    Pacer,
    free_megabytes,
    judge,
    looks_like_no_answer,
    seen_daily_remaining,
    token_total,
    watch_for_daily_limit,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LEGACY = "legacy"
GROUNDED = "grounded"

# What one question is expected to cost in each mode, used only for pacing. Legacy makes three
# model calls to grounded's one, so it is budgeted higher.
ESTIMATE = {LEGACY: 4000, GROUNDED: 2000}


class PromptRecorder(AgentUsageCallback):
    """Every prompt the model was shown, on top of the metrics callback the app already uses.

    It **subclasses** `AgentUsageCallback` rather than sitting beside it, and that is the whole
    point. `with_config({"callbacks": [...]})` replaces the handler list rather than adding to
    it, so a recorder passed on its own silently switches off the token and call counters - the
    first version of this script did exactly that and reported 0 tokens for runs that really
    spent them, which also left the budget guard with nothing to guard. Inheriting means one
    handler does both jobs and the counters keep working.

    A callback rather than a wrapper around the model, because the legacy path composes
    `prompt | model` into a chain and an object that is not a Runnable cannot stand in it.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def reset(self) -> None:
        self.prompts = []

    def text(self) -> str:
        return "\n".join(self.prompts)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        for batch in messages:
            for message in batch:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    self.prompts.append(content)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.prompts.extend(p for p in prompts if isinstance(p, str))

    def on_llm_new_token(self, token, **kwargs) -> None:
        """Nothing to do; defined so langchain does not warn once per token."""


def instrument(provider, recorder: PromptRecorder):
    """Have every chat model the provider hands out report to the recorder, metrics included."""
    original = provider.get_chat_model

    def recorded():
        return original().with_config({"callbacks": [recorder]})

    provider.get_chat_model = recorded
    return provider


def model_calls() -> int:
    """Model calls so far, from the counter the pipeline already keeps."""
    from prometheus_client import REGISTRY

    total = 0.0
    for metric in REGISTRY.collect():
        if metric.name != "codeatlas_agent_model_calls":
            continue
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return int(total)


def ask(client, repo_id: str, question: str) -> dict:
    """One streamed question, timing the first token the visitor would see."""
    payload = {"question": question, "repo_id": repo_id, "mode": "fast", "session_id": uuid.uuid4().hex}
    started = time.perf_counter()
    first_token = None
    parts: list[str] = []
    done: dict = {}
    with client.stream("POST", "/ask/stream", json=payload, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                if first_token is None:
                    first_token = time.perf_counter() - started
                parts.append(event["content"])
            elif event["type"] == "done":
                done = event
            elif event["type"] == "error":
                done = {"citations": [], "error": event["content"]}
    answer = "".join(parts)
    return {
        "answer": answer,
        "citations": done.get("citations", []),
        "agents_used": done.get("agents_used", []),
        "first_token_seconds": first_token,
        "total_seconds": time.perf_counter() - started,
        "error": done.get("error"),
        "no_answer": bool(done.get("error")) or looks_like_no_answer(answer),
    }


def cited_in_prompt(citations: list[str], prompt_text: str) -> tuple[int, int]:
    """How many cited files appear in what the model was shown, out of how many were cited."""
    files = {c.split(" (")[0].split(" | ")[0].strip().replace("\\", "/") for c in citations}
    files = {f for f in files if f}
    if not files:
        return 0, 0
    haystack = prompt_text.replace("\\", "/")
    return sum(1 for f in files if f in haystack), len(files)


def summarise(rows: list[dict], mode: str) -> dict:
    mine = [r for r in rows if r["chat_mode"] == mode]
    answered = [r for r in mine if not r["no_answer"]]
    first = [r["first_token_seconds"] for r in answered if r["first_token_seconds"] is not None]
    cited, total_cited = 0, 0
    for row in mine:
        cited += row["cited_in_prompt"]
        total_cited += row["citations_total"]
    return {
        "runs": len(mine),
        "answered": len(answered),
        "passed": sum(1 for r in mine if r["passed"]),
        "expected_file_cited": sum(1 for r in mine if r["file_ok"]),
        "fact_present": sum(1 for r in mine if r["fact_ok"]),
        "no_answer": sum(1 for r in mine if r["no_answer"]),
        "first_token_seconds_median": round(_median(first), 2) if first else None,
        "total_seconds_median": round(_median([r["total_seconds"] for r in mine]), 2) if mine else None,
        "model_calls_total": sum(r["model_calls"] for r in mine),
        "tokens_total": sum(r["tokens"] for r in mine),
        "cited_files_in_prompt": cited,
        "cited_files_total": total_cited,
    }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0, help="first N questions per repository")
    parser.add_argument(
        "--max-session-tokens",
        type=int,
        default=60_000,
        help="stop cleanly once this machine has spent this many tokens today",
    )
    parser.add_argument("--min-free-mb", type=int, default=600)
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.limit:
        kept: dict[str, int] = {}
        chosen = []
        for item in items:
            if kept.get(item["repo"], 0) < args.limit:
                chosen.append(item)
                kept[item["repo"]] = kept.get(item["repo"], 0) + 1
        items = chosen

    work = Path(tempfile.mkdtemp(prefix="codeatlas-chatmode-"))
    os.environ.update(
        CODEATLAS_INDEX_DIR=str(work / "indexes"),
        CODEATLAS_STATE_DIR=str(work / "state"),
        # The run works in a temporary directory, and the ledger path is relative by default,
        # so without this the day's spend would be written into the temp folder and thrown
        # away - and the session cap would start from zero on every run.
        CODEATLAS_TOKEN_LEDGER=str(PROJECT / ".codeatlas" / "token-spend.json"),
        CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_CLONE_GLOBAL_PER_MINUTE="1000",
        CODEATLAS_LLM_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_LLM_GLOBAL_PER_MINUTE="1000",
    )
    os.chdir(work)
    started = time.perf_counter()

    from dataclasses import replace

    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    # By name only. The value is never read into a variable, printed or logged.
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise SystemExit("no model key found in .env; this comparison needs a real model")

    import httpx

    from codeatlas.app import di
    from codeatlas.app.main import create_app
    from scripts.mode_compare import serve

    watch_for_daily_limit()
    recorder = PromptRecorder()
    instrument(di.get_llm_provider(), recorder)

    base_config = di.get_config()
    apps = {
        LEGACY: create_app(replace(base_config, chat_mode=LEGACY)),
        GROUNDED: create_app(replace(base_config, chat_mode=GROUNDED)),
    }
    clients = {}
    for mode, app in apps.items():
        base_url, _ = serve(app)
        clients[mode] = httpx.Client(base_url=base_url, timeout=600)

    print(f"model: {base_config.llm_provider} / {base_config.llm_model}")
    print(f"session token cap: {args.max_session_tokens:,}")

    # Indexed once; the DI singletons are shared, so both apps see the same index.
    repo_ids: dict[str, str] = {}
    for name in sorted({i["repo"] for i in items}):
        body = clients[LEGACY].post("/analyze-repo", json={"repo_url": REPOS[name]}, timeout=600).json()
        repo_ids[name] = body["repository_id"]
        print(f"indexed {name}: {body['file_count']} files")

    # A warm-up question per mode, not recorded: the first call to a mode pays for whatever
    # the process has not done yet, and the comparison is of warm requests.
    warm = items[0]
    for mode in (LEGACY, GROUNDED):
        ask(clients[mode], repo_ids[warm["repo"]], warm["question"])
    print("warmed both modes\n")

    pacer = Pacer()
    rows: list[dict] = []
    stopped_early = ""
    plan = [(mode, item) for item in items for mode in (LEGACY, GROUNDED)]

    for mode, item in plan:
        free = free_megabytes()
        if 0 <= free < args.min_free_mb:
            stopped_early = f"only {free} MB of memory free, need {args.min_free_mb} MB"
            print(f"\nSTOPPING CLEANLY: {stopped_early}")
            break
        allowed, left = token_ledger.within_budget(args.max_session_tokens)
        if not allowed or left < ESTIMATE[mode]:
            spent = token_ledger.spent_today().total
            stopped_early = (
                f"this machine has spent {spent:,} tokens today, leaving {left:,} of the "
                f"{args.max_session_tokens:,} cap, too little for another {mode} run"
            )
            print(f"\nSTOPPING CLEANLY: {stopped_early}")
            break

        waited = pacer.wait_for(ESTIMATE[mode])
        recorder.reset()
        tokens_before = token_total()
        calls_before = model_calls()
        result = ask(clients[mode], repo_ids[item["repo"]], item["question"])
        used = token_total() - tokens_before
        calls = model_calls() - calls_before
        pacer.record(used)
        verdict = judge(item, result)
        in_prompt, total_cited = cited_in_prompt(result["citations"], recorder.text())
        rows.append(
            {
                "repo": item["repo"],
                "question_id": f"{item['repo']}-{items.index(item) + 1}",
                "chat_mode": mode,
                "passed": verdict["passed"],
                "file_ok": verdict["file_ok"],
                "fact_ok": verdict["fact_ok"],
                "first_token_seconds": round(result["first_token_seconds"], 3)
                if result["first_token_seconds"] is not None
                else None,
                "total_seconds": round(result["total_seconds"], 3),
                "model_calls": calls,
                "tokens": used,
                "citations_total": total_cited,
                "cited_in_prompt": in_prompt,
                "no_answer": result["no_answer"],
                "agents_used": result["agents_used"],
            }
        )
        # Written after every run, so a run stopped by the cap, by the provider or from
        # outside keeps what is finished rather than losing tokens that were already spent.
        if args.out:
            _write(args.out, _payload(rows, items, base_config, stopped_early, started))
        reported = seen_daily_remaining()
        if reported is not None and reported <= 0:
            stopped_early = f"the provider reports {reported:,} daily tokens left"
            print(f"\nSTOPPING CLEANLY: {stopped_early}")
            break
        mark = "PASS" if verdict["passed"] else "FAIL"
        first = f"{result['first_token_seconds']:5.2f}s" if result["first_token_seconds"] is not None else "   -  "
        print(
            f"{mark} {mode:8s} {item['repo']:12s} first {first} total {result['total_seconds']:5.1f}s "
            f"calls {calls} tokens {used:5d}" + (f"  waited {waited:.0f}s" if waited else "")
        )

    summary = {mode: summarise(rows, mode) for mode in (LEGACY, GROUNDED)}
    spent = token_ledger.spent_today()
    payload = {
        "recorded": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model": f"{base_config.llm_provider} / {base_config.llm_model}",
        "questions": len(items),
        "weak_check": (
            "Two string tests per answer: is the expected file cited, and is one hand-written word "
            "present. No judge model, and nothing reads whether the explanation is correct."
        ),
        "stopped_early": stopped_early,
        "summary": summary,
        "rows": rows,
        "tokens_spent_today": {"input": spent.input_tokens, "output": spent.output_tokens, "calls": spent.calls},
        "seconds": round(time.perf_counter() - started, 1),
    }

    print(f"\nmodel: {payload['model']}")
    header = f"{'mode':<10}{'runs':>5}{'passed':>8}{'file':>6}{'fact':>6}{'first tok':>11}{'total':>8}{'calls':>7}{'tokens':>8}"
    print(header)
    for mode in (LEGACY, GROUNDED):
        s = summary[mode]
        first = f"{s['first_token_seconds_median']:.2f}s" if s["first_token_seconds_median"] is not None else "-"
        print(
            f"{mode:<10}{s['runs']:>5}{s['passed']:>8}{s['expected_file_cited']:>6}{s['fact_present']:>6}"
            f"{first:>11}{s['total_seconds_median']:>7.1f}s{s['model_calls_total']:>7}{s['tokens_total']:>8}"
        )
        print(
            f"          cited files in the prompt: {s['cited_files_in_prompt']} of {s['cited_files_total']}"
            f"; runs with no model answer: {s['no_answer']}"
        )
    print(f"\ntokens spent today: {spent.total:,} ({spent.calls} calls)")
    if stopped_early:
        print(f"stopped early: {stopped_early}")

    if args.out:
        print(f"wrote {_write(args.out, payload)}")
    return 0


def _payload(rows, items, base_config, stopped_early: str, started: float) -> dict:
    spent = token_ledger.spent_today()
    return {
        "recorded": datetime.now(UTC).strftime("%Y-%m-%d"),
        "model": f"{base_config.llm_provider} / {base_config.llm_model}",
        "questions": len(items),
        "weak_check": (
            "Two string tests per answer: is the expected file cited, and is one hand-written word "
            "present. No judge model, and nothing reads whether the explanation is correct."
        ),
        "stopped_early": stopped_early,
        "summary": {mode: summarise(rows, mode) for mode in (LEGACY, GROUNDED)},
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
