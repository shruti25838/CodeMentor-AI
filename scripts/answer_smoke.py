"""A weak, deterministic check that answers mention the right file and the right word.

    python scripts/answer_smoke.py                  # both modes, all questions
    python scripts/answer_smoke.py --mode deep --limit 5

A question passes when the cited files include the file the answer should come from, and
the answer text contains a short fact written by hand from the source. There is no judge
model and no scoring of prose: this catches an answer that cites nothing, cites the wrong
file, or has drifted off the subject entirely. **It does not measure answer quality.** An
answer can pass while being badly wrong about everything else, and a good answer can fail
for using a synonym.

Needs a model key in the project's .env. Not part of CI: it costs real tokens and its
result depends on the model and the provider's mood.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from codeatlas.observability import token_ledger  # noqa: E402
from scripts.token_budget import DEFAULT_MAX_DAILY_TOKENS  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS = PROJECT / "eval" / "answer_smoke_questions.json"

REPOS = {
    "itsdangerous": "https://github.com/pallets/itsdangerous",
    "flask": "https://github.com/pallets/flask",
}

# Groq's free tier allows 8,000 tokens a minute. Staying a little under leaves room for the
# request that is already in flight when the window rolls over.
TOKEN_BUDGET_PER_MINUTE = 7000
ESTIMATE = {"fast": 4000, "deep": 7000}


def free_megabytes() -> int:
    """Physical memory still available, or -1 when it cannot be read.

    A long run was once stopped from outside because the machine ran out of memory, which
    loses the runs still to come. Checking first lets the run stop on its own terms, with
    everything finished so far already written to disk.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return -1
            return int(status.ullAvailPhys // (1024 * 1024))
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1
    return -1


def key_of(row: dict) -> tuple[str, str, str]:
    return (row["repo"], row["question"], row["mode"])


def load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []


@dataclass
class Pacer:
    """Waits only as long as the provider's per-minute token allowance requires."""

    budget: int = TOKEN_BUDGET_PER_MINUTE
    window: list[tuple[float, int]] = field(default_factory=list)

    def spent_recently(self) -> int:
        cutoff = time.monotonic() - 60
        self.window = [(t, n) for t, n in self.window if t > cutoff]
        return sum(n for _, n in self.window)

    def wait_for(self, estimate: int) -> float:
        waited = 0.0
        while self.window and self.spent_recently() + estimate > self.budget:
            oldest = self.window[0][0]
            sleep_for = max(0.5, 61 - (time.monotonic() - oldest))
            time.sleep(sleep_for)
            waited += sleep_for
        return waited

    def record(self, tokens: int) -> None:
        self.window.append((time.monotonic(), tokens))


def failure_counts() -> dict[str, int]:
    """413s and 429s so far, read from the metrics the pipeline already records."""
    from prometheus_client import REGISTRY

    out = {}
    for reason in ("payload_too_large", "rate_limited"):
        total = 0.0
        for metric in REGISTRY.collect():
            if metric.name != "codeatlas_agent_failures":
                continue
            for sample in metric.samples:
                if sample.name.endswith("_total") and sample.labels.get("reason") == reason:
                    total += sample.value
        out[reason] = int(total)
    return out


def token_total() -> int:
    from prometheus_client import REGISTRY

    total = 0.0
    for metric in REGISTRY.collect():
        if metric.name != "codeatlas_agent_tokens":
            continue
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return int(total)


# Set by the log watcher below whenever the provider states its daily token position.
_seen_daily: list[int] = []


def seen_daily_remaining() -> int | None:
    """Tokens left today, as last stated by the provider, or None if it never has."""
    return _seen_daily[-1] if _seen_daily else None


def watch_for_daily_limit() -> None:
    """Listen to the app's own warnings for the one message that names the daily budget.

    The figure is never in a header; it appears only in the body of a 429 after the budget
    is gone. The pipeline already logs that body when a call is refused, so reading the log
    costs nothing and needs no extra request.
    """
    import logging

    from token_budget import parse_tpd

    class Watcher(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            found = parse_tpd(record.getMessage())
            if found:
                limit, used = found
                _seen_daily.append(max(limit - used, 0))

    handler = Watcher(level=logging.WARNING)
    logging.getLogger().addHandler(handler)
    logging.getLogger("codeatlas").addHandler(handler)


def looks_like_no_answer(text: str) -> bool:
    """Whether the model never actually answered: empty, an error, or a quota message.

    A run like this measures the provider's mood, not the pipeline, so it is recorded and
    redone rather than counted as a failure.
    """
    from codeatlas.services.agents.coding_mentor_agent import EMPTY_ANSWER
    from codeatlas.services.agents.orchestration import ANSWER_ERROR_PREFIX
    from codeatlas.services.llm.quota import MESSAGES

    stripped = (text or "").strip()
    if not stripped:
        return True
    if stripped.startswith(ANSWER_ERROR_PREFIX) or stripped.startswith(EMPTY_ANSWER[:40]):
        return True
    if "Top relevant locations:" in stripped:
        return True
    return any(message in stripped for message in MESSAGES.values())


def ask(client, repo_id: str, question: str, mode: str) -> dict:
    payload = {"question": question, "repo_id": repo_id, "mode": mode, "session_id": uuid.uuid4().hex}
    started = time.perf_counter()
    answer_parts: list[str] = []
    done: dict = {}
    with client.stream("POST", "/ask/stream", json=payload, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "token":
                answer_parts.append(event["content"])
            elif event["type"] == "done":
                done = event
            elif event["type"] == "error":
                done = {"citations": [], "error": event["content"]}
    answer = "".join(answer_parts)
    return {
        "answer": answer,
        "citations": done.get("citations", []),
        "agents_used": done.get("agents_used", []),
        "seconds": time.perf_counter() - started,
        "error": done.get("error"),
        "no_answer": bool(done.get("error")) or looks_like_no_answer(answer),
    }


def judge(item: dict, result: dict) -> dict:
    """Two string checks. Nothing here looks at whether the answer is any good."""
    cited_files = {c.split(" (")[0].split(" | ")[0].strip().replace("\\", "/") for c in result["citations"]}
    expected = item["expected_file"]
    file_ok = any(f == expected or f.endswith("/" + expected.split("/")[-1]) and expected in f for f in cited_files)
    if not file_ok:
        file_ok = any(expected in f or f in expected for f in cited_files)
    fact_ok = item["fact"].lower() in result["answer"].lower()
    return {
        "file_ok": file_ok,
        "fact_ok": fact_ok,
        "passed": file_ok and fact_ok,
        "cited_files": sorted(cited_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "deep", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="first N questions per repository")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip runs already present in --out, and keep appending to it",
    )
    parser.add_argument(
        "--max-daily-tokens",
        type=int,
        default=DEFAULT_MAX_DAILY_TOKENS,
        help="stop cleanly once this machine has spent this many tokens today",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=600,
        help="stop cleanly before a run if less memory than this is free",
    )
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

    work = Path(tempfile.mkdtemp(prefix="codeatlas-smoke-"))
    os.environ.update(
        CODEATLAS_INDEX_DIR=str(work / "indexes"),
        CODEATLAS_STATE_DIR=str(work / "state"),
        CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_CLONE_GLOBAL_PER_MINUTE="1000",
        CODEATLAS_LLM_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_LLM_GLOBAL_PER_MINUTE="1000",
    )
    os.chdir(work)
    started = time.perf_counter()

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env")
        if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise SystemExit("no model key found in .env; this check needs a real model")

        import httpx

        from codeatlas.app import di
        from codeatlas.app.main import create_app
        from scripts.mode_compare import serve

        watch_for_daily_limit()
        app = create_app()
        base_url, server = serve(app)
        client = httpx.Client(base_url=base_url, timeout=600)
        config = di.get_config()
        print(f"model: {config.llm_provider} / {config.llm_model}")

        repo_ids: dict[str, str] = {}
        for name in sorted({i["repo"] for i in items}):
            body = client.post("/analyze-repo", json={"repo_url": REPOS[name]}, timeout=600).json()
            repo_ids[name] = body["repository_id"]
            print(f"indexed {name}: {body['file_count']} files")
        print()

        modes = ["fast", "deep"] if args.mode == "both" else [args.mode]
        pacer = Pacer()
        before_failures = failure_counts()

        out_path = Path(args.out) if args.out else None
        rows = load_done(out_path) if (args.resume and out_path) else []
        # A run that got no model answer is not finished work: it measured the provider's
        # mood. It stays in the file for the record but is done again.
        already = {key_of(r) for r in rows if not r.get("no_answer")}
        redo = [r for r in rows if r.get("no_answer")]
        rows = [r for r in rows if not r.get("no_answer")]
        if already:
            print(f"resuming: {len(already)} run(s) already answered, skipping those")
        if redo:
            print(f"{len(redo)} earlier run(s) got no model answer and will be done again")

        planned = [(mode, item) for mode in modes for item in items]
        remaining = [(m, i) for m, i in planned if (i["repo"], i["question"], m) not in already]
        print(f"{len(remaining)} run(s) to do of {len(planned)}\n")

        stopped_early = ""
        for mode, item in remaining:
            free = free_megabytes()
            if 0 <= free < args.min_free_mb:
                stopped_early = f"only {free} MB of memory free, need {args.min_free_mb} MB"
                print(f"\nSTOPPING CLEANLY: {stopped_early}")
                break
            # The running total on disk, not a guess from the provider's headers. It
            # survives the process, so it counts what earlier runs spent today as well.
            allowed, left = token_ledger.within_budget(args.max_daily_tokens)
            if not allowed or left < ESTIMATE[mode]:
                spent = token_ledger.spent_today().total
                stopped_early = (
                    f"this machine has spent {spent:,} tokens today, leaving {left:,} of the "
                    f"{args.max_daily_tokens:,} limit, too little for another {mode} run"
                )
                print(f"\nSTOPPING CLEANLY: {stopped_early}")
                break
            waited = pacer.wait_for(ESTIMATE[mode])
            tokens_before = token_total()
            result = ask(client, repo_ids[item["repo"]], item["question"], mode)
            used = token_total() - tokens_before
            pacer.record(used)
            verdict = judge(item, result)
            rows.append(
                {
                    **item,
                    "mode": mode,
                    **verdict,
                    "seconds": result["seconds"],
                    "tokens": used,
                    "no_answer": result["no_answer"],
                }
            )
            # Written after every run, so a run stopped from outside keeps what is finished.
            if out_path:
                out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            # The provider only ever states the daily figure while refusing a call; when it
            # does, that exact number beats the local count.
            reported = seen_daily_remaining()
            if reported is not None and reported <= 0:
                stopped_early = f"the provider reports {reported:,} daily tokens left"
                print(f"\nSTOPPING CLEANLY: {stopped_early}")
                break
            mark = "PASS" if verdict["passed"] else "FAIL"
            why = "" if verdict["passed"] else f"  (file {verdict['file_ok']}, fact {verdict['fact_ok']})"
            print(
                f"{mark} {mode:4s} {item['repo']:12s} {item['question'][:46]:46s}"
                f" {result['seconds']:5.1f}s{why}" + (f"  waited {waited:.0f}s" if waited else "")
            )

        after_failures = failure_counts()
        elapsed = time.perf_counter() - started

        print(f"\nmodel: {config.llm_provider} / {config.llm_model}")
        for mode in modes:
            mode_rows = [r for r in rows if r["mode"] == mode]
            passed = sum(1 for r in mode_rows if r["passed"])
            files = sum(1 for r in mode_rows if r["file_ok"])
            facts = sum(1 for r in mode_rows if r["fact_ok"])
            print(
                f"{mode:4s}: {passed}/{len(mode_rows)} passed"
                f"  (expected file cited {files}/{len(mode_rows)}, fact present {facts}/{len(mode_rows)})"
            )
        print(
            f"413 payload-too-large: {after_failures['payload_too_large'] - before_failures['payload_too_large']}"
            f" | 429 rate-limited: {after_failures['rate_limited'] - before_failures['rate_limited']}"
        )
        spend = token_ledger.spent_today()
        print(f"tokens spent today on this machine: {spend.total:,} over {spend.calls} call(s)")
        print(f"wall time: {elapsed:.0f}s")
        print("\nThis is a weak check: two string tests per answer, no judge model, no measure of quality.")

        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"wrote {args.out}")
        server.should_exit = True
        return 0
    finally:
        os.chdir(PROJECT)
        resolved = work.resolve()
        assert str(resolved).startswith(str(Path(tempfile.gettempdir()).resolve())), resolved
        assert "codeatlas-smoke-" in resolved.name, resolved
        shutil.rmtree(resolved, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
