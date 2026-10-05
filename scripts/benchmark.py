"""Time chat requests against a running server (local or deployed): p50 and p95 of time to first
byte, time to the first answer token, and total response time, with cold and warm requests apart.

    python scripts/benchmark.py http://127.0.0.1:8000 --start-server --runs 3
    python scripts/benchmark.py https://your-backend.example.com --runs 2 --count 2

Each run asks --count questions from the question file (default eval/questions/itsdangerous.json,
in file order) about that file's repository through POST /ask/stream, the endpoint the website uses.

Cold and warm:
  --start-server  start a fresh uvicorn on the URL's host and port for every run, with an empty
                  temporary state folder. The first question after each start is cold (new
                  process, first request); the rest are warm. Only for a local URL.
  otherwise       the server is already running. The first request of the whole benchmark is
                  reported as "first": it is cold only if the server had been idle or asleep,
                  which this script cannot know. Everything after it is warm.

Model: with --start-server and no GROQ_API_KEY/OPENAI_API_KEY in the environment, the server uses
its built-in stand-in, which answers instantly, so the times cover everything except the model.
Against a server that has a key, every question makes real model calls (3 per question) and uses
quota. The output says which happened: the stand-in's answer is recognized.

Rate limits are respected: requests are spaced by --delay seconds, at most 30 questions are sent
per benchmark, and a 429 stops the benchmark. Indexing the repository counts once per run against
the clone limit (10 per minute per client by default).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "eval" / "questions" / "itsdangerous.json"
STAND_IN_ANSWER = "LLM provider not configured"
MAX_QUESTIONS = 30


@dataclass
class Sample:
    ttfb_ms: float
    first_token_ms: float | None
    total_ms: float
    server_llm_ms: float | None
    stand_in: bool


@dataclass
class Group:
    samples: list[Sample] = field(default_factory=list)


class RateLimitedError(Exception):
    pass


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest value with at least pct% of values at or below it."""
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil
    return ordered[int(rank) - 1]


def post_json(base: str, path: str, payload: dict, timeout: float) -> tuple[int, dict, dict]:
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        body = err.read()
        return err.code, dict(err.headers), json.loads(body) if body else {}


def ask(base: str, repo_id: str, question: str, timeout: float) -> Sample:
    """One streamed question; times are measured from just before the request is sent."""
    payload = json.dumps({"question": question, "repo_id": repo_id}).encode()
    request = urllib.request.Request(base + "/ask/stream", data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        resp = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as err:
        if err.code == 429:
            raise RateLimitedError(err.headers.get("Retry-After", "?"))
        raise SystemExit(f"/ask/stream returned {err.code}: {err.read()[:200]!r}")
    ttfb = first_token = None
    answer = ""
    done: dict = {}
    with resp:
        for raw in resp:
            now = (time.perf_counter() - start) * 1000
            if ttfb is None:
                ttfb = now
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "token":
                if first_token is None:
                    first_token = now
                answer += event.get("content", "")
            elif event.get("type") == "done":
                done = event
            elif event.get("type") == "error":
                raise SystemExit(f"server reported an error: {event.get('content')}")
    total = (time.perf_counter() - start) * 1000
    return Sample(
        ttfb_ms=ttfb if ttfb is not None else total,
        first_token_ms=first_token,
        total_ms=total,
        server_llm_ms=done.get("timings_ms", {}).get("llm"),
        stand_in=STAND_IN_ANSWER in answer,
    )


def wait_until_up(base: str, proc: subprocess.Popen, timeout: float, log: Path) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if proc.poll() is not None:
            tail = log.read_text(errors="replace")[-2000:] if log.exists() else ""
            raise SystemExit(f"the server exited during startup:\n{tail}")
        try:
            with urllib.request.urlopen(base + "/repos", timeout=2) as resp:
                if resp.status == 200:
                    return (time.perf_counter() - start) * 1000
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.1)
    raise SystemExit(f"the server did not answer within {timeout:.0f} s")


def start_server(base: str, workdir: Path) -> subprocess.Popen:
    parts = urlsplit(base)
    if parts.hostname not in ("127.0.0.1", "localhost"):
        raise SystemExit("--start-server only works with a local URL (127.0.0.1 or localhost)")
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        CODEATLAS_INDEX_DIR=str(workdir / "indexes"),
        CODEATLAS_STATE_DIR=str(workdir / "state"),
    )
    if not (env.get("GROQ_API_KEY") or env.get("OPENAI_API_KEY")):
        env["CODEATLAS_LLM_PROVIDER"] = "none"
    cmd = [sys.executable, "-m", "uvicorn", "codeatlas.app.main:app", "--host", parts.hostname]
    cmd += ["--port", str(parts.port or 80), "--log-level", "warning"]
    # Clones go to the work folder's .codeatlas/repos. The app's load_dotenv() does not override
    # variables already set here, so CODEATLAS_LLM_PROVIDER=none holds even if a .env has keys.
    log = (workdir / "server.log").open("wb")
    return subprocess.Popen(cmd, cwd=workdir, env=env, stdout=log, stderr=subprocess.STDOUT)


def index(base: str, repo_url: str, timeout: float) -> tuple[str, float]:
    start = time.perf_counter()
    status, headers, body = post_json(base, "/analyze-repo", {"repo_url": repo_url}, timeout)
    if status == 429:
        raise RateLimitedError(headers.get("Retry-After", "?"))
    if status != 200:
        raise SystemExit(f"/analyze-repo returned {status}: {body}")
    return body["repository_id"], (time.perf_counter() - start) * 1000


def report(name: str, group: Group) -> None:
    n = len(group.samples)
    if not n:
        print(f"{name:<6} n=0")
        return
    cols = []
    for label, values in (
        ("ttfb", [s.ttfb_ms for s in group.samples]),
        ("first_token", [s.first_token_ms for s in group.samples if s.first_token_ms is not None]),
        ("total", [s.total_ms for s in group.samples]),
    ):
        if values:
            cols.append(f"{label} p50 {percentile(values, 50):7.1f} p95 {percentile(values, 95):7.1f}")
    print(f"{name:<6} n={n:<3} " + " | ".join(cols) + "  (ms)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--count", type=int, default=3, help="questions per run (default 3)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1)")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--start-server", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    spec = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = spec["questions"][: args.count]
    if len(questions) * args.runs > MAX_QUESTIONS:
        raise SystemExit(f"at most {MAX_QUESTIONS} questions per benchmark; lower --runs or --count")

    cold, warm = Group(), Group()
    index_ms: list[float] = []
    startup_ms: list[float] = []
    try:
        for run in range(1, args.runs + 1):
            proc = workdir = None
            if args.start_server:
                workdir = Path(tempfile.mkdtemp(prefix="codeatlas-bench-"))
                proc = start_server(base, workdir)
            try:
                if proc:
                    startup_ms.append(wait_until_up(base, proc, 120, workdir / "server.log"))
                if proc or run == 1:
                    repo_id, ms = index(base, spec["repo_url"], args.timeout)
                    index_ms.append(ms)
                    time.sleep(args.delay)
                for i, q in enumerate(questions):
                    sample = ask(base, repo_id, q["question"], args.timeout)
                    is_cold = i == 0 and (proc is not None or run == 1)
                    (cold if is_cold else warm).samples.append(sample)
                    time.sleep(args.delay)
            finally:
                if proc:
                    proc.terminate()
                    proc.wait(timeout=30)
                if workdir:
                    shutil.rmtree(workdir, ignore_errors=True)
    except RateLimitedError as exc:
        print(f"STOPPED: the server answered 429 (rate limited, Retry-After {exc}). Results so far:")

    samples = cold.samples + warm.samples
    stand_in = [s.stand_in for s in samples]
    print(f"server: {base} ({'started fresh for each run' if args.start_server else 'already running'})")
    print(
        f"questions: {args.questions.name} {[q['id'] for q in questions]} x {args.runs} runs, repo {spec['repo_url']}"
    )
    if stand_in and all(stand_in):
        print("model: NOT TIMED. The server answered with its stand-in (no model key), so every time below")
        print("       covers retrieval, prompt building and streaming, but not the language model.")
    elif any(stand_in):
        print("model: MIXED. Some answers came from the stand-in; do not compare these times.")
    else:
        print("model: TIMED. The server called its language model (3 calls per question).")
    if startup_ms:
        print(f"server startup until /repos answered: p50 {percentile(startup_ms, 50):.0f} ms (n={len(startup_ms)})")
    if index_ms:
        print(f"/analyze-repo: p50 {percentile(index_ms, 50):.0f} ms (n={len(index_ms)})")
    report("cold" if args.start_server else "first", cold)
    report("warm", warm)
    llm = [s.server_llm_ms for s in samples if s.server_llm_ms is not None]
    if llm:
        print(f"server-reported llm stage: p50 {percentile(llm, 50):.1f} ms (stand-in or model, per the line above)")
    print("ttfb = first line of the event stream (the server sends a status event before retrieval);")
    print("first_token = first answer token; total = until the stream closed. Nearest-rank percentiles:")
    print("with fewer than 20 requests in a group, p95 is the slowest one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
