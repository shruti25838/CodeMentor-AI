"""Compare fast mode and deep mode on the same questions, with a real model.

    python scripts/mode_compare.py --questions 3 --pause 65

Measures, per question and mode: time to the first streamed event, time to the first answer
token, time to the final answer, how many model calls were made, and how many tokens those
calls used. Needs a model key in the project's .env, which is loaded by the app and never
read here.

The provider's per-minute token allowance is small, so `--pause` waits between questions.
Without it a run spends most of its time in the provider's 429 backoff and the timings
measure the backoff rather than the pipeline.
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
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS = [
    "How does Signer create and verify a signature?",
    "What does URLSafeTimedSerializer do?",
    "Where is BadSignature raised?",
    "How does the timestamp signer check expiry?",
    "What is the difference between Serializer and Signer?",
]


class Usage:
    """Counts model calls and tokens for the calls made inside a `with` block."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def make_counter(usage: Usage):
    from langchain_core.callbacks import BaseCallbackHandler

    class Counter(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs) -> None:
            usage.calls += 1
            data = (response.llm_output or {}).get("token_usage") or {}
            if not data:
                for generations in response.generations:
                    for generation in generations:
                        meta = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                        usage.input_tokens += meta.get("input_tokens", 0)
                        usage.output_tokens += meta.get("output_tokens", 0)
                return
            usage.input_tokens += data.get("prompt_tokens", 0)
            usage.output_tokens += data.get("completion_tokens", 0)

    return Counter()


def serve(app) -> tuple[str, object]:
    """Run the app on a real local port, in a thread.

    FastAPI's TestClient buffers a streaming response and hands it over whole, so every
    event looks as if it arrived at the end. Measuring when an event really reaches the
    client needs a real server and a real socket.
    """
    import socket
    import threading

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise SystemExit("the local server did not start")
    return f"http://127.0.0.1:{port}", server


def run_once(client, repo_id: str, question: str, mode: str, usage: Usage) -> dict:
    """One streamed question, timed from the request going out."""
    usage.reset()
    payload = {
        "question": question,
        "repo_id": repo_id,
        "mode": mode,
        "session_id": uuid.uuid4().hex,
    }
    start = time.perf_counter()
    first_event = first_token = None
    events: list[dict] = []
    agents: list[str] = []

    with client.stream("POST", "/ask/stream", json=payload, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            now = time.perf_counter() - start
            event = json.loads(line[6:])
            if first_event is None:
                first_event = now
            if event["type"] == "token" and first_token is None:
                first_token = now
            if event["type"] == "agent":
                agents.append(event["name"])
            events.append(event)
    total = time.perf_counter() - start

    done = events[-1] if events else {}
    if done.get("type") != "done":
        raise SystemExit(f"{mode} mode failed: {done}")

    return {
        "mode": mode,
        "question": question,
        "first_event_s": first_event,
        "first_token_s": first_token,
        "total_s": total,
        "model_calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "citations": len(done.get("citations", [])),
        "agents_used": done.get("agents_used", []),
        "agent_events": agents,
        "answer_chars": sum(len(e["content"]) for e in events if e["type"] == "token"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="https://github.com/pallets/itsdangerous")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--pause", type=float, default=65.0, help="seconds between runs, for the token limit")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="codeatlas-modes-"))
    os.environ.update(
        CODEATLAS_INDEX_DIR=str(work / "indexes"),
        CODEATLAS_STATE_DIR=str(work / "state"),
        CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_CLONE_GLOBAL_PER_MINUTE="1000",
        CODEATLAS_LLM_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_LLM_GLOBAL_PER_MINUTE="1000",
    )
    os.chdir(work)

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT / ".env")
        if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
            raise SystemExit("no model key found in .env; this comparison needs a real model")

        from codeatlas.app import di
        from codeatlas.app.main import create_app

        usage = Usage()
        counter = make_counter(usage)

        # Count every model call the app makes, whichever agent makes it.
        real_get = di.LlmProvider.get_chat_model

        def counted(self):
            model = real_get(self)
            return model.with_config({"callbacks": [counter]})

        di.LlmProvider.get_chat_model = counted
        di.get_llm_provider.cache_clear()
        di.get_answer_service.cache_clear()
        di.get_agent_orchestrator.cache_clear()

        import httpx

        app = create_app()
        base_url, server = serve(app)
        client = httpx.Client(base_url=base_url, timeout=600)
        config = di.get_config()
        print(f"model: {config.llm_provider} / {config.llm_model}")

        body = client.post("/analyze-repo", json={"repo_url": args.repo_url}, timeout=600).json()
        repo_id = body["repository_id"]
        print(f"indexed {body['file_count']} files\n")

        rows: list[dict] = []
        questions = QUESTIONS[: args.questions]
        runs = [(q, mode) for q in questions for mode in ("fast", "deep")]
        for i, (question, mode) in enumerate(runs):
            row = run_once(client, repo_id, question, mode, usage)
            rows.append(row)
            print(
                f"{mode:4s} | first event {row['first_event_s']:6.2f}s | first token {row['first_token_s']:7.2f}s"
                f" | total {row['total_s']:7.2f}s | calls {row['model_calls']} | tokens {row['total_tokens']:6d}"
                f" | citations {row['citations']} | {question[:34]}"
            )
            if i < len(runs) - 1 and args.pause:
                time.sleep(args.pause)

        print()
        _summary(rows)
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")
        return 0
    finally:
        os.chdir(PROJECT)
        resolved = work.resolve()
        assert str(resolved).startswith(str(Path(tempfile.gettempdir()).resolve())), resolved
        assert "codeatlas-modes-" in resolved.name, resolved
        shutil.rmtree(resolved, ignore_errors=True)


def _summary(rows: list[dict]) -> None:
    def mean(mode: str, key: str) -> float:
        values = [r[key] for r in rows if r["mode"] == mode and r[key] is not None]
        return sum(values) / len(values) if values else 0.0

    print(f"{'':6s}{'first event':>13s}{'first token':>13s}{'total':>10s}{'calls':>8s}{'tokens':>9s}{'cites':>7s}")
    for mode in ("fast", "deep"):
        print(
            f"{mode:6s}{mean(mode, 'first_event_s'):12.2f}s{mean(mode, 'first_token_s'):12.2f}s"
            f"{mean(mode, 'total_s'):9.2f}s{mean(mode, 'model_calls'):8.1f}{mean(mode, 'total_tokens'):9.0f}"
            f"{mean(mode, 'citations'):7.1f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
