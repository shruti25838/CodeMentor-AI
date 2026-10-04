"""Time indexing and one chat question, stage by stage, in a fresh process.

    python scripts/time_stages.py https://github.com/pallets/itsdangerous --runs 3

Run 1 is cold (first request in a new process: lazy imports, first parser load, empty caches);
later runs are warm. Each run indexes the repo again and asks the question against it, through
the real endpoints (/analyze-repo and /ask/stream) with an in-process client. Everything is
written to a temporary folder that is deleted afterwards. With the index cache on (the default),
runs after the first reuse the first index; set CODEATLAS_INDEX_CACHE=false to index every run.

With no GROQ_API_KEY or OPENAI_API_KEY set, the model is replaced by the built-in stand-in that
answers instantly, so the `llm` stage and first-token times exclude the model.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_url")
    parser.add_argument("--question", default="How does Signer create and check a signature?")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="codeatlas-timing-"))
    has_key = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    os.environ.update(
        CODEATLAS_INDEX_DIR=str(workdir / "indexes"),
        CODEATLAS_STATE_DIR=str(workdir / "state"),
        CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_CLONE_GLOBAL_PER_MINUTE="1000",
        CODEATLAS_LLM_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_LLM_GLOBAL_PER_MINUTE="1000",
    )
    if not has_key:
        os.environ["CODEATLAS_LLM_PROVIDER"] = "none"
    sys.path.insert(0, str(ROOT))
    os.chdir(workdir)  # clones go to ./.codeatlas/repos

    try:
        start = time.perf_counter()
        from fastapi.testclient import TestClient

        from codeatlas.app.main import create_app

        client = TestClient(create_app())
        print(f"model: {'real (key found)' if has_key else 'NOT TIMED: no model key, stand-in answers instantly'}")
        print(f"app import + startup: {(time.perf_counter() - start) * 1000:.0f} ms")
        for run in range(1, args.runs + 1):
            label = "cold" if run == 1 else "warm"
            resp = client.post("/analyze-repo", json={"repo_url": args.repo_url})
            resp.raise_for_status()
            body = resp.json()
            print(
                f"run {run} ({label}) index: {resp.headers['Server-Timing']}"
                f" | files={body['file_count']} edges={body['dependency_edges']}"
            )
            resp = client.post("/ask/stream", json={"question": args.question, "repo_id": body["repository_id"]})
            events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
            done = events[-1]
            if done.get("type") != "done":
                raise SystemExit(f"chat failed: {done}")
            print(f"run {run} ({label}) chat:  {json.dumps(done['timings_ms'])}")
    finally:
        os.chdir(ROOT)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
