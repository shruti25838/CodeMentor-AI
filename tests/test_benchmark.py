import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parent.parent / "scripts" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def test_percentile_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert benchmark.percentile(values, 50) == 3.0
    assert benchmark.percentile(values, 95) == 5.0
    assert benchmark.percentile([7.0], 95) == 7.0
    assert benchmark.percentile(list(map(float, range(1, 101))), 95) == 95.0


class _Handler(BaseHTTPRequestHandler):
    answer = "LLM provider not configured. Set CODEATLAS_LLM_PROVIDER and relevant API keys."
    status = 200

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.status != 200:
            self.send_response(self.status)
            self.send_header("Retry-After", "60")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"type": "status", "content": "Retrieving context..."},
            *({"type": "token", "content": word + " "} for word in self.answer.split()),
            {"type": "done", "citations": [], "timings_ms": {"llm": 4.0, "total": 9.0}},
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    _Handler.answer = "LLM provider not configured. Set CODEATLAS_LLM_PROVIDER and relevant API keys."
    _Handler.status = 200


def test_ask_times_the_stream_and_recognizes_the_stand_in(server):
    sample = benchmark.ask(server, "repo", "q", timeout=10)
    assert sample.stand_in
    assert 0 <= sample.ttfb_ms <= sample.first_token_ms <= sample.total_ms
    assert sample.server_llm_ms == 4.0


def test_ask_with_a_real_model_answer(server):
    _Handler.answer = "Signer joins the value and an HMAC signature."
    assert not benchmark.ask(server, "repo", "q", timeout=10).stand_in


def test_rate_limit_stops_the_benchmark(server):
    _Handler.status = 429
    with pytest.raises(benchmark.RateLimitedError):
        benchmark.ask(server, "repo", "q", timeout=10)


def test_start_server_refuses_a_remote_url(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.start_server("https://example.com", tmp_path)
