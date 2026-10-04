import logging
import re
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from codeatlas.app.di import (
    get_agent_orchestrator,
    get_explain_service,
    get_llm_provider,
    get_repository_loader,
)
from codeatlas.app.main import create_app
from codeatlas.app.rate_limit import (
    ConcurrencyLimit,
    RateLimitExceeded,
    RequestLimit,
    SlidingWindowLimiter,
    client_address,
)
from codeatlas.services.ingestion.git_loader import GitRepositoryLoader
from codeatlas.utils.config import AppConfig

BASE_CONFIG = AppConfig(
    embedding_provider="hash",
    embedding_model="",
    index_dir=".codeatlas/indexes",
    state_dir=".codeatlas/state",
    llm_provider="groq",
    llm_model="",
    llm_temperature=0.2,
    api_key=None,
    auth_enabled=False,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _request(peer: str = "10.0.0.1", xff: list[str] | None = None) -> Request:
    headers = [(b"x-forwarded-for", v.encode()) for v in (xff or [])]
    return Request({"type": "http", "client": (peer, 1234), "headers": headers})


# ---------- client address ----------


def test_zero_hops_ignores_forwarded_header():
    assert client_address(_request("10.0.0.1", ["1.2.3.4"]), 0) == "10.0.0.1"


def test_one_hop_takes_rightmost_entry_not_spoofed_leftmost():
    # The caller sent "6.6.6.6"; the trusted proxy appended the real address 1.2.3.4.
    assert client_address(_request("10.0.0.1", ["6.6.6.6, 1.2.3.4"]), 1) == "1.2.3.4"


def test_two_hops_takes_second_from_right():
    assert client_address(_request("10.0.0.1", ["6.6.6.6, 1.2.3.4, 10.0.0.9"]), 2) == "1.2.3.4"


def test_multiple_forwarded_headers_are_joined():
    assert client_address(_request("10.0.0.1", ["6.6.6.6", "1.2.3.4"]), 1) == "1.2.3.4"


@pytest.mark.parametrize("xff", [None, [""], ["1.2.3.4"]])
def test_too_few_entries_falls_back_to_peer(xff):
    assert client_address(_request("10.0.0.1", xff), 2) == "10.0.0.1"


def test_non_ip_entry_falls_back_to_peer():
    assert client_address(_request("10.0.0.1", ["not-an-ip"]), 1) == "10.0.0.1"


# ---------- sliding window ----------


def test_window_blocks_over_limit_then_expires():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60, clock=clock)
    limiter.record("a", clock.now)
    clock.now += 10
    limiter.record("a", clock.now)
    assert limiter.retry_after("a", clock.now) == pytest.approx(50)
    assert limiter.retry_after("b", clock.now) == 0
    clock.now += 50
    assert limiter.retry_after("a", clock.now) == 0


def test_idle_clients_are_swept():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, clock=clock)
    for i in range(100):
        limiter.record(f"c{i}", clock.now)
    assert len(limiter) == 100
    clock.now += 61
    limiter.record("new", clock.now)
    assert len(limiter) == 1


def test_tracked_clients_are_capped_evicting_least_recent():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(limit=5, window_seconds=60, max_keys=3, clock=clock)
    for key in ["a", "b", "c"]:
        limiter.record(key, clock.now)
    limiter.record("a", clock.now)  # "a" is now most recent; "b" is least recent
    limiter.record("d", clock.now)
    assert len(limiter) == 3
    assert "b" not in limiter._events
    assert "a" in limiter._events


# ---------- per-client + global ----------


def test_per_client_limit_does_not_affect_other_clients():
    limit = RequestLimit("chat", per_client=2, global_limit=100, window_seconds=60, max_clients=100, clock=FakeClock())
    limit.check("a")
    limit.check("a")
    with pytest.raises(RateLimitExceeded):
        limit.check("a")
    limit.check("b")


def test_global_limit_caps_all_clients_together():
    limit = RequestLimit("chat", per_client=100, global_limit=3, window_seconds=60, max_clients=100, clock=FakeClock())
    for client in ["a", "b", "c"]:
        limit.check(client)
    with pytest.raises(RateLimitExceeded) as exc:
        limit.check("d")
    assert "server is handling too many" in exc.value.detail


def test_rejected_requests_are_not_counted():
    clock = FakeClock()
    limit = RequestLimit("chat", per_client=1, global_limit=100, window_seconds=60, max_clients=100, clock=clock)
    limit.check("a")
    for _ in range(10):
        with pytest.raises(RateLimitExceeded):
            limit.check("a")
    clock.now += 60
    limit.check("a")  # rejections above didn't push the window forward


def test_429_has_friendly_message_and_retry_after():
    exc = RateLimitExceeded("You've sent too many chat requests.", 12.3)
    assert exc.status_code == 429
    assert exc.headers == {"Retry-After": "13"}
    assert exc.detail == "You've sent too many chat requests. Please try again in 13 seconds."


# ---------- concurrency ----------


def test_concurrency_limit_rejects_when_full_and_frees_on_release():
    limit = ConcurrencyLimit(2)
    limit.acquire()
    limit.acquire()
    with pytest.raises(RateLimitExceeded) as exc:
        limit.acquire()
    assert exc.value.status_code == 429
    limit.release()
    limit.acquire()


# ---------- endpoints ----------


def _client(tmp_path, **overrides) -> TestClient:
    app = create_app(replace(BASE_CONFIG, **overrides))
    app.dependency_overrides[get_repository_loader] = lambda: GitRepositoryLoader(base_dir=str(tmp_path))
    app.dependency_overrides[get_agent_orchestrator] = lambda: None
    app.dependency_overrides[get_llm_provider] = lambda: None
    app.dependency_overrides[get_explain_service] = lambda: None
    return TestClient(app)


def test_chat_endpoints_return_429_with_retry_after(tmp_path):
    client = _client(tmp_path, llm_per_client_per_minute=2)
    # Empty bodies fail validation (422) after the limiter has counted them.
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/explain", json={}).status_code == 422
    resp = client.post("/ask/stream", json={})
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) > 0
    assert resp.json()["detail"].startswith("You've sent too many chat requests.")


def test_clone_endpoint_limited_per_client(tmp_path):
    client = _client(tmp_path, clone_per_client_per_minute=1)
    assert client.post("/analyze-repo", json={"repo_url": "not a url"}).status_code == 400
    resp = client.post("/analyze-repo", json={"repo_url": "not a url"})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_clone_slot_is_released_after_each_request(tmp_path):
    client = _client(tmp_path, max_concurrent_clones=1)
    for _ in range(3):
        assert client.post("/analyze-repo", json={"repo_url": "not a url"}).status_code == 400


def test_browsing_endpoints_are_not_limited(tmp_path):
    client = _client(tmp_path, llm_per_client_per_minute=1, llm_global_per_minute=1, clone_global_per_minute=1)
    for _ in range(5):
        assert client.get("/repos").status_code == 200


def test_limits_use_forwarded_address_when_hops_set(tmp_path):
    client = _client(tmp_path, trusted_proxy_hops=1, llm_per_client_per_minute=1)
    assert client.post("/ask", json={}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 422
    # Different visitor behind the same proxy: separate allowance.
    assert client.post("/ask", json={}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 422
    assert client.post("/ask", json={}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429


def test_spoofed_header_ignored_when_hops_zero(tmp_path):
    client = _client(tmp_path, llm_per_client_per_minute=1)
    assert client.post("/ask", json={}, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 422
    assert client.post("/ask", json={}, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429


def test_cors_exposes_retry_after(tmp_path):
    client = _client(tmp_path, llm_per_client_per_minute=0)
    resp = client.post("/ask", json={}, headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 429
    assert "retry-after" in resp.headers["access-control-expose-headers"].lower()


@pytest.mark.parametrize("hops, expected", [(0, "direct connection address"), (2, "entry 2 from the right")])
def test_startup_logs_mode_without_addresses(caplog, hops, expected):
    with caplog.at_level(logging.INFO, logger="codeatlas.app.rate_limit"):
        create_app(replace(BASE_CONFIG, trusted_proxy_hops=hops))
    lines = [r.getMessage() for r in caplog.records if r.name == "codeatlas.app.rate_limit"]
    assert len(lines) == 1
    assert expected in lines[0]
    assert not re.search(r"\d+\.\d+\.\d+\.\d+|::", lines[0])
