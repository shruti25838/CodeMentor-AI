import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.di import get_agent_orchestrator, get_explain_service, get_llm_provider
from codeatlas.app.main import create_app
from codeatlas.utils.config import AppConfig

KEY = "test-admin-key"

CONFIG = AppConfig(
    embedding_provider="hash",
    embedding_model="",
    index_dir=".codeatlas/indexes",
    state_dir=".codeatlas/state",
    llm_provider="groq",
    llm_model="",
    llm_temperature=0.2,
    api_key=KEY,
    auth_enabled=True,
)

# Everything the website (codementor-ui/lib/api.ts) calls.
USER_FACING = {
    ("POST", "/ask"),
    ("POST", "/ask/stream"),
    ("POST", "/analyze-repo"),
    ("POST", "/files"),
    ("POST", "/files/content"),
    ("POST", "/repo-overview"),
    ("GET", "/repos"),
    ("POST", "/dependencies/graph"),
    ("GET", "/eval/stats"),
}

ADMIN = {
    ("POST", "/dependencies"),
    ("POST", "/explain"),
    ("POST", "/search"),
    ("POST", "/generate-code"),
    ("GET", "/metrics"),
    ("GET", "/docs"),
    ("GET", "/docs/oauth2-redirect"),
    ("GET", "/redoc"),
    ("GET", "/openapi.json"),
}

API_TS = Path(__file__).resolve().parent.parent / "codementor-ui" / "lib" / "api.ts"


def _client(config: AppConfig = CONFIG) -> TestClient:
    app = create_app(config)
    # No LLM in tests: these would otherwise be built before the empty body is rejected.
    app.dependency_overrides[get_agent_orchestrator] = lambda: None
    app.dependency_overrides[get_llm_provider] = lambda: None
    app.dependency_overrides[get_explain_service] = lambda: None
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _client()


def _call(client: TestClient, method: str, path: str, headers: dict | None = None):
    # POSTs send an empty body: a 422 shows the request got past any auth check
    # without running the handler (no clone, no LLM call).
    if method == "POST":
        return client.post(path, json={}, headers=headers)
    return client.get(path, headers=headers)


def test_every_route_is_classified() -> None:
    app = create_app(CONFIG)
    routes = {(m, r.path) for r in app.routes for m in getattr(r, "methods", ()) if m != "HEAD"}
    assert routes == USER_FACING | ADMIN


def test_website_only_calls_user_facing_endpoints() -> None:
    paths = set(re.findall(r"\$\{BASE_URL\}(/[\w\-/]*)", API_TS.read_text(encoding="utf-8")))
    assert paths == {p for _, p in USER_FACING}


def test_website_sends_no_api_key() -> None:
    source = API_TS.read_text(encoding="utf-8").lower()
    assert "x-api-key" not in source
    assert "api_key" not in source


@pytest.mark.parametrize("method,path", sorted(USER_FACING))
def test_user_facing_needs_no_key(client: TestClient, method: str, path: str) -> None:
    response = _call(client, method, path)
    assert response.status_code not in (401, 403, 503), response.text


@pytest.mark.parametrize("method,path", sorted(ADMIN))
def test_admin_rejects_missing_key(client: TestClient, method: str, path: str) -> None:
    assert _call(client, method, path).status_code == 401


@pytest.mark.parametrize("method,path", sorted(ADMIN))
def test_admin_rejects_wrong_key(client: TestClient, method: str, path: str) -> None:
    assert _call(client, method, path, headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("method,path", sorted(ADMIN))
def test_admin_accepts_correct_key(client: TestClient, method: str, path: str) -> None:
    response = _call(client, method, path, headers={"X-API-Key": KEY})
    assert response.status_code not in (401, 403, 503), response.text


def test_admin_fails_closed_without_configured_key() -> None:
    client = _client(replace(CONFIG, api_key=None))
    assert client.get("/metrics").status_code == 503
    assert client.get("/metrics", headers={"X-API-Key": ""}).status_code == 503
    assert client.get("/repos").status_code == 200


def test_openapi_schema_with_key_lists_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json", headers={"X-API-Key": KEY}).json()
    assert "/ask" in schema["paths"]


def test_auth_enabled_by_default(monkeypatch) -> None:
    from codeatlas.utils.config import load_config

    monkeypatch.delenv("CODEATLAS_AUTH_ENABLED", raising=False)
    assert load_config().auth_enabled is True
