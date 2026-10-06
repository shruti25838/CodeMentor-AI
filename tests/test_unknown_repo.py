from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.di import (
    get_agent_orchestrator,
    get_explain_service,
    get_llm_provider,
    get_repo_state_store,
)
from codeatlas.app.main import create_app
from codeatlas.controllers.repo_guard import REPO_NOT_FOUND
from codeatlas.services.agents.types import AnswerResult, GenerateResult
from codeatlas.utils.config import AppConfig

CONFIG = AppConfig(
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

KNOWN = "known-repo"


class FakeStateStore:
    def get(self, repo_id: str):
        return SimpleNamespace() if repo_id == KNOWN else None


class ModelSpy:
    """Stands in for everything that can reach the language model and records calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_chat_model(self):
        self.calls.append("get_chat_model")
        raise AssertionError("model requested")

    def handle_question(self, question, repo_id, **kwargs):
        self.calls.append("handle_question")
        return AnswerResult(answer="ok", citations=[], reasoning_steps=[])

    def handle_question_fast(self, question, repo_id, **kwargs):
        self.calls.append("handle_question_fast")
        return AnswerResult(answer="ok", citations=[], reasoning_steps=[])

    def handle_generation(self, prompt, repo_id):
        self.calls.append("handle_generation")
        return GenerateResult(diff="", notes=[], citations=[])

    def explain(self, repo_id, node_id):
        self.calls.append("explain")
        return SimpleNamespace(node_id=node_id, summary="ok", snippet="")


@pytest.fixture
def spy() -> ModelSpy:
    return ModelSpy()


@pytest.fixture
def client(spy: ModelSpy) -> TestClient:
    app = create_app(CONFIG)
    app.dependency_overrides[get_repo_state_store] = FakeStateStore
    app.dependency_overrides[get_agent_orchestrator] = lambda: spy
    app.dependency_overrides[get_llm_provider] = lambda: spy
    app.dependency_overrides[get_explain_service] = lambda: spy
    return TestClient(app)


REQUESTS = [
    ("/ask", {"question": "what does this do?"}),
    ("/ask/stream", {"question": "what does this do?"}),
    ("/explain", {"node_id": "src/app.py:1-5"}),
    ("/generate-code", {"prompt": "add a test"}),
]


@pytest.mark.parametrize("path,body", REQUESTS)
def test_unknown_repo_returns_404_without_calling_model(
    client: TestClient, spy: ModelSpy, path: str, body: dict
) -> None:
    response = client.post(path, json={**body, "repo_id": "does-not-exist"})
    assert response.status_code == 404
    assert response.json() == {"detail": REPO_NOT_FOUND}
    assert spy.calls == []


def test_unknown_repo_stream_is_not_opened(client: TestClient) -> None:
    response = client.post("/ask/stream", json={"question": "q", "repo_id": "does-not-exist"})
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path,body", REQUESTS)
def test_known_repo_reaches_model(client: TestClient, spy: ModelSpy, path: str, body: dict) -> None:
    response = client.post(path, json={**body, "repo_id": KNOWN})
    assert response.status_code == 200
    assert spy.calls
