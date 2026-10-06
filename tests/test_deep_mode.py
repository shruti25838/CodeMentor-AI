"""Deep analysis mode on /ask/stream: the five-agent pipeline, streamed step by step.

Fast mode stays the default, so a request that does not ask for deep mode must behave
exactly as before.
"""

import json
import logging

import pytest
from test_agent_pipeline import FakeAgent, plan_of
from test_analyze_indexing import Env

from codeatlas.app.di import get_agent_orchestrator, get_conversation_store, get_llm_provider
from codeatlas.schemas.ask import MAX_QUESTION_CHARS
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.memory.conversation import ConversationStore

SESSION = "s" * 32
MARKER = "zzmarkerzz"
PLAN = plan_of(("retrieval", "find it"), ("analyst", "structure"), ("mentor", "explain"))


def orchestrator(plan: str = PLAN) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=FakeAgent("planner", plan),
        retrieval_agent=FakeAgent("retrieval", "Answer: found\n\nCitations:\n- a.py (lines 1-2) | code"),
        analyst_agent=FakeAgent("analyst", "helper is called from 2 places"),
        mentor_agent=FakeAgent("mentor", "the final answer"),
        memory_agent=FakeAgent("memory"),
    )


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    e.client.app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator()
    e.client.app.dependency_overrides[get_llm_provider] = lambda: None
    e.client.app.dependency_overrides[get_conversation_store] = lambda: ConversationStore()
    e.repo_id = e.analyze().json()["repository_id"]
    return e


def events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def ask(env, **body):
    payload = {"question": "How does it work?", "repo_id": env.repo_id, **body}
    response = env.client.post("/ask/stream", json=payload)
    assert response.status_code == 200, response.text
    return events(response)


# ---------- deep mode streams each agent ----------


def test_deep_mode_streams_one_event_per_agent(env) -> None:
    agents = [e for e in ask(env, mode="deep") if e["type"] == "agent"]
    assert [e["name"] for e in agents] == ["planner", "retrieval", "analyst", "mentor"]


def test_each_agent_event_carries_a_short_structural_detail(env) -> None:
    agents = [e for e in ask(env, mode="deep") if e["type"] == "agent"]
    by_name = {e["name"]: e["detail"] for e in agents}
    assert "3 step(s)" in by_name["planner"]
    assert "snippet" in by_name["retrieval"]
    assert by_name["analyst"] and by_name["mentor"]


def test_agent_events_arrive_before_the_answer(env) -> None:
    stream = ask(env, mode="deep")
    first_token = next(i for i, e in enumerate(stream) if e["type"] == "token")
    last_agent = max(i for i, e in enumerate(stream) if e["type"] == "agent")
    assert last_agent < first_token


def test_deep_mode_streams_the_final_answer_then_done(env) -> None:
    stream = ask(env, mode="deep")
    answer = "".join(e["content"] for e in stream if e["type"] == "token")
    assert answer == "the final answer"
    assert stream[-1]["type"] == "done"


def test_deep_mode_reports_the_agents_that_ran(env) -> None:
    done = ask(env, mode="deep")[-1]
    assert done["agents_used"] == ["planner", "retrieval", "analyst", "mentor"]


def test_deep_mode_returns_citations(env) -> None:
    done = ask(env, mode="deep")[-1]
    assert done["citations"] == ["a.py (lines 1-2) | code"]


def test_deep_mode_times_the_first_agent_event(env) -> None:
    done = ask(env, mode="deep")[-1]
    assert "first_agent_event" in done["timings_ms"]


# ---------- fast stays the default ----------


def test_fast_is_the_default(env) -> None:
    stream = ask(env)
    assert [e for e in stream if e["type"] == "agent"] == []
    assert stream[-1]["agents_used"] == ["retrieval", "mentor"]


def test_fast_can_be_asked_for_explicitly(env) -> None:
    assert ask(env, mode="fast")[-1]["agents_used"] == ["retrieval", "mentor"]


def test_an_unknown_mode_is_rejected(env) -> None:
    response = env.client.post("/ask/stream", json={"question": "q", "repo_id": env.repo_id, "mode": "turbo"})
    assert response.status_code == 422


# ---------- the existing protections still apply ----------


def test_an_unknown_repo_is_404_before_any_agent_runs(env) -> None:
    response = env.client.post("/ask/stream", json={"question": "q", "repo_id": "a" * 36, "mode": "deep"})
    assert response.status_code == 404


def test_a_question_over_the_cap_is_rejected(env) -> None:
    response = env.client.post(
        "/ask/stream",
        json={"question": "x" * (MAX_QUESTION_CHARS + 1), "repo_id": env.repo_id, "mode": "deep"},
    )
    assert response.status_code == 422


def test_an_empty_question_is_rejected(env) -> None:
    assert env.client.post("/ask/stream", json={"question": "", "repo_id": env.repo_id}).status_code == 422


def test_a_question_at_the_cap_is_accepted(env) -> None:
    response = env.client.post("/ask/stream", json={"question": "x" * MAX_QUESTION_CHARS, "repo_id": env.repo_id})
    assert response.status_code == 200


def test_deep_mode_keeps_the_turn_in_conversation_memory(env) -> None:
    store = ConversationStore()
    env.client.app.dependency_overrides[get_conversation_store] = lambda: store
    ask(env, mode="deep", session_id=SESSION)

    history = store.history(SESSION, env.repo_id)
    assert len(history) == 1
    assert history[0].answer == "the final answer"


def test_deep_mode_without_a_session_keeps_nothing(env) -> None:
    store = ConversationStore()
    env.client.app.dependency_overrides[get_conversation_store] = lambda: store
    ask(env, mode="deep")
    assert store.session_count() == 0


# ---------- privacy ----------


def test_no_question_or_answer_text_reaches_the_logs(env, caplog) -> None:
    with caplog.at_level(logging.DEBUG):
        ask(env, mode="deep", question=f"What does {MARKER} do?", session_id=SESSION)

    assert MARKER not in caplog.text
    assert "the final answer" not in caplog.text


def test_agent_details_carry_no_visitor_text(env) -> None:
    stream = ask(env, mode="deep", question=f"What does {MARKER} do?")
    for event in stream:
        if event["type"] == "agent":
            assert MARKER not in event["detail"]


# ---------- the route list is unchanged ----------


def test_deep_mode_adds_no_new_route() -> None:
    """Deep mode is a field on /ask/stream, so tests/test_auth.py needs no new entry."""
    from test_auth import ADMIN, CONFIG, USER_FACING

    from codeatlas.app.main import create_app

    app = create_app(CONFIG)
    routes = {(m, r.path) for r in app.routes for m in getattr(r, "methods", ()) if m != "HEAD"}
    assert routes == USER_FACING | ADMIN
