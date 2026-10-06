"""What is recorded as having run is what really ran.

The /ask analytics used to record a fixed list. Measured against itsdangerous with a real
model, it recorded ['planner', 'retrieval', 'mentor', 'validator'] while the agents that
actually ran were ['planner', 'retrieval', 'mentor', 'mentor', 'analyst', 'mentor']: the
analyst was invisible, the mentor's three calls counted once, and a validator was credited
although no validator agent exists.
"""

import json

from test_agent_pipeline import FakeAgent, plan_of

from codeatlas.services.agents.orchestration import AgentOrchestrator


def build(raw_plan: str) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=FakeAgent("planner", raw_plan),
        retrieval_agent=FakeAgent("retrieval", "Answer: x\n\nCitations:\n- a.py (lines 1-2) | code"),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=FakeAgent("memory"),
    )


def test_the_recorded_list_matches_the_plan_that_ran() -> None:
    plan = plan_of(("retrieval", "find"), ("mentor", "explain"), ("mentor", "again"), ("analyst", "summarise"))
    result = build(plan).handle_question("q", "repo")

    assert result.agents_used == ["planner", "retrieval", "mentor", "mentor", "analyst"]


def test_repeats_are_kept_not_collapsed() -> None:
    result = build(plan_of(("mentor", "a"), ("mentor", "b"), ("mentor", "c"))).handle_question("q", "repo")
    assert result.agents_used == ["planner", "mentor", "mentor", "mentor"]


def test_the_analyst_is_recorded_when_it_runs() -> None:
    result = build(plan_of(("analyst", "structure"))).handle_question("q", "repo")
    assert "analyst" in result.agents_used


def test_no_validator_is_ever_recorded() -> None:
    result = build(plan_of(("retrieval", "find"), ("mentor", "explain"))).handle_question("q", "repo")
    assert "validator" not in result.agents_used


def test_an_agent_that_did_not_run_is_not_recorded() -> None:
    result = build(plan_of(("retrieval", "find"))).handle_question("q", "repo")
    assert result.agents_used == ["planner", "retrieval"]
    assert "memory" not in result.agents_used
    assert "analyst" not in result.agents_used


def test_a_bad_plan_still_records_the_planner_and_the_default_steps() -> None:
    result = build("not json").handle_question("q", "repo")
    assert result.agents_used == ["planner", "retrieval", "mentor"]


def test_the_fast_path_records_its_two_agents() -> None:
    result = build("{}").handle_question_fast("q", "repo")
    assert result.agents_used == ["retrieval", "mentor"]
    assert "planner" not in result.agents_used


def test_the_reasoning_steps_say_what_ran() -> None:
    result = build(plan_of(("retrieval", "find"), ("analyst", "summarise"))).handle_question("q", "repo")
    assert any("Agents that ran: planner -> retrieval -> analyst." in s for s in result.reasoning_steps)


def test_a_failing_agent_is_still_recorded_as_having_run() -> None:
    class Boom(FakeAgent):
        def run(self, prompt, repo_id=None, **kwargs):
            raise RuntimeError("down")

    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan_of(("analyst", "x"))),
        retrieval_agent=FakeAgent("retrieval"),
        analyst_agent=Boom("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=FakeAgent("memory"),
    )
    result = orchestrator.handle_question("q", "repo")
    assert result.agents_used == ["planner", "analyst"]


# ---------- through the endpoints ----------


def _chat_env(tmp_path):
    from test_analyze_indexing import Env

    return Env(tmp_path)


def test_ask_response_and_analytics_agree(tmp_path, monkeypatch) -> None:
    from codeatlas.app.di import get_agent_orchestrator, get_llm_provider
    from codeatlas.observability.tracker import tracker

    env = _chat_env(tmp_path)
    repo_id = env.analyze().json()["repository_id"]
    orchestrator = build(plan_of(("retrieval", "find"), ("analyst", "summarise")))
    env.client.app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator
    env.client.app.dependency_overrides[get_llm_provider] = lambda: None

    before = len(tracker._records)
    body = env.client.post("/ask", json={"question": "q", "repo_id": repo_id}).json()

    assert body["agents_used"] == ["planner", "retrieval", "analyst"]
    assert tracker._records[before].agents_used == body["agents_used"]


def test_stream_done_event_reports_the_agents(tmp_path) -> None:
    from codeatlas.app.di import get_agent_orchestrator, get_llm_provider

    env = _chat_env(tmp_path)
    repo_id = env.analyze().json()["repository_id"]
    env.client.app.dependency_overrides[get_agent_orchestrator] = lambda: build("{}")
    env.client.app.dependency_overrides[get_llm_provider] = lambda: None

    resp = env.client.post("/ask/stream", json={"question": "q", "repo_id": repo_id})
    done = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")][-1]

    assert done["type"] == "done"
    assert done["agents_used"] == ["retrieval", "mentor"]
