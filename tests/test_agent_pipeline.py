"""The five-agent graph: which agents exist, and what runs for a given plan.

Every agent here is a fake, so these tests need no model key and make no network call.
"""

import json

from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.agents.orchestration import SEARCH_PREFIX, AgentOrchestrator

FIVE_AGENTS = {"planner", "retrieval", "analyst", "mentor", "memory"}


class FakeAgent(Agent):
    """Records every call and answers with a fixed string."""

    def __init__(self, name: str, reply: str = "") -> None:
        self.name = name
        self.reply = reply or f"{name}-output"
        self.calls: list[str] = []

    def run(self, prompt: str, repo_id: str | None = None, **kwargs) -> str:
        self.calls.append(prompt)
        return self.reply


class FakePlanner(FakeAgent):
    def __init__(self, raw: str) -> None:
        super().__init__("planner", raw)


def build(raw_plan: str) -> tuple[AgentOrchestrator, dict[str, FakeAgent]]:
    agents = {
        "retrieval": FakeAgent("retrieval", "Answer: found it\n\nCitations:\n- a.py (lines 1-2) | code"),
        "analyst": FakeAgent("analyst"),
        "mentor": FakeAgent("mentor"),
        "memory": FakeAgent("memory"),
    }
    planner = FakePlanner(raw_plan)
    orchestrator = AgentOrchestrator(
        planner=planner,
        retrieval_agent=agents["retrieval"],
        analyst_agent=agents["analyst"],
        mentor_agent=agents["mentor"],
        memory_agent=agents["memory"],
    )
    agents["planner"] = planner
    return orchestrator, agents


def plan_of(*steps: tuple[str, str]) -> str:
    return json.dumps({"steps": [{"agent": a, "instruction": i} for a, i in steps]})


# ---------- the validator is gone ----------


def test_graph_has_no_validator_node() -> None:
    orchestrator, _ = build(plan_of(("retrieval", "find x")))
    nodes = set(orchestrator._graph.get_graph().nodes)
    assert "validator" not in nodes
    assert nodes >= FIVE_AGENTS, f"missing agent nodes: {FIVE_AGENTS - nodes}"


def test_orchestrator_has_no_validator_attribute() -> None:
    orchestrator, _ = build(plan_of(("retrieval", "find x")))
    assert not hasattr(orchestrator, "_validator_node")


def test_plan_ends_after_its_last_step_with_no_extra_agent_call() -> None:
    """Before, finishing the steps routed to a validator, costing one more model call."""
    orchestrator, agents = build(plan_of(("retrieval", "find x")))
    result = orchestrator.handle_question("q", "repo")

    # Retrieval searches with the visitor's question, not the planner's wording for the step.
    assert agents["retrieval"].calls == ["q"]
    # The mentor was the validator's stand-in; with the validator gone it must not be called.
    assert agents["mentor"].calls == []
    assert agents["analyst"].calls == []
    # No mentor in the plan, so the retrieval output stands in, labelled a search result
    # rather than passed off as a written answer.
    assert agents["retrieval"].reply in result.answer
    assert result.answer.startswith(SEARCH_PREFIX)


def test_mentor_answer_is_not_rewritten_after_the_plan_finishes() -> None:
    orchestrator, agents = build(plan_of(("retrieval", "find x"), ("mentor", "explain")))
    result = orchestrator.handle_question("q", "repo")

    assert agents["mentor"].calls, "the mentor step should still run"
    assert len(agents["mentor"].calls) == 1, "no second, validating call"
    assert result.answer == "mentor-output"
