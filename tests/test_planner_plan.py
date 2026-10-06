"""Plan parsing: every shape a planner model really produces, and the safe default.

Four of these shapes used to raise inside the orchestrator's routing function, which
turned the whole /ask request into a 500. None of them may raise now.
"""

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_agent_pipeline import FakeAgent

from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.agents.plan import (
    DEFAULT_AGENTS,
    KNOWN_AGENTS,
    MAX_STEPS,
    Plan,
    default_plan,
    parse_plan,
)
from codeatlas.services.agents.planner_agent import PlannerAgent

QUESTION = "How does Signer verify a signature?"


class ScriptedModel(BaseChatModel):
    """Returns a fixed string, or raises if it is an exception."""

    reply: object = ""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(self.reply, Exception):
            raise self.reply
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


# ---------- shapes that must produce the default plan, never an exception ----------

BAD_OUTPUTS = [
    ("prose", "Sure! First I would search the code, then explain it."),
    ("empty", ""),
    ("whitespace", "   \n  "),
    ("steps is a string", '{"steps": "retrieval"}'),
    ("steps is a dict", '{"steps": {"agent": "retrieval"}}'),
    ("steps is a number", '{"steps": 3}'),
    ("steps missing", '{"plan": []}'),
    ("empty object", "{}"),
    ("empty steps list", '{"steps": []}'),
    ("top-level list", '[{"agent": "retrieval", "instruction": "x"}]'),
    ("top-level string", '"retrieval"'),
    ("top-level number", "42"),
    ("null", "null"),
    ("steps of strings", '{"steps": ["retrieval", "mentor"]}'),
    ("steps of numbers", '{"steps": [1, 2]}'),
    ("unknown agents only", '{"steps": [{"agent": "validator", "instruction": "check"}]}'),
    ("agent is not a string", '{"steps": [{"agent": 7, "instruction": "x"}]}'),
    ("agent missing", '{"steps": [{"instruction": "x"}]}'),
    ("truncated json", '{"steps": [{"agent": "retrieval",'),
]


@pytest.mark.parametrize("label,raw", BAD_OUTPUTS, ids=[label for label, _ in BAD_OUTPUTS])
def test_bad_planner_output_gives_the_default_plan(label: str, raw: str) -> None:
    plan = parse_plan(raw, QUESTION)
    assert plan.agents == list(DEFAULT_AGENTS)
    assert all(step.instruction == QUESTION for step in plan.steps)
    assert plan.note, "a fallback must say why"


@pytest.mark.parametrize("label,raw", BAD_OUTPUTS, ids=[label for label, _ in BAD_OUTPUTS])
def test_bad_planner_output_never_raises_while_routing(label: str, raw: str) -> None:
    """The regression: _route_step used to raise AttributeError/KeyError on several of these."""
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", raw),
        retrieval_agent=FakeAgent("retrieval"),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=FakeAgent("memory"),
    )
    result = orchestrator.handle_question(QUESTION, "repo")
    assert result.answer, "a usable answer even from an unusable plan"


def test_planner_that_raises_gives_the_default_plan() -> None:
    class Boom(FakeAgent):
        def run(self, prompt, repo_id=None, **kwargs):
            raise RuntimeError("model down")

    orchestrator = AgentOrchestrator(
        planner=Boom("planner"),
        retrieval_agent=FakeAgent("retrieval"),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=FakeAgent("memory"),
    )
    result = orchestrator.handle_question(QUESTION, "repo")
    assert result.answer
    assert any("default plan" in step for step in result.reasoning_steps)


# ---------- shapes that must be accepted ----------


def test_plain_json_is_used_as_given() -> None:
    raw = json.dumps({"steps": [{"agent": "retrieval", "instruction": "find Signer"}]})
    plan = parse_plan(raw, QUESTION)
    assert plan.agents == ["retrieval"]
    assert plan.steps[0].instruction == "find Signer"
    assert plan.note == ""


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"steps": [{"agent": "mentor", "instruction": "explain"}]}\n```',
        '```JSON\n{"steps": [{"agent": "mentor", "instruction": "explain"}]}\n```',
        '```\n{"steps": [{"agent": "mentor", "instruction": "explain"}]}\n```',
        'Here is the plan:\n{"steps": [{"agent": "mentor", "instruction": "explain"}]}\nHope that helps!',
    ],
)
def test_fenced_or_wrapped_json_is_recovered(raw: str) -> None:
    """A fenced plan used to be thrown away silently and replaced by the fallback."""
    plan = parse_plan(raw, QUESTION)
    assert plan.agents == ["mentor"]
    assert plan.steps[0].instruction == "explain"


def test_agent_name_is_trimmed_and_lowercased() -> None:
    plan = parse_plan('{"steps": [{"agent": "  Retrieval ", "instruction": "x"}]}', QUESTION)
    assert plan.agents == ["retrieval"]


def test_step_without_an_instruction_falls_back_to_the_question() -> None:
    """It used to hand the agent an empty prompt."""
    plan = parse_plan('{"steps": [{"agent": "retrieval"}]}', QUESTION)
    assert plan.steps[0].instruction == QUESTION


def test_unusable_steps_are_dropped_and_the_rest_kept() -> None:
    raw = json.dumps(
        {
            "steps": [
                {"agent": "retrieval", "instruction": "find it"},
                {"agent": "nonsense", "instruction": "ignored"},
                "not a step",
                {"agent": "mentor", "instruction": "explain it"},
            ]
        }
    )
    plan = parse_plan(raw, QUESTION)
    assert plan.agents == ["retrieval", "mentor"]
    assert "Dropped 2" in plan.note


def test_too_many_steps_are_capped() -> None:
    raw = json.dumps({"steps": [{"agent": "mentor", "instruction": f"s{i}"} for i in range(20)]})
    plan = parse_plan(raw, QUESTION)
    assert len(plan.steps) == MAX_STEPS
    assert f"kept the first {MAX_STEPS}" in plan.note


def test_planner_is_not_a_step_a_plan_can_ask_for() -> None:
    assert "planner" not in KNOWN_AGENTS
    plan = parse_plan('{"steps": [{"agent": "planner", "instruction": "plan again"}]}', QUESTION)
    assert plan.agents == list(DEFAULT_AGENTS)


# ---------- the agent itself ----------


def test_planner_agent_without_a_model_returns_the_default_plan() -> None:
    plan = PlannerAgent(llm=None).plan(QUESTION)
    assert plan.agents == list(DEFAULT_AGENTS)
    assert "No planner model" in plan.note


def test_planner_agent_run_always_returns_parseable_json() -> None:
    model = ScriptedModel()
    model.reply = "I will not give you JSON."
    raw = PlannerAgent(llm=model).run(QUESTION)
    reparsed = parse_plan(raw, QUESTION)
    assert reparsed.agents == list(DEFAULT_AGENTS)
    assert reparsed.note == "", "run() output is already valid, so re-parsing finds nothing wrong"


def test_planner_agent_survives_a_failing_model() -> None:
    model = ScriptedModel()
    model.reply = RuntimeError("groq is down")
    plan = PlannerAgent(llm=model).plan(QUESTION)
    assert plan.agents == list(DEFAULT_AGENTS)
    assert "failed" in plan.note


def test_planner_agent_uses_a_good_plan_from_the_model() -> None:
    model = ScriptedModel()
    model.reply = '{"steps": [{"agent": "analyst", "instruction": "who calls Signer.sign"}]}'
    plan = PlannerAgent(llm=model).plan(QUESTION)
    assert plan.agents == ["analyst"]
    assert plan.steps[0].instruction == "who calls Signer.sign"


def test_default_plan_is_retrieval_then_mentor() -> None:
    assert default_plan(QUESTION).agents == ["retrieval", "mentor"]


def test_plan_as_json_round_trips() -> None:
    plan = Plan(steps=parse_plan('{"steps": [{"agent": "memory", "instruction": "recall"}]}', QUESTION).steps)
    assert json.loads(plan.as_json()) == {"steps": [{"agent": "memory", "instruction": "recall"}]}
