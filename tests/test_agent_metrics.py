"""Prometheus metrics for the agent pipeline, with a fake model and no network.

The point of each test is either that a number is right, or that nothing resembling a
question, an answer or a key can reach a label.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from prometheus_client import REGISTRY
from test_agent_pipeline import FakeAgent, plan_of

from codeatlas.observability.agent_metrics import (
    AGENTS,
    FALLBACK_LOCATIONS,
    PAYLOAD_TOO_LARGE,
    RATE_LIMITED,
    REASONS,
    UNATTRIBUTED,
    AgentUsageCallback,
    agent_run,
    classify,
    current_agent,
)
from codeatlas.services.agents.orchestration import AgentOrchestrator

MARKER = "zzsecretzz"


def value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class TokenModel(BaseChatModel):
    """Answers instantly and reports token usage, like a real provider does."""

    state: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"error": None})()

    @property
    def _llm_type(self) -> str:
        return "token"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.state.error:
            raise self.state.error
        message = AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------- attribution ----------


def test_the_running_agent_is_known_inside_its_turn() -> None:
    assert current_agent() == UNATTRIBUTED
    with agent_run("retrieval"):
        assert current_agent() == "retrieval"
    assert current_agent() == UNATTRIBUTED


def test_an_unknown_agent_name_never_becomes_a_label() -> None:
    with agent_run("definitely-not-an-agent"):
        assert current_agent() == UNATTRIBUTED


def test_a_run_is_counted_and_timed() -> None:
    before = value("codeatlas_agent_runs_total", agent="analyst")
    with agent_run("analyst"):
        pass
    assert value("codeatlas_agent_runs_total", agent="analyst") == before + 1
    assert value("codeatlas_agent_duration_seconds_count", agent="analyst") >= 1


def test_a_raising_agent_is_recorded_as_a_failure_and_still_raises() -> None:
    before = value("codeatlas_agent_failures_total", agent="mentor", reason="model_error")
    with pytest.raises(RuntimeError), agent_run("mentor"):
        raise RuntimeError("boom")
    assert value("codeatlas_agent_failures_total", agent="mentor", reason="model_error") == before + 1


# ---------- model calls and tokens ----------


def test_model_calls_and_tokens_are_counted_against_the_running_agent() -> None:
    model = TokenModel().with_config({"callbacks": [AgentUsageCallback()]})
    calls = value("codeatlas_agent_model_calls_total", agent="mentor")
    inp = value("codeatlas_agent_tokens_total", agent="mentor", direction="input")
    out = value("codeatlas_agent_tokens_total", agent="mentor", direction="output")

    with agent_run("mentor"):
        model.invoke("hello")

    assert value("codeatlas_agent_model_calls_total", agent="mentor") == calls + 1
    assert value("codeatlas_agent_tokens_total", agent="mentor", direction="input") == inp + 11
    assert value("codeatlas_agent_tokens_total", agent="mentor", direction="output") == out + 7


def test_a_call_outside_any_agent_is_not_lost() -> None:
    model = TokenModel().with_config({"callbacks": [AgentUsageCallback()]})
    before = value("codeatlas_agent_model_calls_total", agent=UNATTRIBUTED)
    model.invoke("hello")
    assert value("codeatlas_agent_model_calls_total", agent=UNATTRIBUTED) == before + 1


# ---------- the provider's refusals ----------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Error code: 429 - rate_limit_exceeded", RATE_LIMITED),
        ("Error code: 413 - Request too large", PAYLOAD_TOO_LARGE),
        ("connection reset", "model_error"),
    ],
)
def test_provider_errors_map_to_a_reason_from_the_fixed_list(message: str, expected: str) -> None:
    assert classify(RuntimeError(message)) == expected
    assert classify(RuntimeError(message)) in REASONS


def test_a_413_records_both_the_reason_and_the_degraded_answer(tmp_path) -> None:
    from test_retrieval_snippets import RecordingModel, function_record, service

    small = tmp_path / "core.py"
    small.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    model = RecordingModel()
    model.state.error = RuntimeError("Error code: 413 - Request too large")

    too_large = value("codeatlas_agent_failures_total", agent="retrieval", reason=PAYLOAD_TOO_LARGE)
    fallback = value("codeatlas_agent_failures_total", agent="retrieval", reason=FALLBACK_LOCATIONS)

    with agent_run("retrieval"):
        result = service([function_record(small, "add", 1, 2)], llm=model).answer("r", "q")

    from codeatlas.services.llm.quota import MESSAGES, TOO_LARGE

    assert MESSAGES[TOO_LARGE] in result.answer
    assert value("codeatlas_agent_failures_total", agent="retrieval", reason=PAYLOAD_TOO_LARGE) == too_large + 1
    assert value("codeatlas_agent_failures_total", agent="retrieval", reason=FALLBACK_LOCATIONS) == fallback + 1


# ---------- the planner ----------


def build(plan: str) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=FakeAgent("planner", plan),
        retrieval_agent=FakeAgent("retrieval", "Answer: x"),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=FakeAgent("memory"),
    )


def test_a_model_plan_and_its_routes_are_counted() -> None:
    plans = value("codeatlas_planner_plans_total", source="model")
    routes = value("codeatlas_planner_routes_total", route="analyst")

    build(plan_of(("retrieval", "a"), ("analyst", "b"))).handle_question("q", "repo")

    assert value("codeatlas_planner_plans_total", source="model") == plans + 1
    assert value("codeatlas_planner_routes_total", route="analyst") == routes + 1


def test_a_plan_that_fell_back_is_counted_as_default() -> None:
    before = value("codeatlas_planner_plans_total", source="default")
    build("not json at all").handle_question("q", "repo")
    assert value("codeatlas_planner_plans_total", source="default") == before + 1


def test_every_agent_in_a_plan_is_counted_once_per_run() -> None:
    before = value("codeatlas_agent_runs_total", agent="memory")
    build(plan_of(("memory", "a"), ("memory", "b"))).handle_question("q", "repo")
    assert value("codeatlas_agent_runs_total", agent="memory") == before + 2


# ---------- labels stay small, fixed and free of content ----------


def test_the_label_sets_are_closed() -> None:
    assert set(AGENTS) == {"planner", "retrieval", "analyst", "mentor", "memory"}
    assert UNATTRIBUTED not in AGENTS
    assert len(REASONS) == 5


def test_no_question_or_answer_text_can_reach_a_label() -> None:
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan_of(("mentor", MARKER))),
        retrieval_agent=FakeAgent("retrieval", MARKER),
        analyst_agent=FakeAgent("analyst", MARKER),
        mentor_agent=FakeAgent("mentor", MARKER),
        memory_agent=FakeAgent("memory", MARKER),
    )
    orchestrator.handle_question(f"what is {MARKER}?", "repo")

    for metric in REGISTRY.collect():
        if not metric.name.startswith("codeatlas_agent") and not metric.name.startswith("codeatlas_planner"):
            continue
        for sample in metric.samples:
            for label_value in sample.labels.values():
                assert MARKER not in label_value, f"{sample.name} leaked question text"


def test_agent_metric_label_values_stay_within_the_fixed_lists() -> None:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name.startswith("codeatlas_agent_failures"):
                assert sample.labels["agent"] in (*AGENTS, UNATTRIBUTED)
                assert sample.labels["reason"] in REASONS
            elif sample.name.startswith("codeatlas_agent_tokens"):
                assert sample.labels["direction"] in ("input", "output")
            elif sample.name.startswith("codeatlas_planner_plans"):
                assert sample.labels["source"] in ("model", "default")


def test_the_metrics_endpoint_exposes_the_agent_series(tmp_path) -> None:
    from test_analyze_indexing import Env

    env = Env(tmp_path)
    with agent_run("retrieval"):
        pass
    body = env.client.get("/metrics", headers={"X-API-Key": "test-key"}).text
    assert "codeatlas_agent_runs_total" in body or "codeatlas_agent_runs_created" in body


# ---------- the dashboard ----------


def test_the_grafana_dashboard_queries_the_metrics_that_exist() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    dashboard = json.loads((root / "ops" / "grafana" / "dashboards" / "codeatlas-agents.json").read_text("utf-8"))
    expressions = " ".join(t["expr"] for p in dashboard["panels"] for t in p["targets"])

    for metric in (
        "codeatlas_agent_runs_total",
        "codeatlas_agent_duration_seconds_bucket",
        "codeatlas_agent_model_calls_total",
        "codeatlas_agent_tokens_total",
        "codeatlas_agent_failures_total",
        "codeatlas_planner_routes_total",
        "codeatlas_planner_plans_total",
    ):
        assert metric in expressions, f"{metric} has no panel"
