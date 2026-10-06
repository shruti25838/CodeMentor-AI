"""Prometheus metrics for the agent pipeline.

Nothing here costs a model call: every number is taken from work the pipeline was doing
anyway. Labels are a small, fixed set — the agent's name, a reason from a closed list, and
"input"/"output" for tokens. A question, an answer, a file path or a key must never become a
label value: Prometheus keeps one time series per distinct label combination, so free text
would both leak content and grow without limit.

Which agent a model call belongs to is tracked with a context variable set around each
agent's turn, so one callback on the chat model can attribute calls and tokens correctly.
"""

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Histogram

# The only agent names that may appear as a label.
AGENTS = ("planner", "retrieval", "analyst", "mentor", "memory")
UNATTRIBUTED = "none"

# The only failure reasons that may appear as a label.
RATE_LIMITED = "rate_limited"
PAYLOAD_TOO_LARGE = "payload_too_large"
MODEL_ERROR = "model_error"
AGENT_ERROR = "agent_error"
FALLBACK_LOCATIONS = "fallback_to_location_list"
REASONS = (RATE_LIMITED, PAYLOAD_TOO_LARGE, MODEL_ERROR, AGENT_ERROR, FALLBACK_LOCATIONS)

AGENT_RUNS = Counter(
    "codeatlas_agent_runs_total",
    "Times each agent ran, taken from the graph as it executed",
    ["agent"],
)

AGENT_DURATION = Histogram(
    "codeatlas_agent_duration_seconds",
    "How long each agent's turn took",
    ["agent"],
)

AGENT_MODEL_CALLS = Counter(
    "codeatlas_agent_model_calls_total",
    "Model calls made while an agent was running",
    ["agent"],
)

AGENT_TOKENS = Counter(
    "codeatlas_agent_tokens_total",
    "Tokens used by model calls, by agent and direction",
    ["agent", "direction"],
)

AGENT_FAILURES = Counter(
    "codeatlas_agent_failures_total",
    "Agent failures by reason, including provider rejections and degraded answers",
    ["agent", "reason"],
)

PLANNER_ROUTES = Counter(
    "codeatlas_planner_routes_total",
    "Which agent the planner routed a step to, and whether the plan came from the model",
    ["route"],
)

PLANNER_PLANS = Counter(
    "codeatlas_planner_plans_total",
    "Plans by source: the model's own plan, or the safe default",
    ["source"],
)

_current_agent: contextvars.ContextVar[str] = contextvars.ContextVar("current_agent", default=UNATTRIBUTED)


def current_agent() -> str:
    return _current_agent.get()


@contextmanager
def agent_run(name: str) -> Iterator[None]:
    """Count and time one agent's turn, and attribute its model calls to it."""
    agent = name if name in AGENTS else UNATTRIBUTED
    token = _current_agent.set(agent)
    AGENT_RUNS.labels(agent=agent).inc()
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        record_failure(agent, classify(exc))
        raise
    finally:
        AGENT_DURATION.labels(agent=agent).observe(time.perf_counter() - started)
        _current_agent.reset(token)


def record_failure(agent: str, reason: str) -> None:
    AGENT_FAILURES.labels(
        agent=agent if agent in AGENTS else UNATTRIBUTED,
        reason=reason if reason in REASONS else AGENT_ERROR,
    ).inc()


def classify(exc: BaseException) -> str:
    """A reason from the fixed list. The provider reports these as HTTP status codes."""
    text = str(exc)
    status = getattr(exc, "status_code", None)
    if status == 429 or "429" in text or "rate_limit" in text:
        return RATE_LIMITED
    if status == 413 or "413" in text:
        return PAYLOAD_TOO_LARGE
    return MODEL_ERROR


def record_plan(source: str, routes: list[str]) -> None:
    """One plan: where it came from, and which agents it routes to."""
    PLANNER_PLANS.labels(source="model" if source == "model" else "default").inc()
    for route in routes:
        PLANNER_ROUTES.labels(route=route if route in AGENTS else UNATTRIBUTED).inc()


class AgentUsageCallback:
    """Counts model calls and tokens against whichever agent is running.

    Implemented as a plain object with the two hooks langchain calls, so the metrics do not
    depend on langchain's callback base class staying put.
    """

    raise_error = False
    run_inline = False
    ignore_llm = False
    ignore_chain = True
    ignore_agent = True
    ignore_retriever = True
    ignore_chat_model = False
    ignore_custom_event = True

    def on_llm_end(self, response, **kwargs) -> None:
        agent = current_agent()
        AGENT_MODEL_CALLS.labels(agent=agent).inc()
        prompt_tokens, completion_tokens = _tokens_from(response)
        if prompt_tokens:
            AGENT_TOKENS.labels(agent=agent, direction="input").inc(prompt_tokens)
        if completion_tokens:
            AGENT_TOKENS.labels(agent=agent, direction="output").inc(completion_tokens)

    def on_llm_error(self, error, **kwargs) -> None:
        record_failure(current_agent(), classify(error))

    # langchain calls these; they are deliberately empty.
    def __getattr__(self, name: str):
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def _tokens_from(response) -> tuple[int, int]:
    usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
    if usage:
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    prompt = completion = 0
    for generations in getattr(response, "generations", []) or []:
        for generation in generations:
            meta = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
            prompt += meta.get("input_tokens", 0)
            completion += meta.get("output_tokens", 0)
    return prompt, completion
