"""The plan the orchestrator runs, built from whatever the planner model returned.

A language model asked for JSON returns prose, fenced JSON, a list where an object was
asked for, or an object whose "steps" is a string. Before this module each of those either
crashed the request inside the routing function or silently produced an empty answer, so
parsing is deliberately total: `parse_plan` never raises and always returns a runnable plan.

Anything it cannot use is dropped and reported in `Plan.note`, which the orchestrator puts
in the answer's reasoning steps, so a bad plan is visible rather than silent.
"""

import json
from dataclasses import dataclass

# The agents the dispatcher can route to. "planner" is not here: it builds the plan, it is
# not a step in it, and a plan that asked for it would loop.
KNOWN_AGENTS = ("retrieval", "analyst", "mentor", "memory")

# A model that misunderstands the format can ask for dozens of steps, and every step is at
# least one model call. Extra steps are dropped rather than run.
MAX_STEPS = 6

# Used when the model's output cannot be used at all. Retrieval finds the code, the mentor
# answers from it: the same two steps the fast path runs.
DEFAULT_AGENTS = ("retrieval", "mentor")


@dataclass(frozen=True)
class PlanStep:
    agent: str
    instruction: str


@dataclass(frozen=True)
class Plan:
    steps: tuple[PlanStep, ...]
    # Empty when the model's plan was used as given; otherwise says what was wrong.
    note: str = ""

    @property
    def agents(self) -> list[str]:
        return [step.agent for step in self.steps]

    def as_json(self) -> str:
        return json.dumps({"steps": [{"agent": s.agent, "instruction": s.instruction} for s in self.steps]})


def default_plan(question: str, note: str = "") -> Plan:
    return Plan(steps=tuple(PlanStep(agent, question) for agent in DEFAULT_AGENTS), note=note)


def parse_plan(raw: str, question: str) -> Plan:
    """A runnable plan for any input. Never raises."""
    payload, error = _load_json(raw)
    if error:
        return default_plan(question, f"Planner output was not usable ({error}); used the default plan.")

    if not isinstance(payload, dict):
        kind = type(payload).__name__
        return default_plan(question, f"Planner returned a {kind}, not an object; used the default plan.")

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        kind = "nothing" if raw_steps is None else type(raw_steps).__name__
        return default_plan(question, f"Planner's 'steps' was {kind}, not a list; used the default plan.")

    steps: list[PlanStep] = []
    dropped: list[str] = []
    for item in raw_steps:
        step, why = _parse_step(item, question)
        if step is None:
            dropped.append(why)
            continue
        steps.append(step)

    if not steps:
        detail = f" ({dropped[0]})" if dropped else ""
        return default_plan(question, f"No usable step in the planner's plan{detail}; used the default plan.")

    notes: list[str] = []
    if dropped:
        notes.append(f"Dropped {len(dropped)} unusable step(s): {'; '.join(sorted(set(dropped)))}.")
    if len(steps) > MAX_STEPS:
        notes.append(f"Plan had {len(steps)} steps; kept the first {MAX_STEPS}.")
        steps = steps[:MAX_STEPS]

    return Plan(steps=tuple(steps), note=" ".join(notes))


def _parse_step(item: object, question: str) -> tuple[PlanStep | None, str]:
    if not isinstance(item, dict):
        return None, f"step was a {type(item).__name__}, not an object"

    raw_agent = item.get("agent")
    if not isinstance(raw_agent, str):
        return None, "step had no agent name"
    agent = raw_agent.strip().lower()
    if agent not in KNOWN_AGENTS:
        return None, f"unknown agent {agent!r}"

    raw_instruction = item.get("instruction")
    # A step with no instruction used to hand the agent an empty prompt; the question is a
    # better stand-in than nothing.
    instruction = raw_instruction.strip() if isinstance(raw_instruction, str) else ""
    return PlanStep(agent=agent, instruction=instruction or question), ""


def _load_json(raw: object) -> tuple[object, str]:
    """The JSON value in `raw`, or (None, reason). Tolerates code fences and surrounding prose."""
    if not isinstance(raw, str):
        return None, f"a {type(raw).__name__}, not text"
    text = _strip_fence(raw.strip())
    if not text:
        return None, "empty"
    try:
        return json.loads(text), ""
    except ValueError:
        pass
    # Models often wrap the object in a sentence; try the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), ""
        except ValueError:
            pass
    return None, "not JSON"


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    body = text[3:]
    # ```json / ```JSON and friends: drop the language word on the first line.
    newline = body.find("\n")
    if newline != -1 and body[:newline].strip().isalpha():
        body = body[newline + 1 :]
    closing = body.rfind("```")
    return (body[:closing] if closing != -1 else body).strip()
