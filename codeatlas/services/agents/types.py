from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
    # The agents that really ran, in order, with repeats: a plan may use one agent twice.
    # Taken from the graph as it executed, never from a list written down in advance.
    agents_used: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerateResult:
    diff: str
    notes: list[str]
    citations: list[str]
