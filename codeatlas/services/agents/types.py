from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[str]
    reasoning_steps: list[str]


@dataclass(frozen=True)
class GenerateResult:
    diff: str
    notes: list[str]
    citations: list[str]
