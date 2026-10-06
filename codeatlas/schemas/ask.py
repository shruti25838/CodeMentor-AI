from typing import Literal

from pydantic import BaseModel, Field

# A question longer than this is refused before any model call. Deep mode sends the question
# to several agents, so a very long one multiplies across the whole pipeline.
MAX_QUESTION_CHARS = 4000


class AskRequest(BaseModel):
    repo_id: str | None = None
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    # Random id made by the browser for this tab's session (see codementor-ui/lib/chatSession.ts).
    # With it, /ask/stream keeps the last few turns; without it, every question stands alone.
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{16,128}$")
    # "fast" is retrieval then mentor, and stays the default. "deep" runs the planned
    # five-agent pipeline and streams each agent's step as it finishes.
    mode: Literal["fast", "deep"] = "fast"


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
    # The agents that actually ran, in order. Empty for a question answered without agents.
    agents_used: list[str] = Field(default_factory=list)
