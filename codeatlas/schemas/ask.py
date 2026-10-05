from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    repo_id: str | None = None
    question: str
    # Random id made by the browser for this tab's session (see codementor-ui/lib/chatSession.ts).
    # With it, /ask/stream keeps the last few turns; without it, every question stands alone.
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{16,128}$")


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
