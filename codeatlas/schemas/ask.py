from pydantic import BaseModel


class AskRequest(BaseModel):
    repo_id: str | None = None
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    reasoning_steps: list[str]
