from pydantic import BaseModel


class GenerateCodeRequest(BaseModel):
    repo_id: str
    prompt: str


class GenerateCodeResponse(BaseModel):
    diff: str
    notes: list[str]
    citations: list[str]
