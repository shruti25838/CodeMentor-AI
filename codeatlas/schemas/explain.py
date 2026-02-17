from pydantic import BaseModel


class ExplainRequest(BaseModel):
    repo_id: str
    node_id: str


class ExplainResponse(BaseModel):
    node_id: str
    summary: str
    snippet: str
