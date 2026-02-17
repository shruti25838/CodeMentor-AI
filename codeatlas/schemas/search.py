from pydantic import BaseModel


class SearchRequest(BaseModel):
    repo_id: str
    query: str
    top_k: int = 5


class SearchHit(BaseModel):
    record_id: str
    scope: str
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    results: list[SearchHit]
