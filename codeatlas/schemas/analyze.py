from pydantic import BaseModel, HttpUrl


class AnalyzeRepoRequest(BaseModel):
    repo_url: HttpUrl


class AnalyzeRepoResponse(BaseModel):
    repository_id: str
    file_count: int
    dependency_edges: int
    indexing_status: str = "complete"
