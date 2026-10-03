from pydantic import BaseModel


class AnalyzeRepoRequest(BaseModel):
    # Plain str so a bad URL gets the loader's friendly message, not pydantic's 422 error list.
    repo_url: str


class AnalyzeRepoResponse(BaseModel):
    repository_id: str
    file_count: int
    dependency_edges: int
    indexing_status: str = "complete"
