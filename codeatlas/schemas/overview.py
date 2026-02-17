from pydantic import BaseModel


class RepoOverviewRequest(BaseModel):
    repo_id: str


class RepoOverviewResponse(BaseModel):
    repo_id: str
    file_count: int
    function_count: int
    language_counts: dict[str, int]
    dependency_edges: int
