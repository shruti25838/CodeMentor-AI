from pydantic import BaseModel


class RepoInfo(BaseModel):
    repo_id: str
    name: str


class ListReposResponse(BaseModel):
    repo_ids: list[str]
    repos: list[RepoInfo] = []