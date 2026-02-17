from pydantic import BaseModel


class DependenciesRequest(BaseModel):
    repo_id: str
    node_id: str
    direction: str = "inbound"


class DependenciesResponse(BaseModel):
    node_id: str
    direction: str
    neighbors: list[str]


class GraphRequest(BaseModel):
    repo_id: str


class GraphEdge(BaseModel):
    source: str
    target: str


class GraphResponse(BaseModel):
    nodes: list[str]
    edges: list[GraphEdge]
