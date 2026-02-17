from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    relation: str
