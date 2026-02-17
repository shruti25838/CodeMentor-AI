from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    path: str
    language: str
    size_bytes: int
