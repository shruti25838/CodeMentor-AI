from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingRecord:
    record_id: str
    scope: str
    vector: list[float]
    metadata: dict[str, str]
