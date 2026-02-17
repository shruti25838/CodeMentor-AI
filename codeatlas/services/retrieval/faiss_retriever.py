import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.retrieval.interfaces import CodeRetriever


class FaissCodeRetriever(CodeRetriever):
    def __init__(self, base_dir: str | None = None) -> None:
        self._indexes: dict[str, _RepoIndex] = {}
        self._base_dir = Path(base_dir).resolve() if base_dir else None
        self._logger = logging.getLogger(__name__)
        if self._base_dir:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._load_all()

    def index(self, repo_id: str, records: list[EmbeddingRecord]) -> None:
        if not records:
            return
        vectors = np.array([record.vector for record in records], dtype="float32")
        vectors = _normalize(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._indexes[repo_id] = _RepoIndex(index=index, records=records)
        self._persist(repo_id)
        self._logger.info("FAISS index stored for repo %s", repo_id)

    def search(
        self, repo_id: str, query_vector: list[float], top_k: int
    ) -> list[EmbeddingRecord]:
        repo_index = self._indexes.get(repo_id)
        if repo_index is None:
            return []
        query = np.array([query_vector], dtype="float32")
        query = _normalize(query)
        distances, indices = repo_index.index.search(query, top_k)
        result_indices = indices[0]
        results: list[EmbeddingRecord] = []
        for idx in result_indices:
            if idx == -1:
                continue
            if 0 <= idx < len(repo_index.records):
                results.append(repo_index.records[idx])
        return results

    def _persist(self, repo_id: str) -> None:
        if not self._base_dir:
            return
        repo_index = self._indexes.get(repo_id)
        if repo_index is None:
            return
        index_path = self._base_dir / f"{repo_id}.faiss"
        meta_path = self._base_dir / f"{repo_id}.pkl"
        faiss.write_index(repo_index.index, str(index_path))
        meta_path.write_bytes(pickle.dumps(repo_index.records))
        self._logger.info("Persisted FAISS index to %s", index_path)

    def _load_all(self) -> None:
        if not self._base_dir:
            return
        for index_path in self._base_dir.glob("*.faiss"):
            repo_id = index_path.stem
            meta_path = self._base_dir / f"{repo_id}.pkl"
            if not meta_path.exists():
                continue
            try:
                index = faiss.read_index(str(index_path))
                records = pickle.loads(meta_path.read_bytes())
            except Exception:
                continue
            self._indexes[repo_id] = _RepoIndex(index=index, records=records)
            self._logger.info("Loaded FAISS index for repo %s", repo_id)


@dataclass(frozen=True)
class _RepoIndex:
    index: faiss.IndexFlatIP
    records: list[EmbeddingRecord]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
