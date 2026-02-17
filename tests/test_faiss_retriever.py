from pathlib import Path

from codeatlas.models.embedding_record import EmbeddingRecord
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever


def test_faiss_persist_and_search(tmp_path: Path) -> None:
    repo_id = "repo-1"
    retriever = FaissCodeRetriever(base_dir=str(tmp_path))
    records = [
        EmbeddingRecord(
            record_id="a.py:1-2",
            scope="function",
            vector=[1.0, 0.0, 0.0],
            metadata={"path": "a.py"},
        ),
        EmbeddingRecord(
            record_id="b.py:1-2",
            scope="function",
            vector=[0.0, 1.0, 0.0],
            metadata={"path": "b.py"},
        ),
    ]
    retriever.index(repo_id, records)

    reloaded = FaissCodeRetriever(base_dir=str(tmp_path))
    results = reloaded.search(repo_id, [1.0, 0.0, 0.0], top_k=1)
    assert results
    assert results[0].record_id == "a.py:1-2"
