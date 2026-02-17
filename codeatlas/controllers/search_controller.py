from fastapi import APIRouter, Depends

from codeatlas.app.di import get_code_retriever, get_embedder
from codeatlas.schemas.search import SearchRequest, SearchResponse, SearchHit
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.interfaces import CodeRetriever

router = APIRouter(prefix="/search", tags=["retrieval"])


@router.post("", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever: CodeRetriever = Depends(get_code_retriever),
    embedder: EmbeddingService = Depends(get_embedder),
) -> SearchResponse:
    query_vector = embedder.embed_query(request.query)
    records = retriever.search(request.repo_id, query_vector, request.top_k)
    results = [
        SearchHit(
            record_id=record.record_id,
            scope=record.scope,
            metadata=record.metadata,
        )
        for record in records
    ]
    return SearchResponse(results=results)
