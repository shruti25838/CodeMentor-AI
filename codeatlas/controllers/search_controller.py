"""Raw vector search over one repository's index.

**This endpoint bypasses the retrieval pipeline**, and that is what it is for: it embeds the
query and returns what the index gives back, with none of the reranking, stopword dropping,
test-file filtering or fallback-to-hash that a question through /ask goes through. Its results
are therefore not what a visitor would see for the same words, and it is not the thing to
measure retrieval quality with - scripts/eval_retrieval.py goes through AnswerService, which
is the pipeline. It is behind the admin key, and it stays as it is on purpose: a way to look
at the index itself rather than at the answer built from it.

It does follow the selected provider: the index it searches is the one that provider writes.
"""

from fastapi import APIRouter, Depends

from codeatlas.app.di import get_code_retriever, get_embedder
from codeatlas.schemas.search import SearchHit, SearchRequest, SearchResponse
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
