import asyncio
import json
import time

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from codeatlas.app.di import get_agent_orchestrator, get_llm_provider, get_repo_state_store
from codeatlas.controllers.repo_guard import require_known_repo
from codeatlas.observability.timing import StageTimer, run_timed, timed_request
from codeatlas.observability.tracker import tracker
from codeatlas.schemas.ask import AskRequest, AskResponse
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.llm.provider import LlmProvider
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/ask", tags=["qa"])


# ---------- normal (non-streaming) endpoint ----------
@router.post("", response_model=AskResponse)
def ask(
    request: AskRequest,
    response: Response,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    llm_provider: LlmProvider = Depends(get_llm_provider),
    state_store: RepoStateStore = Depends(get_repo_state_store),
) -> AskResponse:
    if request.repo_id:
        require_known_repo(state_store, request.repo_id)
    with timed_request() as timer:
        resp = _ask(request, orchestrator, llm_provider, timer)
        response.headers["Server-Timing"] = timer.server_timing()
        timer.log("ask")
    return resp


def _ask(request, orchestrator, llm_provider, timer: StageTimer) -> AskResponse:
    start = time.perf_counter()
    # General mode — no repo, just answer the coding question directly
    if not request.repo_id:
        llm = llm_provider.get_chat_model()
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a senior software engineer and coding mentor. "
                    "Answer the user's coding question clearly with examples where helpful.",
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm
        with timer.stage("llm"):
            response = chain.invoke({"question": request.question})
        resp = AskResponse(
            answer=response.content,
            citations=[],
            reasoning_steps=["General mode: answered without repo context."],
        )
        _track(request, resp, start)
        return resp

    result = orchestrator.handle_question(request.question, request.repo_id)
    resp = AskResponse(
        answer=result.answer,
        citations=result.citations,
        reasoning_steps=result.reasoning_steps,
    )
    _track(request, resp, start)
    return resp


# ---------- streaming SSE endpoint ----------
@router.post("/stream")
async def ask_stream(
    request: AskRequest,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    llm_provider: LlmProvider = Depends(get_llm_provider),
    state_store: RepoStateStore = Depends(get_repo_state_store),
):
    # Checked before the stream opens so the client gets a real 404, not a 200 with an error event.
    if request.repo_id:
        require_known_repo(state_store, request.repo_id)

    async def generate():
        start = time.perf_counter()
        # Not a contextvar here: the repo-mode work runs in a thread pool, which gets it via run_timed.
        timer = StageTimer()
        try:
            if not request.repo_id:
                # ---- General mode: true LLM token streaming ----
                yield _sse({"type": "status", "content": "Thinking..."})
                llm = llm_provider.get_chat_model()
                from langchain_core.prompts import ChatPromptTemplate

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a senior software engineer and coding mentor. "
                            "Answer the user's coding question clearly with examples where helpful.",
                        ),
                        ("human", "{question}"),
                    ]
                )
                chain = prompt | llm
                full_answer = ""
                for chunk in chain.stream({"question": request.question}):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        timer.mark("first_model_token")
                        full_answer += token
                        timer.mark("first_token_sent")
                        yield _sse({"type": "token", "content": token})

                latency = (time.perf_counter() - start) * 1000
                tracker.record_query(
                    question=request.question,
                    repo_id=None,
                    latency_ms=latency,
                    citation_count=0,
                    agents_used=["mentor"],
                )
                timer.log("ask_stream")
                yield _sse(
                    {
                        "type": "done",
                        "citations": [],
                        "reasoning_steps": ["General mode: streamed response."],
                        "timings_ms": timer.as_dict(),
                    }
                )
            else:
                # ---- Repo mode: fast path (retrieval → mentor, no planner/validator) ----
                yield _sse({"type": "status", "content": "Retrieving context..."})

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    run_timed,
                    timer,
                    lambda: orchestrator.handle_question_fast(request.question, request.repo_id),
                )

                yield _sse({"type": "status", "content": "Streaming answer..."})

                # Stream answer in small chunks for natural feel
                answer = result.answer
                chunk_size = 4  # words per chunk
                words = answer.split(" ")
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i : i + chunk_size])
                    if i + chunk_size < len(words):
                        chunk += " "
                    timer.mark("first_token_sent")
                    yield _sse({"type": "token", "content": chunk})
                    await asyncio.sleep(0.008)

                latency = (time.perf_counter() - start) * 1000
                tracker.record_query(
                    question=request.question,
                    repo_id=request.repo_id,
                    latency_ms=latency,
                    citation_count=len(result.citations),
                    agents_used=["retrieval", "mentor"],
                )
                timer.log("ask_stream")
                yield _sse(
                    {
                        "type": "done",
                        "citations": result.citations,
                        "reasoning_steps": result.reasoning_steps,
                        "timings_ms": timer.as_dict(),
                    }
                )
        except Exception as e:
            yield _sse({"type": "error", "content": str(e)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- helpers ----------
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _track(request: AskRequest, resp: AskResponse, start: float) -> None:
    latency = (time.perf_counter() - start) * 1000
    agents = ["mentor"] if not request.repo_id else ["planner", "retrieval", "mentor", "validator"]
    tracker.record_query(
        question=request.question,
        repo_id=request.repo_id,
        latency_ms=latency,
        citation_count=len(resp.citations),
        agents_used=agents,
    )
