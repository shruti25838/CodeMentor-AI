import asyncio
import json
import queue
import time
from dataclasses import replace

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from codeatlas.app.di import (
    get_agent_orchestrator,
    get_conversation_store,
    get_llm_provider,
    get_repo_state_store,
    get_retrieval_service,
)
from codeatlas.controllers.repo_guard import require_known_repo
from codeatlas.observability.metrics import ERROR_COUNT
from codeatlas.observability.timing import StageTimer, record_fact, run_timed, stage, timed_request
from codeatlas.observability.tracker import tracker
from codeatlas.schemas.ask import AskRequest, AskResponse
from codeatlas.services.agents.orchestration import ANSWER_ERROR_PREFIX, AgentOrchestrator
from codeatlas.services.llm.provider import LlmProvider
from codeatlas.services.memory.conversation import ConversationStore, Turn, format_history
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.qa.followup import needs_rewrite, rewrite_question
from codeatlas.services.qa.grounded_chat import (
    GROUNDED,
    STAGE_ANSWERING,
    STAGE_PLANNING,
    STAGE_RETRIEVING,
    STAGE_THINKING,
    build_messages,
    grounded_context,
    retrieved_event,
    stage_event,
    stream_tokens,
)
from codeatlas.services.state.repo_state_store import RepoStateStore

router = APIRouter(prefix="/ask", tags=["qa"])

# A question with no repository is answered by one model call, not by the agent graph.
# Recording it as "mentor" would claim an agent ran that did not.
DIRECT_ANSWER = ("direct",)

# Grounded mode runs no agent graph: one retrieval and one model call. Recording it as
# "mentor" would name an agent that did not run, which is the mistake fixed at 5d129c7.
GROUNDED_ANSWER = ("grounded",)


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
            agents_used=list(DIRECT_ANSWER),
        )
        _track(request, resp, start)
        return resp

    # session_id was ignored here before, so /ask had no conversation memory at all.
    result = orchestrator.handle_question(request.question, request.repo_id, session_id=request.session_id)
    resp = AskResponse(
        answer=result.answer,
        citations=result.citations,
        reasoning_steps=result.reasoning_steps,
        agents_used=result.agents_used,
        retrieval_note=result.note,
    )
    _track(request, resp, start)
    return resp


# ---------- streaming SSE endpoint ----------
@router.post("/stream")
async def ask_stream(
    request: AskRequest,
    http_request: Request,
    orchestrator: AgentOrchestrator = Depends(get_agent_orchestrator),
    llm_provider: LlmProvider = Depends(get_llm_provider),
    state_store: RepoStateStore = Depends(get_repo_state_store),
    conversations: ConversationStore = Depends(get_conversation_store),
    answers: AnswerService = Depends(get_retrieval_service),
):
    # Checked before the stream opens so the client gets a real 404, not a 200 with an error event.
    if request.repo_id:
        require_known_repo(state_store, request.repo_id)

    async def generate():
        start = time.perf_counter()
        # Not a contextvar here: the repo-mode work runs in a thread pool, which gets it via run_timed.
        timer = StageTimer()
        session_id = request.session_id
        history = conversations.history(session_id, request.repo_id) if session_id else []
        # How much of this conversation goes into the prompt. The first candidate for a chat
        # that slows down as it goes on, and the one number no stage duration can reveal.
        timer.fact("history_turns", len(history))
        try:
            if not request.repo_id:
                # ---- General mode: true LLM token streaming ----
                yield _sse(stage_event(STAGE_THINKING, "Thinking..."))
                llm = llm_provider.get_chat_model()
                from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a senior software engineer and coding mentor. "
                            "Answer the user's coding question clearly with examples where helpful.",
                        ),
                        MessagesPlaceholder("history"),
                        ("human", "{question}"),
                    ]
                )
                chain = prompt | llm
                full_answer = ""
                for chunk in chain.stream({"question": request.question, "history": _as_messages(history)}):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        timer.mark("first_model_token")
                        full_answer += token
                        timer.mark("first_token_sent")
                        yield _sse({"type": "token", "content": token})

                if session_id and full_answer:
                    conversations.add(session_id, None, request.question, full_answer)
                latency = (time.perf_counter() - start) * 1000
                tracker.record_query(
                    repo_id=None,
                    latency_ms=latency,
                    citation_count=0,
                    agents_used=list(DIRECT_ANSWER),
                )
                timer.log("ask_stream")
                yield _sse(
                    {
                        "type": "done",
                        "citations": [],
                        "reasoning_steps": ["General mode: streamed response."],
                        "agents_used": list(DIRECT_ANSWER),
                        "timings_ms": timer.as_dict(),
                    }
                )
            elif _grounded(http_request) and request.mode != "deep":
                # ---- Repo mode, grounded: one search, one streamed model call ----
                async for event in _run_grounded(answers, llm_provider, request, history, timer, conversations, start):
                    yield _sse(event)
            else:
                # ---- Repo mode ----
                if request.mode == "deep":
                    # The planned five-agent pipeline. Each agent's step is sent as it
                    # finishes, so the wait is visible rather than silent.
                    yield _sse(stage_event(STAGE_PLANNING, "Planning..."))
                    result = None
                    async for event in _run_deep(orchestrator, llm_provider, request, history, timer):
                        if event["type"] == "result":
                            result = event["result"]
                        else:
                            yield _sse(event)
                    if result is None:  # pragma: no cover - the generator always ends with one
                        raise RuntimeError("deep mode produced no answer")
                else:
                    # ---- Repo mode, the default: one search, then one model call ----
                    yield _sse(stage_event(STAGE_RETRIEVING, "Retrieving context..."))
                    loop = asyncio.get_event_loop()
                    search_question = await loop.run_in_executor(
                        None,
                        run_timed,
                        timer,
                        lambda: _search_question(llm_provider, request.question, history),
                    )
                    retrieved = await loop.run_in_executor(
                        None,
                        run_timed,
                        timer,
                        lambda: orchestrator.search(search_question or request.question, request.repo_id),
                    )
                    # The search is finished and the model has not been asked anything yet. This
                    # path could not send this event while it summarised its own search through
                    # the model first; now that it does not, the one moment grounded mode has -
                    # after retrieval, before any spending - exists here too. The files are
                    # `retrieved.snippets`, and the citations below are `retrieved.citations`,
                    # built from that same list by the same search, so the two cannot disagree.
                    yield _sse(retrieved_event(retrieved.snippets))
                    result = await loop.run_in_executor(
                        None,
                        run_timed,
                        timer,
                        lambda: _answer_from(orchestrator, request, retrieved, history, search_question),
                    )

                yield _sse(stage_event(STAGE_ANSWERING, "Streaming answer..."))

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

                if session_id and not answer.startswith(ANSWER_ERROR_PREFIX):
                    conversations.add(session_id, request.repo_id, request.question, answer)
                latency = (time.perf_counter() - start) * 1000
                tracker.record_query(
                    repo_id=request.repo_id,
                    latency_ms=latency,
                    citation_count=len(result.citations),
                    agents_used=result.agents_used,
                )
                timer.log("ask_stream")
                yield _sse(
                    {
                        "type": "done",
                        "citations": result.citations,
                        "reasoning_steps": result.reasoning_steps,
                        "agents_used": result.agents_used,
                        "retrieval_note": result.note,
                        "timings_ms": timer.as_dict(),
                    }
                )
        except Exception as e:
            # The stream already answered 200, so the middleware cannot see this failure.
            ERROR_COUNT.labels(path="/ask/stream", kind="stream_error").inc()
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


def _grounded(http_request: Request) -> bool:
    """Whether this server answers repository questions the grounded way. Default: it does not."""
    return getattr(http_request.app.state.config, "chat_mode", "") == GROUNDED


async def _run_grounded(answers, llm_provider, request: AskRequest, history, timer, conversations, start):
    """One search, then the model's own tokens, forwarded as they arrive.

    Everything the legacy path guarantees still holds: the unknown-repo 404 happened before the
    stream opened, the question cap was enforced by the request schema, the session history is
    in the prompt and the answer goes back into it, the semantic fallback note reaches the done
    event, and a provider refusal becomes the same plain sentence with the files beneath it.
    """
    loop = asyncio.get_event_loop()
    yield stage_event(STAGE_RETRIEVING, "Retrieving context...")

    rewritten = await loop.run_in_executor(
        None,
        run_timed,
        timer,
        lambda: _search_question(llm_provider, request.question, history),
    )
    search_question = rewritten or request.question

    context = await loop.run_in_executor(
        None,
        run_timed,
        timer,
        lambda: grounded_context(answers, request.repo_id, search_question, top_k=5, total_max_chars=6000),
    )

    # The search is finished and the model has not been asked anything yet, so this is the one
    # moment at which the files that will be cited are known and nothing has been spent. A
    # reasoning model then spends its first seconds on tokens nobody sees; rather than leaving
    # the visitor with a retrieval message that is already stale, they get the file list they
    # are about to be shown underneath the answer. It is the same list, by construction - see
    # retrieved_event - and no question text goes into it.
    yield retrieved_event(context.snippets)

    llm = llm_provider.get_chat_model()
    messages = build_messages(request.question, context, history)
    outcome = None
    stream = stream_tokens(llm, messages, context, on_first_token=lambda: timer.mark("first_model_token"))
    # This mode's one model call, timed as the `llm` stage the other paths record, so a grounded
    # question's line can be read beside a legacy one. `run_timed` reaches the worker thread the
    # stream is pulled on: the usage callback and the retry watcher both report into the timer
    # from in there, and without it neither had one to report into.
    model_started = time.perf_counter()
    while True:
        item = await loop.run_in_executor(None, run_timed, timer, lambda it=stream: next(it, None))
        if item is None:
            break
        token, finished = item
        if finished is not None:
            outcome = finished
            break
        if token:
            timer.mark("first_token_sent")
            yield {"type": "token", "content": token}
    timer.add("llm", (time.perf_counter() - model_started) * 1000)

    answer = outcome.answer if outcome else ""
    steps = list(context.reasoning_steps)
    if history:
        steps.append(f"Used {len(history)} earlier turn(s) of this conversation.")
    if search_question != request.question:
        steps.append(f"Searched with the follow-up rewritten as: {search_question}")

    if request.session_id and answer and outcome and not outcome.failed and not outcome.quota:
        conversations.add(request.session_id, request.repo_id, request.question, answer)
    tracker.record_query(
        repo_id=request.repo_id,
        latency_ms=(time.perf_counter() - start) * 1000,
        citation_count=len(context.citations),
        agents_used=list(GROUNDED_ANSWER),
    )
    timer.log("ask_stream")
    yield {
        "type": "done",
        "citations": context.citations,
        "reasoning_steps": steps,
        "agents_used": list(GROUNDED_ANSWER),
        "retrieval_note": context.note,
        "timings_ms": timer.as_dict(),
    }


def _search_question(llm_provider, question: str, history: list[Turn]) -> str | None:
    """The standalone rewrite of a follow-up, or None when the question stands on its own.

    One place rather than three, so `rewrite_ran` - whether this question paid for an extra
    model call - is recorded on every path that could make one, instead of on whichever paths
    someone remembered. The fact is recorded either way: a line with no `rewrite_ran` at all
    would be indistinguishable from an older line that never had the field.
    """
    if not needs_rewrite(question, history):
        record_fact("rewrite_ran", 0)
        return None
    record_fact("rewrite_ran", 1)
    with stage("rewrite"):
        return rewrite_question(llm_provider.get_chat_model(), question, history)


async def _run_deep(orchestrator, llm_provider, request: AskRequest, history: list[Turn], timer: StageTimer):
    """Bridge the orchestrator's blocking generator into the event loop.

    The graph calls the model, so it must not run on the event loop. A worker thread pushes
    each event onto a queue as the graph produces it, and this reads them as they arrive, so
    an agent's step reaches the browser while the next agent is still working.
    """
    events: queue.Queue = queue.Queue()
    done = object()

    def produce() -> None:
        try:
            # A follow-up is rewritten into a standalone question for search, exactly as the
            # fast path does, so deep mode searches with the visitor's meaning rather than
            # with "why does it do that?".
            search_question = _search_question(llm_provider, request.question, history)
            for event in orchestrator.stream_question(
                request.question, request.repo_id, request.session_id, search_question=search_question
            ):
                events.put(event)
        except Exception as exc:  # noqa: BLE001 - reported to the caller as an event
            events.put({"type": "failed", "error": exc})
        finally:
            events.put(done)

    loop = asyncio.get_event_loop()
    worker = loop.run_in_executor(None, run_timed, timer, produce)
    try:
        while True:
            event = await loop.run_in_executor(None, events.get)
            if event is done:
                break
            if event["type"] == "failed":
                raise event["error"]
            if event["type"] == "agent":
                timer.mark("first_agent_event")
            yield event
    finally:
        await worker


def _answer_from(orchestrator, request: AskRequest, retrieved, history: list[Turn], search_question: str | None):
    """The default path's answer, over snippets the controller already holds.

    The two halves are run separately so the file list can go out between them; this is the
    second. It adds the same conversation notes the combined call used to add.

    **The result is copied, not rebuilt.** `replace` carries every field whether or not anyone
    here remembers the field exists; only `reasoning_steps` is given a new value. The
    field-by-field rebuild this replaces named four of the five and dropped `note`, which is
    the one sentence telling a visitor their answer came from keyword search rather than from
    semantic search - and on the live server, which runs the cloudflare provider, that is
    exactly the case it exists for. A field added to the result type later arrives carried.
    """
    result = orchestrator.answer_from(
        request.question,
        retrieved,
        request.repo_id,
        history=format_history(history),
    )
    steps = list(result.reasoning_steps)
    if history:
        steps.append(f"Used {len(history)} earlier turn(s) of this conversation.")
    if search_question is not None:
        steps.append(f"Searched with the follow-up rewritten as: {search_question}")
    return replace(result, reasoning_steps=steps)


def _as_messages(history: list[Turn]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    messages: list = []
    for turn in history:
        messages += [HumanMessage(turn.question), AIMessage(turn.answer)]
    return messages


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _track(request: AskRequest, resp: AskResponse, start: float) -> None:
    latency = (time.perf_counter() - start) * 1000
    tracker.record_query(
        repo_id=request.repo_id,
        latency_ms=latency,
        citation_count=len(resp.citations),
        # What really ran, as reported by the graph; this used to be a fixed list that named
        # a validator that does not exist and never named the analyst even when it ran.
        agents_used=resp.agents_used,
    )
