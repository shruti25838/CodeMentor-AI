import json
import logging
import threading

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from test_analyze_indexing import Env

from codeatlas.app.di import get_agent_orchestrator, get_llm_provider
from codeatlas.observability.timing import (
    StageTimer,
    run_timed,
    stage,
    timed_request,
    watch_provider_retries,
)
from codeatlas.services.agents.coding_mentor_agent import CodingMentorAgent
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.agents.retrieval_agent import RetrievalAgent
from codeatlas.services.qa.answer_service import AnswerService

QUESTION = "how does add work with secret-question-text?"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_repeated_stages_add_up_and_count_calls():
    clock = FakeClock()
    timer = StageTimer(clock=clock)
    for seconds in (1.0, 2.0):
        with timer.stage("llm"):
            clock.now += seconds
    with timer.stage("search"):
        clock.now += 0.5
    timer.mark("first_token_sent")
    clock.now += 1.0
    timer.mark("first_token_sent")  # keeps the first value
    assert timer.as_dict() == {
        "llm": 3000.0,
        "search": 500.0,
        "first_token_sent": 3500.0,
        "llm_calls": 2,
        # Which of the two calls was the slow one; the total alone cannot say.
        "llm_each": [1000.0, 2000.0],
        "total": 4500.0,
    }
    assert timer.server_timing() == "llm;dur=3000.0, search;dur=500.0, first_token_sent;dur=3500.0, total;dur=4500.0"


def test_counts_and_facts_are_reported_but_never_timed():
    """A token count in Server-Timing would be read by the browser as a number of milliseconds."""
    clock = FakeClock()
    timer = StageTimer(clock=clock)
    with timer.stage("llm"):
        clock.now += 1.0
    timer.count("in_tokens", 1800)
    timer.count("in_tokens", 2400)
    timer.fact("history_turns", 4)
    timer.fact("rewrite_ran", 1)

    timings = timer.as_dict()
    assert timings["in_tokens"] == 4200
    assert timings["in_tokens_each"] == [1800, 2400]
    assert timings["history_turns"] == 4 and timings["rewrite_ran"] == 1
    # One call, so no _each for it: the stage total is already that one call's duration.
    assert "llm_each" not in timings and "llm_calls" not in timings

    assert timer.server_timing() == "llm;dur=1000.0, total;dur=1000.0"
    for name in ("in_tokens", "history_turns", "rewrite_ran"):
        assert name not in timer.server_timing()
    assert timer.durations() == {"llm": 1000.0, "total": 1000.0}


def test_the_log_line_keeps_per_call_lists_on_one_splittable_field(caplog):
    timer = StageTimer(clock=FakeClock())
    timer.count("in_tokens", 1800)
    timer.count("in_tokens", 2400)
    with caplog.at_level(logging.INFO, logger="codeatlas.observability.timing"):
        timer.log("unit")
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("timing unit"))
    assert "in_tokens=4200" in line
    # Joined with a pipe rather than rendered as a Python list, which would put spaces in a
    # field and break anything that reads the line by splitting on whitespace.
    assert "in_tokens_each=1800|2400" in line
    assert "[" not in line


def test_stage_outside_a_request_does_nothing():
    with stage("embed"):
        pass  # no timer, no error


def test_stage_reports_into_the_current_timer_only_inside_the_block():
    with timed_request() as timer, stage("parse"):
        pass
    with stage("parse"):
        pass
    assert timer.as_dict().keys() == {"parse", "total"}


def test_run_timed_reaches_a_worker_thread():
    timer = StageTimer()

    def work():
        with stage("embed_query"):
            pass

    t = threading.Thread(target=run_timed, args=(timer, work))
    t.start()
    t.join()
    assert "embed_query" in timer.as_dict()


def test_analyze_reports_every_indexing_stage(tmp_path, caplog):
    env = Env(tmp_path)
    with caplog.at_level(logging.INFO, logger="codeatlas.observability.timing"):
        resp = env.analyze()
    assert resp.status_code == 200
    names = {part.split(";")[0] for part in resp.headers["Server-Timing"].split(", ")}
    assert names == {
        "cache_lookup",
        "clone",
        "parse",
        "graph",
        "read_files",
        "embed",
        "index_write",
        "save_state",
        "total",
    }
    assert any(r.message.startswith("timing analyze ") and " clone=" in r.message for r in caplog.records)


def _chat_env(tmp_path, answers):
    env = Env(tmp_path)
    llm = FakeListChatModel(responses=answers)
    answer_service = AnswerService(retriever=env.index._retriever, embedder=env.index._embedder, llm=llm)
    retrieval = RetrievalAgent(answer_service=answer_service)
    mentor = CodingMentorAgent(answer_service=answer_service, llm=llm)
    orchestrator = AgentOrchestrator(
        planner=mentor,
        retrieval_agent=retrieval,
        analyst_agent=mentor,
        mentor_agent=mentor,
        memory_agent=mentor,
    )
    env.client.app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator
    return env


def _events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_ask_stream_reports_chat_stages_without_question_text(tmp_path, caplog):
    env = _chat_env(tmp_path, ["the mentor's answer"])
    repo_id = env.analyze().json()["repository_id"]
    with caplog.at_level(logging.INFO):
        resp = env.client.post("/ask/stream", json={"question": QUESTION, "repo_id": repo_id})
    done = _events(resp)[-1]
    assert done["type"] == "done"
    timings = done["timings_ms"]
    # One model call per question: the mentor's. It was three until the mentor stopped running
    # its own second search, and two until the retrieval summary - whose reply nothing read -
    # was removed. `llm_calls` is absent because a count appears only for a stage that ran more
    # than once, so `llm` is that one call's duration and needs no breakdown beside it.
    assert "llm_calls" not in timings and "llm_each" not in timings
    assert "embed_query_calls" not in timings, "one search per question, so no repeat count"
    assert timings["search"] >= 0
    for name in ("embed_query", "search", "rerank", "llm", "first_token_sent", "total"):
        assert name in timings
    assert timings["first_token_sent"] <= timings["total"]
    timing_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("timing ask_stream")]
    assert len(timing_lines) == 1
    assert "secret-question-text" not in caplog.text


def test_ask_sets_server_timing(tmp_path):
    env = _chat_env(tmp_path, ["answer"] * 10)
    llm = FakeListChatModel(responses=["general answer"])
    env.client.app.dependency_overrides[get_llm_provider] = lambda: type(
        "P", (), {"get_chat_model": lambda self: llm}
    )()
    resp = env.client.post("/ask", json={"question": QUESTION})
    assert resp.status_code == 200
    assert "llm;dur=" in resp.headers["Server-Timing"]


def test_history_turns_and_rewrite_ran_are_on_every_chat_line(tmp_path):
    """Two of the four candidates for a chat that slows down are not durations at all."""
    env = _chat_env(tmp_path, ["mentor answer"] * 4)
    llm = FakeListChatModel(responses=["what does add do?"] * 4)
    env.client.app.dependency_overrides[get_llm_provider] = lambda: type(
        "P", (), {"get_chat_model": lambda self: llm}
    )()
    repo_id = env.analyze().json()["repository_id"]
    session = "session-for-timing"

    first = _events(
        env.client.post("/ask/stream", json={"question": QUESTION, "repo_id": repo_id, "session_id": session})
    )[-1]
    assert first["timings_ms"]["history_turns"] == 0
    # Nothing to refer back to on the first question, so no rewrite call was made.
    assert first["timings_ms"]["rewrite_ran"] == 0
    assert "rewrite" not in first["timings_ms"]

    follow_up = {"question": "why does it do that?", "repo_id": repo_id, "session_id": session}
    second = _events(env.client.post("/ask/stream", json=follow_up))[-1]
    assert second["timings_ms"]["history_turns"] == 1
    # "it" points back at the conversation, so this question paid for an extra model call.
    assert second["timings_ms"]["rewrite_ran"] == 1
    assert second["timings_ms"]["rewrite"] >= 0
    # Two model calls on a follow-up: the rewrite and the mentor. They are separate stages, so
    # neither repeats and `llm_calls` stays absent; the second call is visible as `rewrite`
    # having a duration at all. Before the retrieval summary was removed this was three.
    assert "llm_calls" not in second["timings_ms"]


def test_a_provider_retry_backoff_lands_on_the_waiting_request():
    """The client sleeps through a 429 inside the model call, where no stage can see it."""
    watch_provider_retries()
    timer = StageTimer()
    sdk_logger = logging.getLogger("openai._base_client")

    def pretend_the_client_retried():
        # The line openai's own client logs just before it sleeps, args and all.
        sdk_logger.info("Retrying request to %s in %f seconds", "/chat/completions", 8.0)

    run_timed(timer, pretend_the_client_retried)

    timings = timer.as_dict()
    assert timings["retry_wait"] == 8000.0
    assert "/chat/completions" not in str(timings)


def test_a_retry_outside_a_request_is_ignored_and_other_provider_lines_are_too():
    watch_provider_retries()
    logging.getLogger("openai._base_client").info("Retrying request to %s in %f seconds", "/x", 1.0)  # no timer

    timer = StageTimer()
    run_timed(timer, lambda: logging.getLogger("openai").info("Request options: %s", "something"))
    assert "retry_wait" not in timer.as_dict()
