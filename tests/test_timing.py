import json
import logging
import threading

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from test_analyze_indexing import Env

from codeatlas.app.di import get_agent_orchestrator, get_llm_provider
from codeatlas.observability.timing import StageTimer, run_timed, stage, timed_request
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
        "total": 4500.0,
    }
    assert timer.server_timing() == "llm;dur=3000.0, search;dur=500.0, first_token_sent;dur=3500.0, total;dur=4500.0"


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
    assert names == {"clone", "parse", "graph", "read_files", "embed", "index_write", "save_state", "total"}
    assert any(r.message.startswith("timing analyze clone=") for r in caplog.records)


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
    env = _chat_env(tmp_path, ["retrieval summary", "mentor reasoning", "final answer"])
    repo_id = env.analyze().json()["repository_id"]
    with caplog.at_level(logging.INFO):
        resp = env.client.post("/ask/stream", json={"question": QUESTION, "repo_id": repo_id})
    done = _events(resp)[-1]
    assert done["type"] == "done"
    timings = done["timings_ms"]
    # The fast path asks the model three times: retrieval summary, mentor's own retrieval, mentor answer.
    assert timings["llm_calls"] == 3
    assert timings["embed_query_calls"] == 2
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
