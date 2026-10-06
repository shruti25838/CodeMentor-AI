"""What a visitor sees when the provider refuses, and when an answer is cut off.

Before this, a refused call reached the visitor either as raw exception text
(`Error generating answer: Error code: 429 - {'error': ...}`) or as a bare list of paths
headed "Top relevant locations:", which reads as if it were the answer to the question.

Every test here uses a fake model; none needs a key or a network.
"""

import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_agent_pipeline import FakeAgent, plan_of

from codeatlas.services.agents.coding_mentor_agent import CodingMentorAgent
from codeatlas.services.agents.orchestration import ANSWER_ERROR_PREFIX, AgentOrchestrator
from codeatlas.services.llm.quota import (
    DAILY,
    FILES_HEADING,
    MESSAGES,
    NO_FILES,
    PER_MINUTE,
    TOO_LARGE,
    TRUNCATED_NOTE,
    classify_quota,
    quota_answer,
    was_truncated,
)
from codeatlas.services.qa.answer_service import GroundedAnswer
from codeatlas.services.retrieval.snippets import Snippet

SNIPPETS = [Snippet("src/signer.py", 31, 37, "def sign(self): ..."), Snippet("src/exc.py", 1, 5, "class Bad: ...")]

DAILY_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` "
    "on tokens per day (TPD): Limit 200000, Used 199248, Requested 1503.'}}"
)
MINUTE_ERROR = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` "
    "on tokens per minute (TPM): Limit 8000, Requested 1503.'}}"
)
TOO_LARGE_ERROR = (
    "Error code: 413 - {'error': {'message': 'Request too large for model `openai/gpt-oss-20b` "
    "on tokens per minute (TPM): Limit 8000, Requested 10391.'}}"
)


class FailingModel(BaseChatModel):
    state: Any = None

    def __init__(self, error: Exception | None = None, finish_reason: str = "stop", **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"error": error, "finish_reason": finish_reason})()

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.state.error:
            raise self.state.error
        message = AIMessage(
            content="Signing works like this, see src/signer.py (lines 31-37).",
            response_metadata={"finish_reason": self.state.finish_reason},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------- telling the causes apart ----------


def test_a_daily_limit_is_recognised() -> None:
    assert classify_quota(RuntimeError(DAILY_ERROR)) == DAILY


def test_a_per_minute_limit_is_recognised() -> None:
    assert classify_quota(RuntimeError(MINUTE_ERROR)) == PER_MINUTE


def test_a_payload_too_large_is_recognised() -> None:
    assert classify_quota(RuntimeError(TOO_LARGE_ERROR)) == TOO_LARGE


def test_a_daily_limit_is_not_mistaken_for_a_plain_rate_limit() -> None:
    """Both are 429s; the daily one must win, because the advice differs."""
    assert classify_quota(RuntimeError(DAILY_ERROR)) != PER_MINUTE


def test_an_unrelated_failure_is_not_a_quota_problem() -> None:
    assert classify_quota(RuntimeError("connection reset by peer")) is None
    assert classify_quota(ValueError("bad json")) is None


def test_a_status_code_is_enough_without_a_message() -> None:
    error = RuntimeError("refused")
    error.status_code = 429
    assert classify_quota(error) == PER_MINUTE


# ---------- the wording ----------


def test_the_per_minute_message_says_to_wait_a_minute() -> None:
    assert "Wait a minute" in MESSAGES[PER_MINUTE]
    assert "busy" in MESSAGES[PER_MINUTE]


def test_the_daily_message_says_to_try_again_later() -> None:
    assert "try again later" in MESSAGES[DAILY]
    assert "today" in MESSAGES[DAILY]


def test_the_two_limits_are_worded_differently() -> None:
    assert MESSAGES[DAILY] != MESSAGES[PER_MINUTE]
    assert "Wait a minute" not in MESSAGES[DAILY]


def test_every_message_is_short_and_plain() -> None:
    for message in MESSAGES.values():
        assert len(message) < 130, message
        assert "429" not in message and "413" not in message, "no provider codes in visitor copy"
        assert "Error code" not in message


# ---------- the file list sits under the message, not in place of it ----------


def test_the_files_come_after_the_message_under_a_heading() -> None:
    answer = quota_answer(PER_MINUTE, SNIPPETS)
    assert answer.startswith(MESSAGES[PER_MINUTE])
    assert FILES_HEADING in answer
    assert answer.index(MESSAGES[PER_MINUTE]) < answer.index(FILES_HEADING)
    assert "src/signer.py (lines 31-37)" in answer
    assert "src/exc.py (lines 1-5)" in answer


def test_the_bare_list_heading_is_gone_from_a_quota_answer() -> None:
    """'Top relevant locations:' read as though the list were the answer."""
    assert "Top relevant locations:" not in quota_answer(DAILY, SNIPPETS)


def test_no_files_is_said_plainly() -> None:
    answer = quota_answer(DAILY, [])
    assert MESSAGES[DAILY] in answer
    assert NO_FILES in answer
    assert FILES_HEADING not in answer


# ---------- through the answer service ----------


def answer_service_with(error: Exception):
    from test_retrieval_snippets import RecordingModel, function_record, service

    return error, RecordingModel, function_record, service


@pytest.mark.parametrize(
    "error_text,kind",
    [(DAILY_ERROR, DAILY), (MINUTE_ERROR, PER_MINUTE), (TOO_LARGE_ERROR, TOO_LARGE)],
)
def test_the_answer_service_reports_the_right_cause(tmp_path, error_text: str, kind: str) -> None:
    from test_retrieval_snippets import RecordingModel, function_record, service

    path = tmp_path / "core.py"
    path.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    model = RecordingModel()
    model.state.error = RuntimeError(error_text)

    result = service([function_record(path, "add", 1, 2)], llm=model).answer("r", "q")

    assert MESSAGES[kind] in result.answer
    assert FILES_HEADING in result.answer
    assert result.citations, "the citations still describe the code that was read"


# ---------- through the orchestrator, which is what the visitor hits ----------


class QuotaRetrieval(FakeAgent):
    def __init__(self) -> None:
        super().__init__("retrieval")

    def retrieve(self, question, repo_id, top_k=5):
        return GroundedAnswer(
            answer="found it", citations=[s.location for s in SNIPPETS], reasoning_steps=[], snippets=list(SNIPPETS)
        )


def orchestrator_with(mentor) -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=FakeAgent("planner", plan_of(("retrieval", "find"), ("mentor", "explain"))),
        retrieval_agent=QuotaRetrieval(),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=mentor,
        memory_agent=FakeAgent("memory"),
    )


@pytest.mark.parametrize("error_text,kind", [(DAILY_ERROR, DAILY), (MINUTE_ERROR, PER_MINUTE)])
def test_the_fast_path_shows_the_message_not_the_exception(error_text: str, kind: str) -> None:
    mentor = CodingMentorAgent(llm=FailingModel(RuntimeError(error_text)))
    result = orchestrator_with(mentor).handle_question_fast("q", "repo")

    assert MESSAGES[kind] in result.answer
    assert FILES_HEADING in result.answer
    assert ANSWER_ERROR_PREFIX not in result.answer
    assert "Error code" not in result.answer
    assert "429" not in result.answer


def test_a_planned_mentor_step_shows_the_message_too() -> None:
    mentor = CodingMentorAgent(llm=FailingModel(RuntimeError(DAILY_ERROR)))
    result = orchestrator_with(mentor).handle_question("q", "repo")

    assert MESSAGES[DAILY] in result.answer
    assert "Error executing mentor" not in result.answer


def test_a_failure_that_is_not_a_quota_problem_still_reports_itself() -> None:
    mentor = CodingMentorAgent(llm=FailingModel(RuntimeError("connection reset")))
    result = orchestrator_with(mentor).handle_question_fast("q", "repo")

    assert ANSWER_ERROR_PREFIX in result.answer
    assert FILES_HEADING not in result.answer


def test_the_streamed_endpoint_sends_the_message_as_the_answer(tmp_path) -> None:
    from test_analyze_indexing import Env

    from codeatlas.app.di import get_agent_orchestrator, get_llm_provider

    env = Env(tmp_path)
    repo_id = env.analyze().json()["repository_id"]
    mentor = CodingMentorAgent(llm=FailingModel(RuntimeError(DAILY_ERROR)))
    env.client.app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator_with(mentor)
    env.client.app.dependency_overrides[get_llm_provider] = lambda: None

    resp = env.client.post("/ask/stream", json={"question": "q", "repo_id": repo_id})
    events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
    answer = "".join(e["content"] for e in events if e["type"] == "token")

    assert MESSAGES[DAILY] in answer
    assert FILES_HEADING in answer
    assert "Error code" not in answer


# ---------- the truncation note ----------


def test_an_answer_cut_off_by_the_output_limit_says_so() -> None:
    answer = CodingMentorAgent(llm=FailingModel(finish_reason="length")).answer("q", SNIPPETS)
    assert answer.endswith(TRUNCATED_NOTE)
    assert "cut off" in answer


def test_a_complete_answer_gets_no_note() -> None:
    answer = CodingMentorAgent(llm=FailingModel(finish_reason="stop")).answer("q", SNIPPETS)
    assert TRUNCATED_NOTE not in answer
    assert answer.endswith(".")


def test_the_note_is_added_after_the_answer_not_instead_of_it() -> None:
    answer = CodingMentorAgent(llm=FailingModel(finish_reason="length")).answer("q", SNIPPETS)
    assert "src/signer.py (lines 31-37)" in answer, "the real answer is still there"


def test_truncation_is_read_from_the_providers_own_field() -> None:
    assert was_truncated(AIMessage(content="x", response_metadata={"finish_reason": "length"}))
    assert not was_truncated(AIMessage(content="x", response_metadata={"finish_reason": "stop"}))
    assert not was_truncated(AIMessage(content="x"))


def test_truncation_reported_per_generation_is_also_seen() -> None:
    message = AIMessage(content="x", response_metadata={"generation_info": {"finish_reason": "length"}})
    assert was_truncated(message)
