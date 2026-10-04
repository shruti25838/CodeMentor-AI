import json
import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_analyze_indexing import Env

from codeatlas.app.di import get_agent_orchestrator, get_conversation_store, get_llm_provider
from codeatlas.services.agents.coding_mentor_agent import CodingMentorAgent
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.agents.retrieval_agent import RetrievalAgent
from codeatlas.services.memory.conversation import ConversationStore, Turn, estimate_tokens
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.qa.followup import needs_rewrite, rewrite_question

SESSION_A = "a" * 32
SESSION_B = "b" * 32
REWRITTEN = "How does the add function in pkg/core.py handle negative numbers?"
ANSWER_SYSTEM = "You are a senior engineer. Use provided code snippets"
MENTOR_HISTORY = "Earlier in this conversation:"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordingModel(BaseChatModel):
    """Records every prompt; answers a rewrite request with REWRITTEN and anything else with a numbered answer."""

    # Plain object, so the test can change it after the model is built.
    state: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("State", (), {"prompts": [], "fail": False})()

    @property
    def prompts(self) -> list[str]:
        return self.state.prompts

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = "\n".join(str(m.content) for m in messages)
        self.state.prompts.append(text)
        if self.state.fail:
            raise RuntimeError("model down")
        content = REWRITTEN if "Rewrite the user's last question" in text else f"answer-{len(self.state.prompts)}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def prompts_containing(self, marker: str) -> list[str]:
        return [p for p in self.prompts if marker in p]


class Chat:
    def __init__(self, tmp_path, store: ConversationStore | None = None) -> None:
        self.env = Env(tmp_path)
        self.llm = RecordingModel()
        self.store = store or ConversationStore()
        answer_service = AnswerService(
            retriever=self.env.index._retriever, embedder=self.env.index._embedder, llm=self.llm
        )
        mentor = CodingMentorAgent(answer_service=answer_service, llm=self.llm)
        orchestrator = AgentOrchestrator(
            planner=mentor,
            retrieval_agent=RetrievalAgent(answer_service=answer_service),
            analyst_agent=mentor,
            mentor_agent=mentor,
            memory_agent=mentor,
        )
        overrides = self.env.client.app.dependency_overrides
        overrides[get_agent_orchestrator] = lambda: orchestrator
        overrides[get_llm_provider] = lambda: type("P", (), {"get_chat_model": lambda _self: self.llm})()
        overrides[get_conversation_store] = lambda: self.store
        self.repo_id = self.env.analyze().json()["repository_id"]

    def ask(self, question: str, session_id: str | None = SESSION_A, repo: bool = True):
        body = {"question": question, "repo_id": self.repo_id if repo else None, "session_id": session_id}
        resp = self.env.client.post("/ask/stream", json=body)
        assert resp.status_code == 200, resp.text
        events = [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["type"] == "done", events[-1]
        return events[-1]


# ---------- the store ----------


def test_keeps_only_the_last_turns():
    store = ConversationStore(max_turns=2)
    for i in range(4):
        store.add(SESSION_A, "r", f"q{i}", f"a{i}")
    assert store.history(SESSION_A, "r") == [Turn("q2", "a2"), Turn("q3", "a3")]


def test_history_is_capped_by_tokens_dropping_the_oldest_first():
    store = ConversationStore(max_turns=10, max_tokens=60)
    for i in range(5):
        store.add(SESSION_A, "r", f"question {i}", "x" * 80)  # about 23 tokens per turn
    history = store.history(SESSION_A, "r")
    assert [t.question for t in history] == ["question 3", "question 4"]
    assert sum(estimate_tokens(t.question) + estimate_tokens(t.answer) for t in history) <= 60


def test_a_single_turn_over_the_cap_is_shortened_not_dropped():
    store = ConversationStore(max_tokens=50)
    store.add(SESSION_A, "r", "why?", "y" * 1000)
    [turn] = store.history(SESSION_A, "r")
    assert turn.question == "why?"
    assert estimate_tokens(turn.question) + estimate_tokens(turn.answer) <= 52  # plus the " ..." marker


def test_sessions_expire_after_the_idle_time():
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=60, clock=clock)
    store.add(SESSION_A, "r", "q", "a")
    clock.now += 59
    assert store.history(SESSION_A, "r") == [Turn("q", "a")]
    store.add(SESSION_A, "r", "q2", "a2")  # a new message restarts the idle time
    clock.now += 59
    assert len(store.history(SESSION_A, "r")) == 2
    clock.now += 2
    assert store.history(SESSION_A, "r") == []
    assert store.session_count() == 0


def test_sessions_never_see_each_other_or_other_repos():
    store = ConversationStore()
    store.add(SESSION_A, "r1", "secret-a", "answer-a")
    assert store.history(SESSION_B, "r1") == []
    assert store.history(SESSION_A, "r2") == []
    assert store.history(SESSION_A, None) == []


def test_session_count_is_capped_least_recently_used_first():
    store = ConversationStore(max_sessions=2)
    store.add("s1" * 8, "r", "q", "a")
    store.add("s2" * 8, "r", "q", "a")
    store.history("s1" * 8, "r")
    store.add("s3" * 8, "r", "q", "a")
    assert store.session_count() == 2
    assert store.history("s1" * 8, "r") == []  # reads do not count as use; s1 was added first


def test_raw_session_ids_are_not_kept():
    store = ConversationStore()
    store.add(SESSION_A, "r", "q", "a")
    assert SESSION_A not in repr(store._sessions)


# ---------- deciding when to rewrite ----------


@pytest.mark.parametrize(
    "question",
    [
        "why?",
        "and the tests?",
        "What about URLSafeSerializer?",
        "Can you show an example of it?",
        "Where is that defined?",
    ],
)
def test_follow_ups_are_rewritten(question):
    assert needs_rewrite(question, [Turn("How does add work?", "It adds.")])


@pytest.mark.parametrize(
    "question",
    ["How does Signer create and check a signature?", "What does this repo do?", "Where is the config loaded from?"],
)
def test_standalone_questions_are_not_rewritten(question):
    assert not needs_rewrite(question, [Turn("How does add work?", "It adds.")])


def test_nothing_is_rewritten_without_history():
    assert not needs_rewrite("why?", [])


def test_rewrite_falls_back_to_the_question_when_the_model_fails():
    model = RecordingModel()
    model.state.fail = True
    assert rewrite_question(model, "why?", [Turn("q", "a")]) == "why?"


# ---------- through /ask/stream ----------


def test_earlier_turns_reach_the_prompt(tmp_path):
    chat = Chat(tmp_path)
    chat.ask("How does add work in pkg/core.py?")
    first_answer = chat.store.history(SESSION_A, chat.repo_id)[0].answer
    done = chat.ask("Where is sub defined?")
    [mentor_prompt] = chat.llm.prompts_containing(MENTOR_HISTORY)
    assert "User: How does add work in pkg/core.py?" in mentor_prompt
    assert f"Assistant: {first_answer}" in mentor_prompt
    assert "Used 1 earlier turn(s) of this conversation." in done["reasoning_steps"]


def test_another_session_never_gets_this_history(tmp_path):
    chat = Chat(tmp_path)
    chat.ask("secret-question-from-a about add")
    chat.llm.prompts.clear()
    chat.ask("Where is sub defined?", session_id=SESSION_B)
    assert not any("secret-question-from-a" in p for p in chat.llm.prompts)
    assert chat.llm.prompts_containing(MENTOR_HISTORY) == []


def test_follow_up_is_rewritten_once_and_used_for_search(tmp_path):
    chat = Chat(tmp_path)
    chat.ask("How does add work in pkg/core.py?")
    chat.llm.prompts.clear()
    done = chat.ask("what about negative numbers?")
    assert len(chat.llm.prompts_containing("Rewrite the user's last question")) == 1
    # The retrieval summary searched with the rewritten question...
    assert any(REWRITTEN in p for p in chat.llm.prompts_containing(ANSWER_SYSTEM))
    # ...while the mentor still answers the question as asked, with the history.
    [mentor_prompt] = chat.llm.prompts_containing(MENTOR_HISTORY)
    assert "Goal: what about negative numbers?" in mentor_prompt
    assert done["timings_ms"]["llm_calls"] == 3
    assert "rewrite" in done["timings_ms"]


def test_standalone_or_first_questions_make_no_extra_model_call(tmp_path):
    chat = Chat(tmp_path)
    first = chat.ask("How does add work in pkg/core.py?")
    second = chat.ask("Where is sub defined in pkg/core.py?")
    assert chat.llm.prompts_containing("Rewrite the user's last question") == []
    assert "rewrite" not in first["timings_ms"] and "rewrite" not in second["timings_ms"]


def test_without_a_session_id_nothing_is_kept(tmp_path):
    chat = Chat(tmp_path)
    chat.ask("How does add work?", session_id=None)
    chat.ask("why?", session_id=None)
    assert chat.store.session_count() == 0
    assert chat.llm.prompts_containing("Rewrite the user's last question") == []


@pytest.mark.parametrize("bad", ["short", "has spaces in it 1234567", "x" * 129, "../../etc/passwd/aaaaaa"])
def test_malformed_session_ids_are_rejected(tmp_path, bad):
    chat = Chat(tmp_path)
    resp = chat.env.client.post("/ask/stream", json={"question": "q", "repo_id": chat.repo_id, "session_id": bad})
    assert resp.status_code == 422


def test_general_mode_keeps_history_as_messages(tmp_path):
    chat = Chat(tmp_path)
    chat.ask("What is a closure?", repo=False)
    chat.ask("Show me one in Python.", repo=False)
    assert "What is a closure?" in chat.llm.prompts[-1]
    assert chat.store.history(SESSION_A, chat.repo_id) == []  # general chat is kept apart from repo chat


def test_failed_answers_are_not_kept(tmp_path):
    chat = Chat(tmp_path)
    chat.llm.state.fail = True
    chat.ask("How does add work?")
    assert chat.store.history(SESSION_A, chat.repo_id) == []


def test_conversation_content_is_never_logged_or_written_to_disk(tmp_path, caplog):
    chat = Chat(tmp_path)
    with caplog.at_level(logging.DEBUG):
        chat.ask("secret-first-question about add")
        chat.ask("what about secret-follow-up?")
        chat.ask("secret-general-question", repo=False)
    answers = [t.answer for t in chat.store.history(SESSION_A, chat.repo_id)]
    for text in ["secret-first-question", "secret-follow-up", "secret-general-question", REWRITTEN, *answers]:
        assert text not in caplog.text
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert text.encode() not in path.read_bytes(), path
