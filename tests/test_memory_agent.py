"""The memory agent reads the same per-session conversation memory the website chat uses.

It used to have its own JSON file on disk, scoped per repository rather than per visitor,
with no expiry and no size cap, and nothing in the pipeline ever read it to answer anything.
"""

import json
import logging
from pathlib import Path

from test_agent_pipeline import FakeAgent

from codeatlas.services.agents.memory_agent import NO_HISTORY, NO_SESSION, MemoryAgent
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.memory.conversation import ConversationStore

SESSION_A = "a" * 32
SESSION_B = "b" * 32


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_agent_reads_the_store_the_website_writes() -> None:
    store = ConversationStore()
    store.add(SESSION_A, "repo", "How does add work?", "It adds two numbers.")

    out = MemoryAgent(conversations=store).recall(SESSION_A, "repo")

    assert "How does add work?" in out
    assert "It adds two numbers." in out


def test_it_is_the_very_same_store_object_as_the_chat_uses() -> None:
    """The regression: the agent had a JsonMemoryStore while the chat had a ConversationStore."""
    store = ConversationStore()
    agent = MemoryAgent(conversations=store)
    assert agent._conversations is store


def test_no_session_means_no_memory() -> None:
    store = ConversationStore()
    store.add(SESSION_A, "repo", "q", "a")
    assert MemoryAgent(conversations=store).recall(None, "repo") == NO_SESSION


def test_an_empty_conversation_says_so() -> None:
    assert MemoryAgent(conversations=ConversationStore()).recall(SESSION_A, "repo") == NO_HISTORY


# ---------- the caps, expiry and isolation are inherited, not reimplemented ----------


def test_another_session_cannot_be_read() -> None:
    store = ConversationStore()
    store.add(SESSION_A, "repo", "secret question", "secret answer")

    out = MemoryAgent(conversations=store).recall(SESSION_B, "repo")

    assert out == NO_HISTORY
    assert "secret" not in out


def test_the_same_session_on_another_repository_is_separate() -> None:
    store = ConversationStore()
    store.add(SESSION_A, "repo-one", "about one", "answer one")

    assert MemoryAgent(conversations=store).recall(SESSION_A, "repo-two") == NO_HISTORY


def test_the_turn_limit_is_the_chat_s_own() -> None:
    store = ConversationStore(max_turns=2)
    for i in range(5):
        store.add(SESSION_A, "repo", f"question {i}", f"answer {i}")

    out = MemoryAgent(conversations=store).recall(SESSION_A, "repo")

    assert "question 4" in out and "question 3" in out
    assert "question 0" not in out


def test_the_token_cap_is_the_chat_s_own() -> None:
    store = ConversationStore(max_turns=10, max_tokens=40)
    for i in range(10):
        store.add(SESSION_A, "repo", f"q{i}", "answer " * 20)

    out = MemoryAgent(conversations=store).recall(SESSION_A, "repo")

    assert len(out) < 10 * len("answer " * 20)


def test_expiry_is_the_chat_s_own() -> None:
    clock = FakeClock()
    store = ConversationStore(ttl_seconds=100, clock=clock)
    store.add(SESSION_A, "repo", "q", "a")
    agent = MemoryAgent(conversations=store)

    assert "q" in agent.recall(SESSION_A, "repo")
    clock.now += 101
    assert agent.recall(SESSION_A, "repo") == NO_HISTORY


# ---------- privacy ----------


def test_recall_logs_a_count_but_never_the_content(caplog) -> None:
    store = ConversationStore()
    store.add(SESSION_A, "repo", "marker-question-xyzzy", "marker-answer-xyzzy")

    with caplog.at_level(logging.DEBUG):
        MemoryAgent(conversations=store).recall(SESSION_A, "repo")

    assert "xyzzy" not in caplog.text
    assert "Recalled 1 earlier turn(s)" in caplog.text


def test_nothing_is_written_to_disk(tmp_path, monkeypatch) -> None:
    """The old store wrote agent_memory.json under the state directory."""
    monkeypatch.chdir(tmp_path)
    store = ConversationStore()
    store.add(SESSION_A, "repo", "q", "a")
    MemoryAgent(conversations=store).recall(SESSION_A, "repo")

    assert list(Path(tmp_path).rglob("*.json")) == []


def test_the_disk_backed_store_is_gone() -> None:
    for module in (
        "codeatlas.services.memory.json_store",
        "codeatlas.services.memory.in_memory_store",
        "codeatlas.models.agent_memory",
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{module} still exists; it stored visitor content on disk")


# ---------- through the graph ----------


def test_a_planned_memory_step_recalls_this_session() -> None:
    store = ConversationStore()
    store.add(SESSION_A, "repo", "How does add work?", "It adds two numbers.")
    plan = json.dumps({"steps": [{"agent": "memory", "instruction": "what did we discuss?"}]})

    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan),
        retrieval_agent=FakeAgent("retrieval"),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=MemoryAgent(conversations=store),
    )
    result = orchestrator.handle_question("what did we discuss?", "repo", session_id=SESSION_A)

    assert "It adds two numbers." in result.answer


def test_a_memory_step_without_a_session_does_not_fail_the_request() -> None:
    plan = json.dumps({"steps": [{"agent": "memory", "instruction": "recall"}]})
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan),
        retrieval_agent=FakeAgent("retrieval"),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=FakeAgent("mentor"),
        memory_agent=MemoryAgent(conversations=ConversationStore()),
    )
    result = orchestrator.handle_question("recall", "repo")

    assert result.answer == NO_SESSION
