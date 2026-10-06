"""The mentor answers from the snippets it was handed, and cites only those.

Before this, the mentor ran its own search inside `run`, so the code it read was not the
code the user saw as citations: on one real question the answer described `signer.py` while
the citations named `exc.py`, `timed.py` and `url_safe.py`, and the model invented line
numbers for a file it had never been shown.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_agent_pipeline import FakeAgent

from codeatlas.services.agents.coding_mentor_agent import NO_CONTEXT, CodingMentorAgent
from codeatlas.services.agents.orchestration import AgentOrchestrator
from codeatlas.services.retrieval.snippets import Snippet

SNIPPETS = [
    Snippet("src/signer.py", 31, 37, "def sign(self, value):\n    return value + self.get_signature(value)"),
    Snippet("src/signer.py", 40, 44, "def get_signature(self, value):\n    return hmac.new(...).digest()"),
]


class SpyModel(BaseChatModel):
    """Records prompts and replies with a fixed answer."""

    state: Any = None

    def __init__(self, reply: str = "answer", **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"prompts": [], "reply": reply})()

    @property
    def _llm_type(self) -> str:
        return "spy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.state.prompts.append("\n".join(str(m.content) for m in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.state.reply))])

    @property
    def prompt(self) -> str:
        return self.state.prompts[-1]


class ExplodingAnswerService:
    """Any search from inside the mentor is a bug; this makes it loud."""

    def answer(self, *args, **kwargs):
        raise AssertionError("the mentor must not run its own search")


def test_mentor_does_not_search_when_given_snippets() -> None:
    model = SpyModel()
    mentor = CodingMentorAgent(llm=model, answer_service=ExplodingAnswerService())
    out = mentor.answer("How does sign work?", SNIPPETS)
    assert out == "answer"


def test_the_prompt_contains_the_code_it_must_answer_from() -> None:
    model = SpyModel()
    CodingMentorAgent(llm=model).answer("How does sign work?", SNIPPETS)

    assert "def sign(self, value):" in model.prompt
    assert "hmac.new(...).digest()" in model.prompt


def test_the_prompt_lists_exactly_the_allowed_citations() -> None:
    model = SpyModel()
    CodingMentorAgent(llm=model).answer("q", SNIPPETS)

    assert "src/signer.py (lines 31-37)" in model.prompt
    assert "src/signer.py (lines 40-44)" in model.prompt
    assert "Cite only snippets listed below" in model.prompt
    assert "never invent a line number" in model.prompt


def test_snippets_are_numbered_so_the_model_can_refer_to_them() -> None:
    model = SpyModel()
    CodingMentorAgent(llm=model).answer("q", SNIPPETS)
    assert "[1] src/signer.py (lines 31-37)" in model.prompt
    assert "[2] src/signer.py (lines 40-44)" in model.prompt


def test_no_snippets_and_no_facts_says_so_instead_of_calling_the_model() -> None:
    model = SpyModel()
    out = CodingMentorAgent(llm=model).answer("q", [])

    assert out == NO_CONTEXT
    assert model.state.prompts == [], "no model call when there is nothing to answer from"


def test_structural_facts_reach_the_prompt_and_are_marked_exact() -> None:
    model = SpyModel()
    CodingMentorAgent(llm=model).answer("who calls sign?", [], facts="sign is called by Serializer.dumps")

    assert "sign is called by Serializer.dumps" in model.prompt
    assert "exact" in model.prompt


def test_facts_alone_are_enough_to_answer() -> None:
    model = SpyModel()
    out = CodingMentorAgent(llm=model).answer("who calls sign?", [], facts="called by dumps")
    assert out == "answer"


def test_history_reaches_the_prompt() -> None:
    model = SpyModel()
    CodingMentorAgent(llm=model).answer("and negative numbers?", SNIPPETS, history="User: hi\nAssistant: hello")
    assert "Earlier in this conversation:" in model.prompt
    assert "Assistant: hello" in model.prompt


def test_run_without_a_repo_id_is_an_error() -> None:
    assert "repo_id is required" in CodingMentorAgent(llm=SpyModel()).run("q", None)


# ---------- through the orchestrator ----------


class SnippetRetrieval(FakeAgent):
    """A retrieval agent with the structured entry point the orchestrator prefers."""

    def __init__(self) -> None:
        super().__init__("retrieval")

    def retrieve(self, question, repo_id, top_k=5):
        from codeatlas.services.qa.answer_service import GroundedAnswer

        return GroundedAnswer(
            answer="found the signer",
            citations=[s.location for s in SNIPPETS],
            reasoning_steps=[],
            snippets=list(SNIPPETS),
        )


def test_fast_path_searches_once_and_the_mentor_sees_those_snippets() -> None:
    model = SpyModel()
    mentor = CodingMentorAgent(llm=model, answer_service=ExplodingAnswerService())
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner"),
        retrieval_agent=SnippetRetrieval(),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=mentor,
        memory_agent=FakeAgent("memory"),
    )
    result = orchestrator.handle_question_fast("How does sign work?", "repo")

    assert result.answer == "answer"
    assert result.citations == [s.location for s in SNIPPETS]
    # The code behind every citation was in the mentor's prompt.
    for snippet in SNIPPETS:
        assert snippet.location in model.prompt


def test_cited_files_and_answered_files_come_from_one_search() -> None:
    """The regression: citations and answer used to come from two different searches."""
    model = SpyModel()
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner"),
        retrieval_agent=SnippetRetrieval(),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=CodingMentorAgent(llm=model, answer_service=ExplodingAnswerService()),
        memory_agent=FakeAgent("memory"),
    )
    result = orchestrator.handle_question_fast("q", "repo")

    cited_files = {c.split(" (")[0] for c in result.citations}
    prompt_files = {s.path for s in SNIPPETS if s.location in model.prompt}
    assert cited_files == prompt_files


def test_planned_mentor_step_uses_the_retrieved_snippets() -> None:
    import json

    model = SpyModel()
    plan = json.dumps(
        {"steps": [{"agent": "retrieval", "instruction": "find sign"}, {"agent": "mentor", "instruction": "explain"}]}
    )
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan),
        retrieval_agent=SnippetRetrieval(),
        analyst_agent=FakeAgent("analyst"),
        mentor_agent=CodingMentorAgent(llm=model, answer_service=ExplodingAnswerService()),
        memory_agent=FakeAgent("memory"),
    )
    result = orchestrator.handle_question("How does sign work?", "repo")

    assert result.answer == "answer"
    assert "def sign(self, value):" in model.prompt
    assert result.citations == [s.location for s in SNIPPETS]


def test_analyst_output_reaches_the_mentor_as_structural_facts() -> None:
    import json

    model = SpyModel()
    plan = json.dumps(
        {
            "steps": [
                {"agent": "analyst", "instruction": "who calls sign"},
                {"agent": "mentor", "instruction": "explain"},
            ]
        }
    )
    orchestrator = AgentOrchestrator(
        planner=FakeAgent("planner", plan),
        retrieval_agent=SnippetRetrieval(),
        analyst_agent=FakeAgent("analyst", "sign is called by Serializer.dumps at src/serializer.py:309"),
        mentor_agent=CodingMentorAgent(llm=model, answer_service=ExplodingAnswerService()),
        memory_agent=FakeAgent("memory"),
    )
    orchestrator.handle_question("who calls sign?", "repo")

    assert "sign is called by Serializer.dumps at src/serializer.py:309" in model.prompt
    assert "Structural facts" in model.prompt
