"""The analyst answers structural questions from the graph; the model only phrases them."""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from codeatlas.services.agents.repo_analyst_agent import NO_STATE, RepoAnalystAgent, answer_from_graph
from codeatlas.services.analysis.call_graph import CallGraph

SOURCES = {
    "mypkg/__init__.py": "from .core import Engine\n",
    "mypkg/util.py": "def helper(x):\n    return x\n",
    "mypkg/core.py": (
        "from .util import helper\n"
        "\n"
        "class Engine:\n"
        "    def run(self, v):\n"
        "        return helper(v)\n"
        "\n"
        "def make():\n"
        "    return Engine()\n"
    ),
}


class SpyModel(BaseChatModel):
    state: Any = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = type("S", (), {"prompts": [], "error": None})()

    @property
    def _llm_type(self) -> str:
        return "spy"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.state.prompts.append("\n".join(str(m.content) for m in messages))
        if self.state.error:
            raise self.state.error
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="phrased answer"))])


class StubStore:
    def __init__(self, root: str | None) -> None:
        self._root = root

    def get(self, repo_id):
        if self._root is None:
            return None
        return type("S", (), {"root_path": self._root})()


@pytest.fixture
def graph(tmp_path):
    for name, text in SOURCES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return CallGraph.build(tmp_path)


@pytest.fixture
def agent(tmp_path, graph):
    return RepoAnalystAgent(state_store=StubStore(str(tmp_path)), llm=None)


# ---------- the six question shapes ----------


def test_who_calls(graph) -> None:
    out = answer_from_graph(graph, "who calls helper?")
    assert "mypkg.core.Engine.run" in out
    assert "mypkg/core.py:5" in out


def test_what_does_x_call(graph) -> None:
    out = answer_from_graph(graph, "what does Engine.run call?")
    assert "mypkg.util.helper" in out


def test_where_is_x_defined(graph) -> None:
    out = answer_from_graph(graph, "where is Engine defined?")
    assert "mypkg/core.py:3" in out
    assert "class" in out


def test_who_imports_x(graph) -> None:
    out = answer_from_graph(graph, "who imports mypkg.util?")
    assert "mypkg.core" in out


def test_what_does_x_import(graph) -> None:
    out = answer_from_graph(graph, "what does mypkg.core import?")
    assert "mypkg.util" in out


def test_circular_imports_when_there_are_none(graph) -> None:
    assert "no circular imports" in answer_from_graph(graph, "are there circular imports?").lower()


def test_circular_imports_when_there_are_some(tmp_path) -> None:
    for name, text in {
        "p/__init__.py": "",
        "p/a.py": "from .b import x\n",
        "p/b.py": "from .a import y\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    out = answer_from_graph(CallGraph.build(tmp_path), "any import cycles?")
    assert "p.a" in out and "p.b" in out
    assert "->" in out, "an example loop is shown"


def test_an_unmatched_question_gives_a_summary(graph) -> None:
    out = answer_from_graph(graph, "tell me about this project")
    assert "Repository structure:" in out
    assert "call sites" in out


def test_an_unknown_name_is_reported_honestly(graph) -> None:
    assert "No definition of nosuchthing" in answer_from_graph(graph, "who calls nosuchthing?")


def test_a_defined_but_uncalled_function_says_so(graph) -> None:
    assert "Nothing in this repository calls" in answer_from_graph(graph, "who calls make?")


# ---------- the model phrases, it does not compute ----------


def test_without_a_model_the_facts_are_the_answer(agent) -> None:
    out = agent.run("who calls helper?", "repo")
    assert "mypkg.core.Engine.run" in out


def test_the_model_receives_the_facts_and_is_told_to_use_only_them(tmp_path, graph) -> None:
    model = SpyModel()
    agent = RepoAnalystAgent(state_store=StubStore(str(tmp_path)), llm=model)
    out = agent.run("who calls helper?", "repo")

    assert out == "phrased answer"
    prompt = model.state.prompts[0]
    assert "mypkg.core.Engine.run" in prompt, "the computed facts are in the prompt"
    assert "Do not add call relationships" in prompt
    assert "Use only these facts" in prompt


def test_facts_are_computed_without_any_model_call(tmp_path, graph) -> None:
    model = SpyModel()
    agent = RepoAnalystAgent(state_store=StubStore(str(tmp_path)), llm=model)
    agent.facts("who calls helper?", "repo")
    assert model.state.prompts == [], "computing facts must not call the model"


def test_a_failing_model_falls_back_to_the_facts(tmp_path, graph) -> None:
    model = SpyModel()
    model.state.error = RuntimeError("429 rate limited")
    agent = RepoAnalystAgent(state_store=StubStore(str(tmp_path)), llm=model)

    out = agent.run("who calls helper?", "repo")

    assert "mypkg.core.Engine.run" in out


# ---------- wiring ----------


def test_no_repo_state_is_reported(tmp_path) -> None:
    agent = RepoAnalystAgent(state_store=StubStore(None), llm=None)
    assert agent.run("who calls helper?", "repo") == NO_STATE


def test_no_repo_id_is_an_error(tmp_path) -> None:
    assert "repo_id is required" in RepoAnalystAgent(state_store=StubStore(None)).run("q", None)


def test_the_graph_is_built_once_per_repository(tmp_path, graph) -> None:
    agent = RepoAnalystAgent(state_store=StubStore(str(tmp_path)), llm=None)
    first = agent.graph_for("repo")
    second = agent.graph_for("repo")
    assert first is second
