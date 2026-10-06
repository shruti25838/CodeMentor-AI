"""Answers structural questions from the call graph, never from the model's memory.

The facts in the answer are computed by `call_graph`, which reads the syntax tree. The model
is given those facts and asked only to phrase them; it is never asked what calls what. If no
model is configured the facts are returned as a plain list, which is still a true answer.
"""

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.observability.timing import stage
from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.analysis.call_graph import CallGraph
from codeatlas.services.state.repo_state_store import RepoStateStore

NO_STATE = "Error: Repository state not found. Please analyze the repo first."

_PHRASE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a software architect. Below are facts about a repository, computed "
            "exactly from its syntax tree. Restate them as a short, clear answer to the "
            "question.\n"
            "Use only these facts. Do not add call relationships, file names or line numbers "
            "that are not listed. If the facts are empty, say that nothing matched.",
        ),
        ("human", "Question: {question}\n\nFacts:\n{facts}"),
    ]
)

# Each pattern pulls the subject out of a question; the graph answers it.
_WHO_CALLS = re.compile(r"\bwho\s+(?:calls|uses|invokes)\s+(?P<name>[\w.]+)", re.I)
_WHAT_CALLS = re.compile(r"\bwhat\s+does\s+(?P<name>[\w.]+)\s+call", re.I)
_WHERE_DEFINED = re.compile(r"\bwhere\s+is\s+(?P<name>[\w.]+)\s+(?:defined|declared)", re.I)
_WHO_IMPORTS = re.compile(r"\bwho\s+imports\s+(?P<name>[\w.]+)", re.I)
_WHAT_IMPORTS = re.compile(r"\bwhat\s+does\s+(?P<name>[\w.]+)\s+import", re.I)
_CIRCULAR = re.compile(r"\bcircular\s+imports?\b|\bimport\s+cycles?\b", re.I)

MAX_ROWS = 40


class RepoAnalystAgent(Agent):
    def __init__(
        self,
        state_store: RepoStateStore,
        llm: BaseChatModel | None = None,
        graphs: dict[str, CallGraph] | None = None,
    ) -> None:
        self._state_store = state_store
        self._llm = llm
        # One graph per repository, built on first use and kept for later questions.
        self._graphs = graphs if graphs is not None else {}
        self._logger = logging.getLogger(__name__)

    def graph_for(self, repo_id: str) -> CallGraph | None:
        if repo_id in self._graphs:
            return self._graphs[repo_id]
        state = self._state_store.get(repo_id)
        if state is None or not state.root_path:
            return None
        with stage("call_graph"):
            graph = CallGraph.build(state.root_path)
        self._logger.info(
            "Built call graph for %s: %d definitions, %d calls", repo_id, len(graph.definitions), len(graph.calls)
        )
        self._graphs[repo_id] = graph
        return graph

    def facts(self, question: str, repo_id: str) -> str:
        """The structural answer, as lines of fact. No model call happens here."""
        graph = self.graph_for(repo_id)
        if graph is None:
            return NO_STATE
        return answer_from_graph(graph, question)

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        if not repo_id:
            return "Error: repo_id is required for analysis."
        facts = self.facts(prompt, repo_id)
        if self._llm is None or facts == NO_STATE:
            return facts
        try:
            with stage("llm"):
                response = (_PHRASE | self._llm).invoke({"question": prompt, "facts": facts})
        except Exception as exc:
            # The facts are the answer; phrasing is a convenience.
            self._logger.warning("Phrasing the analysis failed (%s); returning the facts", type(exc).__name__)
            return facts
        return str(response.content)


def answer_from_graph(graph: CallGraph, question: str) -> str:
    """Facts for a structural question, or a summary when the question matches no pattern."""
    if _CIRCULAR.search(question):
        return _circular(graph)

    for pattern, handler in (
        (_WHO_CALLS, _who_calls),
        (_WHAT_CALLS, _what_calls),
        (_WHERE_DEFINED, _where_defined),
        (_WHO_IMPORTS, _who_imports),
        (_WHAT_IMPORTS, _what_imports),
    ):
        match = pattern.search(question)
        if match:
            return handler(graph, match.group("name"))

    return _summary(graph)


def _who_calls(graph: CallGraph, name: str) -> str:
    calls = graph.who_calls(name)
    if not calls:
        where = graph.where_is_it_defined(name)
        if not where:
            return f"No definition of {name} was found in this repository."
        return f"Nothing in this repository calls {name} (defined at {where[0].location})."
    rows = sorted({f"{c.caller} at {c.path}:{c.line}" for c in calls})
    return _list(f"{name} is called from {len(rows)} place(s):", rows)


def _what_calls(graph: CallGraph, name: str) -> str:
    calls = graph.what_does_it_call(name)
    if not calls:
        if not graph.where_is_it_defined(name):
            return f"No definition of {name} was found in this repository."
        return f"{name} calls nothing."
    resolved = sorted({f"{c.callee} at {c.path}:{c.line}" for c in calls if c.resolved})
    unresolved = sorted({f"{c.callee_text} at {c.path}:{c.line}" for c in calls if not c.resolved})
    parts = []
    if resolved:
        parts.append(_list(f"{name} calls {len(resolved)} thing(s) in this repository:", resolved))
    if unresolved:
        parts.append(
            _list(
                f"{len(unresolved)} call(s) from {name} are outside this repository or could not be resolved:",
                unresolved,
            )
        )
    return "\n\n".join(parts)


def _where_defined(graph: CallGraph, name: str) -> str:
    found = graph.where_is_it_defined(name)
    if not found:
        return f"No definition of {name} was found in this repository."
    rows = [f"{d.kind} {d.qualname} at {d.path}:{d.start_line}-{d.end_line}" for d in found]
    return _list(f"{name} is defined in {len(rows)} place(s):", rows)


def _who_imports(graph: CallGraph, name: str) -> str:
    edges = graph.who_imports(name)
    if not edges:
        return f"Nothing in this repository imports {name}."
    rows = sorted({f"{e.module} at {e.path}:{e.line}" for e in edges})
    return _list(f"{name} is imported by {len(rows)} module(s):", rows)


def _what_imports(graph: CallGraph, name: str) -> str:
    edges = graph.what_does_it_import(name)
    if not edges:
        return f"No module named {name} was found in this repository."
    internal = sorted({f"{e.resolved_module} at {e.path}:{e.line}" for e in edges if e.internal})
    external = sorted({e.target for e in edges if not e.internal})
    parts = []
    if internal:
        parts.append(_list(f"{name} imports {len(internal)} module(s) in this repository:", internal))
    if external:
        parts.append(_list(f"{name} also imports {len(external)} module(s) from outside:", external))
    return "\n\n".join(parts) if parts else f"{name} imports nothing."


def _circular(graph: CallGraph) -> str:
    groups = graph.circular_imports()
    if not groups:
        return "There are no circular imports between modules of this repository."
    rows = []
    for group in groups:
        example = graph.shortest_import_cycle(group[0])
        arrow = " -> ".join(example) if example else ", ".join(group)
        rows.append(f"{len(group)} modules: {', '.join(group)} (for example {arrow})")
    return _list(f"{len(groups)} group(s) of modules import each other:", rows)


def _summary(graph: CallGraph) -> str:
    resolved, total = graph.resolution_rate()
    busiest = sorted(
        ((len(graph.who_calls(d.qualname)), d) for d in list(graph.definitions.values())[:400]),
        key=lambda pair: -pair[0],
    )[:5]
    rows = [
        f"{len(graph.modules)} Python modules, {len(graph.definitions)} definitions, {total} call sites.",
        f"{resolved} call sites resolve to a definition in this repository; the rest are external or dynamic.",
        f"{len(graph.circular_imports())} group(s) of modules import each other.",
    ]
    rows += [f"{d.qualname} is called from {n} place(s)" for n, d in busiest if n]
    return _list("Repository structure:", rows)


def _list(heading: str, rows: list[str]) -> str:
    shown = rows[:MAX_ROWS]
    text = "\n".join(f"- {row}" for row in shown)
    if len(rows) > MAX_ROWS:
        text += f"\n- ... and {len(rows) - MAX_ROWS} more"
    return f"{heading}\n{text}"
