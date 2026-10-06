import logging
from collections.abc import Iterator
from typing import TypedDict

from langgraph.graph import END, StateGraph

from codeatlas.observability.agent_metrics import AGENT_ERROR, agent_run, record_failure, record_plan
from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.agents.plan import Plan, default_plan, parse_plan
from codeatlas.services.agents.retrieval_agent import format_retrieval
from codeatlas.services.agents.types import AnswerResult, GenerateResult
from codeatlas.services.llm.quota import classify_quota, quota_answer
from codeatlas.services.qa.answer_service import GroundedAnswer
from codeatlas.services.retrieval.snippets import Snippet, render_snippets

# Start of the answer when the mentor fails; such answers are not kept as conversation history.
ANSWER_ERROR_PREFIX = "Error generating answer"


# Define the state for the graph
class OrchestratorState(TypedDict):
    question: str
    repo_id: str
    # The browser's random id for this chat, when there is one; the memory agent needs it.
    session_id: str | None
    plan: Plan
    current_step_index: int
    results: list[str]
    final_answer: str
    # The mentor's own answer, kept apart from final_answer so a later step in the
    # plan cannot overwrite it. A plan ending in the analyst used to replace the
    # answer with structural facts and throw the mentor's work away.
    mentor_answer: str
    citations: list[str]
    # Appended by each node as it runs, so the record reflects the graph, not a guess.
    agents_ran: list[str]
    # The code retrieval actually read, carried so later steps need not search again.
    snippets: list[Snippet]


class AgentOrchestrator:
    def __init__(
        self,
        planner: Agent,
        retrieval_agent: Agent,
        analyst_agent: Agent,
        mentor_agent: Agent,
        memory_agent: Agent,
    ) -> None:
        self._planner = planner
        self._retrieval_agent = retrieval_agent
        self._analyst_agent = analyst_agent
        self._mentor_agent = mentor_agent
        self._memory_agent = memory_agent
        self._logger = logging.getLogger(__name__)

        self._graph = self._build_graph()

    def _initial_state(self, question: str, repo_id: str, session_id: str | None) -> "OrchestratorState":
        return {
            "question": question,
            "repo_id": repo_id,
            "session_id": session_id,
            "plan": default_plan(question),
            "current_step_index": 0,
            "results": [],
            "final_answer": "",
            "mentor_answer": "",
            "citations": [],
            "snippets": [],
            "agents_ran": [],
        }

    def handle_question(self, question: str, repo_id: str, session_id: str | None = None) -> AnswerResult:
        final_state = self._graph.invoke(self._initial_state(question, repo_id, session_id))
        return self._result_from(final_state, question)

    def stream_question(self, question: str, repo_id: str, session_id: str | None = None) -> Iterator[dict]:
        """Run the graph, yielding one event as each agent finishes, then the answer.

        Events are {"type": "agent", ...} as the real graph executes, and finally
        {"type": "result", "result": AnswerResult}. Nothing here is logged.
        """
        state: dict = dict(self._initial_state(question, repo_id, session_id))
        for update in self._graph.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                if delta:
                    state.update(delta)
                if node in _AGENT_NODES:
                    yield {"type": "agent", "name": node, "detail": _detail(node, state)}
        yield {"type": "result", "result": self._result_from(state, question)}

    def _result_from(self, final_state: dict, question: str) -> AnswerResult:
        plan = final_state.get("plan") or default_plan(question)
        reasoning = [f"Plan: {' -> '.join(plan.agents)}"]
        if plan.note:
            reasoning.append(plan.note)
        reasoning.extend(final_state.get("results", []))

        agents_ran = final_state.get("agents_ran", [])
        reasoning.append(f"Agents that ran: {' -> '.join(agents_ran)}." if agents_ran else "No agent ran.")

        answer, note = _final_answer(final_state)
        if note:
            reasoning.append(note)

        return AnswerResult(
            answer=answer,
            citations=final_state.get("citations", []),
            reasoning_steps=reasoning,
            agents_used=agents_ran,
        )

    def handle_question_fast(
        self, question: str, repo_id: str, history: str = "", search_question: str | None = None
    ) -> AnswerResult:
        """Faster path: skip the planner, go straight retrieval → mentor.

        history: earlier turns of this conversation, shown to the mentor. search_question: a
        standalone version of a follow-up, used for retrieval instead of the question as asked.
        """
        # Appended as each agent is actually invoked, so a path that stops early reports
        # only what it reached.
        agents_ran: list[str] = []

        # 1. Retrieve — the only search this path makes.
        agents_ran.append("retrieval")
        retrieved = self._retrieve(search_question or question, repo_id)
        citations = retrieved.citations

        # 2. The mentor answers from exactly those snippets, so the answer and the citations
        #    describe the same code.
        agents_ran.append("mentor")
        try:
            with agent_run("mentor"):
                answer = self._mentor_answer(question, retrieved.snippets, repo_id, history=history)
        except Exception as e:
            self._logger.warning("Mentor failed: %s", type(e).__name__)
            record_failure("mentor", AGENT_ERROR)
            # A provider refusal is a quota problem, not a bug; the visitor is told so and
            # still gets the files that matched, rather than the raw exception text.
            kind = classify_quota(e)
            answer = quota_answer(kind, retrieved.snippets) if kind else f"{ANSWER_ERROR_PREFIX}: {e}"

        return AnswerResult(
            answer=answer,
            citations=citations,
            reasoning_steps=[
                "Fast mode: retrieval + mentor (skipped the planner).",
                f"Retrieved {len(retrieved.snippets)} code snippet(s), cited as {len(citations)} citation(s).",
                f"Agents that ran: {' -> '.join(agents_ran)}.",
            ],
            agents_used=agents_ran,
        )

    def handle_generation(self, prompt: str, repo_id: str) -> GenerateResult:
        """Generate code or example usage grounded in repo context."""
        # 1. Retrieve relevant context and citations
        retrieval_output = self._retrieval_agent.run(prompt, repo_id)
        citations = self._parse_citations_from_retrieval_output(retrieval_output)
        # 2. Ask mentor to generate code/example using that context
        gen_prompt = (
            f"Generate code or example usage for the following goal. "
            f"Use the retrieved code context and follow existing patterns. "
            f"Output the code in a clear block; then add brief notes if needed.\n\nGoal: {prompt}"
        )
        full_prompt = f"{gen_prompt}\n\nContext from retrieval:\n{retrieval_output}"
        try:
            mentor_output = self._mentor_agent.run(full_prompt, repo_id)
        except Exception as e:
            self._logger.warning("Generation failed: %s", e)
            mentor_output = f"Generation failed: {e}"
        # Treat full mentor response as diff; notes as single summary
        notes = ["Generated based on retrieved repo context."] if citations else []
        return GenerateResult(
            diff=mentor_output,
            notes=notes,
            citations=citations,
        )

    @staticmethod
    def _parse_citations_from_retrieval_output(text: str) -> list[str]:
        """Extract citation lines from retrieval agent output (e.g. 'Citations:\\n- ...')."""
        citations: list[str] = []
        if "Citations:" not in text:
            return citations
        after = text.split("Citations:", 1)[-1].strip()
        for line in after.splitlines():
            line = line.strip()
            if line.startswith("- "):
                citations.append(line[2:].strip())
        return citations

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)

        graph.add_node("planner", self._plan_node)
        graph.add_node("dispatcher", self._dispatcher_node)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("mentor", self._mentor_node)
        graph.add_node("memory", self._memory_node)

        graph.set_entry_point("planner")

        graph.add_edge("planner", "dispatcher")

        # Dispatcher conditional edges
        graph.add_conditional_edges(
            "dispatcher",
            self._route_step,
            {
                "retrieval": "retrieval",
                "analyst": "analyst",
                "mentor": "mentor",
                "memory": "memory",
                "end": END,
            },
        )

        # Return to dispatcher after each agent
        graph.add_edge("retrieval", "dispatcher")
        graph.add_edge("analyst", "dispatcher")
        graph.add_edge("mentor", "dispatcher")
        graph.add_edge("memory", "dispatcher")

        return graph.compile()

    # ... _plan_node, _dispatcher_node same ...

    def _plan_node(self, state: OrchestratorState) -> OrchestratorState:
        question = state["question"]
        repo_id = state["repo_id"]
        try:
            with agent_run("planner"):
                raw = self._planner.run(question, repo_id)
        except Exception as exc:
            # The planner is one model call; losing it should cost the default plan, not the answer.
            self._logger.warning("Planner failed (%s), using the default plan", type(exc).__name__)
            plan = default_plan(question, f"Planner failed ({type(exc).__name__}); used the default plan.")
        else:
            plan = parse_plan(raw, question)
            if plan.note:
                self._logger.info("Planner output adjusted: %s", plan.note)

        record_plan("default" if plan.note else "model", plan.agents)
        return {**state, "plan": plan, "current_step_index": 0, "agents_ran": [*state["agents_ran"], "planner"]}

    def _dispatcher_node(self, state: OrchestratorState) -> OrchestratorState:
        # Passthrough node (logic in _route_step), but can be used for logging
        return state

    def _route_step(self, state: OrchestratorState) -> str:
        """The next agent to run. Every step in a parsed Plan names a known agent."""
        steps = state["plan"].steps
        idx = state["current_step_index"]
        if idx >= len(steps):
            return "end"
        return steps[idx].agent

    # ... _execute_agent same ...

    def _execute_agent(self, agent: Agent, state: OrchestratorState) -> OrchestratorState:
        steps = state["plan"].steps
        idx = state["current_step_index"]
        step = steps[idx]

        instruction = step.instruction
        repo_id = state["repo_id"]

        # Add context from previous results if available
        context = "\n".join(state["results"])
        if context:
            full_prompt = f"{instruction}\n\nContext from previous steps:\n{context}"
        else:
            full_prompt = instruction

        try:
            with agent_run(step.agent):
                output = agent.run(full_prompt, repo_id)
        except Exception as e:
            output = f"Error executing {step.agent}: {e}"

        new_results = state["results"] + [f"Step {idx + 1} ({step.agent}): {output}"]
        new_citations = list(state.get("citations", []))
        if agent is self._retrieval_agent:
            new_citations.extend(self._parse_citations_from_retrieval_output(output))

        return {
            **state,
            "results": new_results,
            "final_answer": output,
            "current_step_index": idx + 1,
            "citations": new_citations,
            "agents_ran": [*state["agents_ran"], step.agent],
        }

    def _retrieval_node(self, state: OrchestratorState) -> OrchestratorState:
        """Retrieval keeps its snippets in the state; citations come from them, not from parsed text."""
        steps = state["plan"].steps
        idx = state["current_step_index"]
        step = steps[idx]
        retrieved = self._retrieve(step.instruction or state["question"], state["repo_id"])
        output = format_retrieval(retrieved.answer, retrieved.snippets)

        return {
            **state,
            "results": state["results"] + [f"Step {idx + 1} (retrieval): {output}"],
            "final_answer": output,
            "current_step_index": idx + 1,
            "citations": _merge_citations(state.get("citations", []), retrieved.citations),
            "snippets": _merge_snippets(state.get("snippets", []), retrieved.snippets),
            "agents_ran": [*state["agents_ran"], "retrieval"],
        }

    def _retrieve(self, question: str, repo_id: str) -> GroundedAnswer:
        """Snippets for a question, or an empty result if retrieval fails."""
        retrieve = getattr(self._retrieval_agent, "retrieve", None)
        try:
            with agent_run("retrieval"):
                if retrieve is not None:
                    return retrieve(question, repo_id)
                # A retrieval agent that only satisfies the text Agent contract.
                text = self._retrieval_agent.run(question, repo_id)
                return GroundedAnswer(
                    answer=text,
                    citations=self._parse_citations_from_retrieval_output(text),
                    reasoning_steps=[],
                    snippets=[],
                )
        except Exception as exc:
            self._logger.warning("Retrieval failed (%s)", type(exc).__name__)
            return GroundedAnswer(answer="", citations=[], reasoning_steps=[], snippets=[])

    def _analyst_node(self, state: OrchestratorState) -> OrchestratorState:
        return self._execute_agent(self._analyst_agent, state)

    def _mentor_node(self, state: OrchestratorState) -> OrchestratorState:
        """The mentor answers from the snippets retrieval already read, never its own search."""
        steps = state["plan"].steps
        idx = state["current_step_index"]
        step = steps[idx]
        snippets = state.get("snippets", [])
        facts = _facts_from(state.get("results", []))

        try:
            output = self._mentor_answer(step.instruction or state["question"], snippets, state["repo_id"], facts=facts)
        except Exception as exc:
            kind = classify_quota(exc)
            output = quota_answer(kind, snippets) if kind else f"Error executing mentor: {exc}"

        return {
            **state,
            "results": state["results"] + [f"Step {idx + 1} (mentor): {output}"],
            "final_answer": output,
            # A second mentor step replaces the first; the last mentor to run has the answer.
            "mentor_answer": output,
            "current_step_index": idx + 1,
            "agents_ran": [*state["agents_ran"], "mentor"],
        }

    def _mentor_answer(
        self, question: str, snippets: list[Snippet], repo_id: str, facts: str = "", history: str = ""
    ) -> str:
        """Call the mentor's snippet-based entry point, falling back to the text Agent contract."""
        answer = getattr(self._mentor_agent, "answer", None)
        if answer is not None:
            return answer(question, snippets, facts=facts, history=history)
        prompt = question
        if snippets:
            prompt = f"{question}\n\nRetrieved context:\n{render_snippets(snippets)}"
        if history:
            return self._mentor_agent.run(prompt, repo_id, history=history)
        return self._mentor_agent.run(prompt, repo_id)

    def _memory_node(self, state: OrchestratorState) -> OrchestratorState:
        """Recalls this conversation's earlier turns; needs the session, not a text prompt."""
        idx = state["current_step_index"]
        recall = getattr(self._memory_agent, "recall", None)
        try:
            with agent_run("memory"):
                if recall is not None:
                    output = recall(state.get("session_id"), state["repo_id"])
                else:
                    output = self._memory_agent.run(state["plan"].steps[idx].instruction, state["repo_id"])
        except Exception as exc:
            output = f"Error executing memory: {exc}"

        return {
            **state,
            "results": state["results"] + [f"Step {idx + 1} (memory): {output}"],
            "final_answer": output,
            "current_step_index": idx + 1,
            "agents_ran": [*state["agents_ran"], "memory"],
        }


def _merge_citations(existing: list[str], new: list[str]) -> list[str]:
    """Keep order, drop repeats: two retrieval steps often find the same file."""
    seen = set(existing)
    return existing + [c for c in new if not (c in seen or seen.add(c))]


def _merge_snippets(existing: list[Snippet], new: list[Snippet]) -> list[Snippet]:
    seen = {(s.path, s.start_line, s.end_line) for s in existing}
    out = list(existing)
    for snippet in new:
        key = (snippet.path, snippet.start_line, snippet.end_line)
        if key not in seen:
            seen.add(key)
            out.append(snippet)
    return out


# Analyst output is exact, computed from the code; the mentor is told to prefer it over its
# reading of the snippets. Only analyst steps count as facts.
_FACT_PREFIX = "(analyst): "


def _facts_from(results: list[str]) -> str:
    facts = [r.split(_FACT_PREFIX, 1)[1] for r in results if _FACT_PREFIX in r]
    return "\n\n".join(facts)


# Graph nodes that are agents; the dispatcher is plumbing and is not reported.
_AGENT_NODES = ("planner", "retrieval", "analyst", "mentor", "memory")


def _detail(node: str, state: dict) -> str:
    """A short, structural note about what an agent just did.

    Only counts and plan shape: never the question, the answer or any code, because this
    text is shown in the browser and must stay safe to put in a log line too.
    """
    if node == "planner":
        plan = state.get("plan")
        return f"planned {len(plan.steps)} step(s): {' -> '.join(plan.agents)}" if plan else "planned"
    if node == "retrieval":
        return f"read {len(state.get('snippets', []))} code snippet(s)"
    if node == "mentor":
        return "wrote the answer from the retrieved snippets"
    if node == "analyst":
        return "computed structural facts from the call graph"
    if node == "memory":
        return "recalled this conversation"
    return ""


# Said above a structural result, so a plan with no mentor step does not look as though the
# agent that happened to run last was answering the question in prose.
STRUCTURAL_PREFIX = "This is a structural result, computed from the code rather than written as an explanation.\n\n"


def _final_answer(state: dict) -> tuple[str, str]:
    """The answer to show, and a reasoning note when it is not the mentor's.

    The mentor writes the answer. Whatever else the plan runs afterwards -- an analyst
    summarising, a memory step recalling -- is a step along the way, not a replacement for
    it. Only when the plan has no mentor at all does another agent's output stand in, and
    then it is labelled.
    """
    mentor_answer = (state.get("mentor_answer") or "").strip()
    if mentor_answer:
        last = (state.get("final_answer") or "").strip()
        if last and last != mentor_answer:
            return mentor_answer, "Answer taken from the mentor step; later steps are shown above."
        return mentor_answer, ""

    last = (state.get("final_answer") or "").strip()
    if not last:
        return "No answer generated.", ""
    return STRUCTURAL_PREFIX + last, "No mentor step in the plan; showing the structural result."
