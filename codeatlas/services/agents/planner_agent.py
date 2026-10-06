from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.observability.timing import stage
from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.agents.plan import KNOWN_AGENTS, MAX_STEPS, default_plan, parse_plan


class PlannerAgent(Agent):
    """Breaks a question into steps for the other agents.

    `run` always returns the JSON of a plan that has already been validated, so a caller
    can parse it without guarding against the shapes a model actually produces. Callers
    that want the structure should use `plan()`; the orchestrator re-validates anyway.
    """

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a technical project manager. Break the user's coding question into "
                    "steps for specialised agents.\n"
                    "Available agents:\n"
                    "- 'retrieval': find code, 'where is class X'.\n"
                    "- 'analyst': structure and call/import relationships, 'who calls X'.\n"
                    "- 'mentor': explain, advise, write code, 'how do I fix X'.\n"
                    "- 'memory': recall what this conversation already covered.\n\n"
                    f"Use at most {MAX_STEPS} steps and only those agent names. Most questions "
                    "need retrieval first and mentor last.\n"
                    "Reply with only a JSON object: a key 'steps' holding a list of objects with "
                    "'agent' (string) and 'instruction' (string). No prose, no code fence.\n"
                    "Example:\n"
                    '{{"steps": [{{"agent": "retrieval", "instruction": "Find the auth middleware"}}, '
                    '{{"agent": "mentor", "instruction": "Explain how to add JWT to it"}}]}}',
                ),
                ("human", "{question}"),
            ]
        )

    def plan(self, question: str, repo_id: str | None = None):
        """A validated Plan. Falls back to the default plan when there is no model or it fails."""
        if self._llm is None:
            return default_plan(question, "No planner model configured; used the default plan.")
        try:
            chain = self._prompt | self._llm
            with stage("llm"):
                response = chain.invoke({"question": question})
        except Exception as exc:
            return default_plan(question, f"Planner model failed ({type(exc).__name__}); used the default plan.")
        return parse_plan(str(response.content), question)

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        return self.plan(prompt, repo_id).as_json()


__all__ = ["KNOWN_AGENTS", "PlannerAgent"]
