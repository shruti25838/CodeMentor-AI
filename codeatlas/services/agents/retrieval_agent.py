from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.qa.answer_service import AnswerService, GroundedAnswer
from codeatlas.services.retrieval.snippets import Snippet, render_snippets


class RetrievalAgent(Agent):
    """Finds the code that bears on a question.

    `locate` is the entry point for a caller that wants the code: it searches and returns the
    snippets, and makes **no model call**. `retrieve` is the same search plus a prose summary
    of what was found, which costs a model call and is worth asking for only where that text
    is read. `run` keeps the Agent text contract for the graph, and renders those snippets as
    code rather than as a summary of them.
    """

    def __init__(self, answer_service: AnswerService) -> None:
        self._answer_service = answer_service

    def locate(self, question: str, repo_id: str | None, top_k: int = 5) -> GroundedAnswer:
        """Search only. The snippets, the citations that describe them, and the fallback note."""
        if not repo_id:
            return _no_repo()
        return self._answer_service.locate(repo_id=repo_id, question=question, top_k=top_k)

    def retrieve(self, question: str, repo_id: str | None, top_k: int = 5) -> GroundedAnswer:
        """Search, and summarise what was found through the model, for a caller that reads it."""
        if not repo_id:
            return _no_repo()
        return self._answer_service.answer(repo_id=repo_id, question=question, top_k=top_k)

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        if not repo_id:
            return "Error: repo_id is required for retrieval."

        result = self.retrieve(prompt, repo_id)
        return format_retrieval(result.answer, result.snippets)


def format_retrieval(summary: str, snippets: list[Snippet]) -> str:
    """Retrieval's text output: the code first, then the summary, then the citations.

    The code comes first because it is the part a later step must not have to guess at.
    """
    if not snippets:
        # Nothing was read, so the summary is the whole output; wrapping it would only
        # double any heading it already carries.
        return summary

    sections = [f"Code snippets ({len(snippets)}):\n{render_snippets(snippets)}"]
    if summary:
        sections.append(f"Answer: {summary}")
    sections.append("Citations:\n" + "\n".join(f"- {snippet.location}" for snippet in snippets))
    return "\n\n".join(sections)


def _no_repo() -> GroundedAnswer:
    """Built fresh each time: GroundedAnswer is frozen, but the lists inside it are not."""
    return GroundedAnswer(
        answer="Error: repo_id is required for retrieval.",
        citations=[],
        reasoning_steps=[],
        snippets=[],
    )
