from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.qa.answer_service import AnswerService, GroundedAnswer
from codeatlas.services.retrieval.snippets import Snippet, render_snippets


class RetrievalAgent(Agent):
    """Finds the code that bears on a question.

    `retrieve` is the useful entry point: it returns the snippets themselves, so the next
    agent reads the same code the user is shown. `run` keeps the Agent text contract for
    the graph, and renders those snippets as code rather than as a summary of them.
    """

    def __init__(self, answer_service: AnswerService) -> None:
        self._answer_service = answer_service

    def retrieve(self, question: str, repo_id: str | None, top_k: int = 5) -> GroundedAnswer:
        if not repo_id:
            return GroundedAnswer(
                answer="Error: repo_id is required for retrieval.",
                citations=[],
                reasoning_steps=[],
                snippets=[],
            )
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
