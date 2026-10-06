from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.observability.timing import stage
from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.snippets import Snippet

NO_CONTEXT = (
    "I could not find code in this repository that answers that. "
    "Try naming a file, class or function, or re-index the repository."
)

_SYSTEM = (
    "You are a senior software engineer and mentor.\n"
    "Answer ONLY from the numbered code snippets and the structural facts below. They are the "
    "whole of what you know about this repository.\n"
    "Cite a snippet as `path (lines start-end)`, copying the location exactly as it is given. "
    "Cite only snippets listed below; never cite a file that is not in the list, and never "
    "invent a line number.\n"
    "If the snippets do not contain the answer, say which part is missing instead of guessing. "
    "Structural facts are computed from the code and are exact; prefer them over your "
    "impression of the snippets."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        ("human", "{history}{facts}Code snippets:\n{snippets}\n\nQuestion: {question}"),
    ]
)


class CodingMentorAgent(Agent):
    """Explains and advises, using only the code it was handed.

    It used to run its own search inside `run`, so the snippets it read were not the ones the
    user was shown as citations. On one real question the answer described `signer.py` while
    the citations named `exc.py` and `url_safe.py`, and the model invented line numbers for a
    file it had never seen. The orchestrator now passes the retrieved snippets in, and this
    agent never searches.
    """

    def __init__(self, llm: BaseChatModel, answer_service: AnswerService | None = None) -> None:
        self._llm = llm
        # Only used by `run`, for callers outside the graph that have no snippets of their own.
        self._answer_service = answer_service
        self._prompt = _PROMPT

    def answer(
        self,
        question: str,
        snippets: list[Snippet],
        facts: str = "",
        history: str = "",
    ) -> str:
        """An answer grounded in exactly `snippets`. No search happens here."""
        if not snippets and not facts:
            return NO_CONTEXT

        chain = self._prompt | self._llm
        with stage("llm"):
            response = chain.invoke(
                {
                    "question": question,
                    "snippets": _numbered(snippets),
                    "facts": f"Structural facts (exact, computed from the code):\n{facts}\n\n" if facts else "",
                    "history": f"Earlier in this conversation:\n{history}\n\n" if history else "",
                }
            )
        return str(response.content)

    def run(self, prompt: str, repo_id: str | None = None, history: str = "") -> str:
        """Agent contract for callers that have no snippets; retrieves once, then answers."""
        if not repo_id:
            return "Error: repo_id is required for coding assistance."
        if self._answer_service is None:
            return self.answer(prompt, [], history=history)

        retrieved = self._answer_service.answer(repo_id=repo_id, question=prompt, top_k=5)
        return self.answer(prompt, retrieved.snippets, history=history)


def _numbered(snippets: list[Snippet]) -> str:
    """Snippets as a numbered list, each headed by the exact location the model must cite."""
    if not snippets:
        return "(none)"
    return "\n\n".join(f"[{i}] {s.location}\n{s.text}" for i, s in enumerate(snippets, 1))
