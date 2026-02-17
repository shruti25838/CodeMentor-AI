from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.qa.answer_service import AnswerService


class CodingMentorAgent(Agent):
    def __init__(self, answer_service: AnswerService, llm: BaseChatModel) -> None:
        self._answer_service = answer_service
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Senior Software Engineer and Mentor. "
                    "Use the provided code context to answer the user's request. "
                    "Provide specific code examples and refactoring suggestions where appropriate. "
                    "Cite the file paths you are referencing.",
                ),
                (
                    "human",
                    "Goal: {goal}\n\nExisting Code Context:\n{context}",
                ),
            ]
        )

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        if not repo_id:
            return "Error: repo_id is required for coding assistance."

        # 1. Retrieve relevant context
        # We assume the prompt is the question/goal
        retrieval = self._answer_service.answer(repo_id=repo_id, question=prompt, top_k=3)
        
        context_str = ""
        if retrieval.citations:
             context_str = f"Found relevant code:\n{retrieval.answer}\n\nCitations:\n" + "\n".join(retrieval.citations)
        else:
             context_str = "No relevant code found in the repository index."

        # 2. Generate advice
        chain = self._prompt | self._llm
        response = chain.invoke({"goal": prompt, "context": context_str})
        
        return response.content
