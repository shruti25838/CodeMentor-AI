from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.qa.answer_service import AnswerService


class RetrievalAgent(Agent):
    def __init__(self, answer_service: AnswerService) -> None:
        self._answer_service = answer_service

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        if not repo_id:
            return "Error: repo_id is required for retrieval."
        
        result = self._answer_service.answer(repo_id=repo_id, question=prompt)
        
        # Format the output for the orchestrator/user
        response = [f"Answer: {result.answer}\n"]
        if result.citations:
            response.append("Citations:")
            for citation in result.citations:
                response.append(f"- {citation}")
        
        return "\n".join(response)
