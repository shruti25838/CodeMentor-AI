from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.state.repo_state_store import RepoStateStore


class RepoAnalystAgent(Agent):
    def __init__(self, state_store: RepoStateStore, llm: BaseChatModel) -> None:
        self._state_store = state_store
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a software architect. Analyze the provided repository structure "
                    "and dependency graph to answer the user's question about the architecture. "
                    "Be concise and technical.",
                ),
                (
                    "human",
                    "Repository Context:\n{context}\n\nQuestion: {question}",
                ),
            ]
        )

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        if not repo_id:
            return "Error: repo_id is required for analysis."

        state = self._state_store.get(repo_id)
        if not state:
            return "Error: Repository state not found. Please analyze the repo first."

        # Summarize context for the LLM
        file_count = len(state.parsed_repo.files)
        func_count = len(state.parsed_repo.functions)
        edge_count = len(state.import_graph.edges)
        
        # Simple graph analysis
        most_depended_on = sorted(
            state.import_graph.in_degree, key=lambda x: x[1], reverse=True
        )[:5]
        
        context = (
            f"Files: {file_count}\n"
            f"Functions: {func_count}\n"
            f"Dependency Edges: {edge_count}\n"
            f"Top 5 most used modules: {most_depended_on}\n"
        )
        
        chain = self._prompt | self._llm
        response = chain.invoke({"context": context, "question": prompt})
        return response.content
