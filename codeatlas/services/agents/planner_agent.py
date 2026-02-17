import json
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from codeatlas.services.agents.interfaces import Agent

class PlannerAgent(Agent):
    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a sophisticated technical project manager. "
                    "Break down the user's coding question/request into a series of steps "
                    "executable by specialized agents.\n"
                    "Available Agents:\n"
                    "- 'retrieval': logic searching, finding code snippets, 'where is class X'.\n"
                    "- 'analyst': high-level architecture, dependency analysis, 'how is X structured'.\n"
                    "- 'mentor': coding advice, refactoring, generating new code, 'how to fix X'.\n"
                    "- 'memory': saving or listing conversation notes.\n\n"
                    "Output a JSON object with a key 'steps', where each step is an object "
                    "with 'agent' (string) and 'instruction' (string).\n"
                    "Example:\n"
                    '{{"steps": [{{"agent": "retrieval", "instruction": "Find auth middleware"}}, '
                    '{{"agent": "mentor", "instruction": "Explain how to add JWT based on retrieved code"}}]}}',
                ),
                ("human", "{question}"),
            ]
        )

    def run(self, prompt: str, repo_id: str | None = None) -> str:
        """Legacy run method returning a single intent label for backward compatibility if needed, 
        or a stringified JSON plan."""
        # For now, we return the JSON string so the orchestrator can parse it.
        # Fallback for "explain" vs "answer" if no LLM
        if self._llm is None:
            prompt_lower = prompt.lower()
            if "explain" in prompt_lower or "overview" in prompt_lower:
                return json.dumps({"steps": [{"agent": "analyst", "instruction": prompt}]})
            return json.dumps({"steps": [{"agent": "retrieval", "instruction": prompt}]})

        try:
            chain = self._prompt | self._llm
            response = chain.invoke({"question": prompt})
            # Naively try to parse JSON to ensure it's valid, then return the string
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return content
        except Exception:
            # Fallback
            return json.dumps({"steps": [{"agent": "retrieval", "instruction": prompt}]})
