"""Recalls what this conversation already covered.

This used to be a separate store: a JSON file on disk, scoped `repo:<id>` or `global`, so
one visitor's notes were visible to every other visitor, with no expiry and no size limit,
and nothing in the pipeline ever read it to answer anything. Its only commands were
"save: <text>" and "list"; any real question got a usage string back.

It now reads the same per-session conversation memory the website chat uses, so it inherits
that store's rules exactly: in process only, never written to disk, keyed by a hash of the
browser's random session id plus the repository, the newest few turns within a token
budget, and dropped after an idle period or when too many sessions are held.

Nothing here logs conversation content.
"""

import logging

from codeatlas.services.agents.interfaces import Agent
from codeatlas.services.memory.conversation import ConversationStore, format_history

NO_SESSION = "This conversation has no memory: the question was asked without a session."
NO_HISTORY = "Nothing earlier in this conversation."


class MemoryAgent(Agent):
    def __init__(self, conversations: ConversationStore) -> None:
        self._conversations = conversations
        self._logger = logging.getLogger(__name__)

    def recall(self, session_id: str | None, repo_id: str | None) -> str:
        """The earlier turns of this conversation, as text for a prompt."""
        if not session_id:
            return NO_SESSION
        turns = self._conversations.history(session_id, repo_id)
        # Count only; the turns themselves are never logged.
        self._logger.info("Recalled %d earlier turn(s)", len(turns))
        if not turns:
            return NO_HISTORY
        return format_history(turns)

    def run(self, prompt: str, repo_id: str | None = None, session_id: str | None = None) -> str:
        return self.recall(session_id, repo_id)
