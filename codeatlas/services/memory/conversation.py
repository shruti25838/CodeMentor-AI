"""Short-term chat history, per browser session and repository, kept only in this process.

Nothing here is written to disk or logged. A session is found only by the random id its own
browser sends, and is stored under a hash of that id. Sessions expire after a period without
messages, and the number of sessions is capped (least recently used goes first).
"""

import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English and code; no tokenizer dependency."""
    return (len(text) + 3) // 4


@dataclass
class _Session:
    turns: list[Turn]
    last_used: float


class ConversationStore:
    def __init__(
        self,
        max_turns: int = 4,
        max_tokens: int = 1500,
        ttl_seconds: float = 1800,
        max_sessions: int = 1000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str, repo_id: str | None) -> list[Turn]:
        """The latest turns, oldest first, that fit in the token budget."""
        with self._lock:
            self._expire()
            session = self._sessions.get(_key(session_id, repo_id))
            turns = list(session.turns) if session else []
        return _fit(turns, self._max_tokens)

    def add(self, session_id: str, repo_id: str | None, question: str, answer: str) -> None:
        if self._max_turns <= 0:
            return
        key = _key(session_id, repo_id)
        with self._lock:
            self._expire()
            session = self._sessions.pop(key, None) or _Session(turns=[], last_used=0.0)
            session.turns = [*session.turns, Turn(question, answer)][-self._max_turns :]
            session.last_used = self._clock()
            self._sessions[key] = session
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def session_count(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)

    def _expire(self) -> None:
        cutoff = self._clock() - self._ttl
        # Ordered by last use, so expired sessions are at the front.
        while self._sessions:
            key, session = next(iter(self._sessions.items()))
            if session.last_used > cutoff:
                break
            del self._sessions[key]


def _key(session_id: str, repo_id: str | None) -> str:
    return hashlib.sha256(f"{session_id}\0{repo_id or ''}".encode()).hexdigest()


def _fit(turns: list[Turn], max_tokens: int) -> list[Turn]:
    """Keep the newest turns within max_tokens; shorten the newest answer if it alone is too long."""
    kept: list[Turn] = []
    used = 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn.question) + estimate_tokens(turn.answer)
        if used + cost <= max_tokens:
            kept.append(turn)
            used += cost
            continue
        if not kept:
            room = max_tokens - estimate_tokens(turn.question)
            if room > 0:
                kept.append(Turn(turn.question, turn.answer[: room * 4] + " ..."))
        break
    return list(reversed(kept))


def format_history(turns: list[Turn]) -> str:
    return "\n\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in turns)
