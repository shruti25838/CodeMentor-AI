"""In-memory session tracker for eval dashboard analytics."""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class QueryRecord:
    question: str
    repo_id: str | None
    latency_ms: float
    citation_count: int
    agents_used: list[str]
    timestamp: float = field(default_factory=time.time)


class SessionTracker:
    """Thread-safe singleton for tracking session-level analytics."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._records: list[QueryRecord] = []
                cls._instance._agent_invocations: dict[str, int] = defaultdict(int)
                cls._instance._agent_total_ms: dict[str, float] = defaultdict(float)
        return cls._instance

    def record_query(
        self,
        question: str,
        repo_id: str | None,
        latency_ms: float,
        citation_count: int,
        agents_used: list[str],
    ) -> None:
        record = QueryRecord(
            question=question,
            repo_id=repo_id,
            latency_ms=latency_ms,
            citation_count=citation_count,
            agents_used=agents_used,
        )
        with self._lock:
            self._records.append(record)
            for agent in agents_used:
                self._agent_invocations[agent] += 1
                self._agent_total_ms[agent] += latency_ms / max(len(agents_used), 1)

    def get_stats(self) -> dict:
        with self._lock:
            records = list(self._records)
            agent_usage = dict(self._agent_invocations)
            agent_ms = dict(self._agent_total_ms)

        if not records:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "avg_citations": 0,
                "agent_usage": {},
                "recent_queries": [],
            }

        latencies = sorted(r.latency_ms for r in records)
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)

        return {
            "total_queries": len(records),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms": round(latencies[p95_idx], 1),
            "avg_citations": round(
                sum(r.citation_count for r in records) / len(records), 1
            ),
            "agent_usage": {
                agent: {
                    "invocations": agent_usage.get(agent, 0),
                    "avg_time_ms": round(
                        agent_ms.get(agent, 0)
                        / max(agent_usage.get(agent, 1), 1),
                        1,
                    ),
                }
                for agent in agent_usage
            },
            "recent_queries": [
                {
                    "question": r.question[:120],
                    "latency_ms": round(r.latency_ms, 1),
                    "citations": r.citation_count,
                    "timestamp": r.timestamp,
                }
                for r in records[-20:]
            ],
        }


# Module-level singleton
tracker = SessionTracker()
