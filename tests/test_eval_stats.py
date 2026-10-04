from test_conversation_memory import Chat

MARKER = "private-question-marker-7f3a"


def test_eval_stats_has_counts_and_timings_but_no_question_text(tmp_path):
    chat = Chat(tmp_path)
    chat.ask(f"{MARKER} how does add work?")
    chat.ask(f"{MARKER} what is a closure?", repo=False)
    chat.env.client.post("/ask", json={"question": f"{MARKER} general"})

    resp = chat.env.client.get("/eval/stats")
    assert resp.status_code == 200
    assert MARKER not in resp.text
    stats = resp.json()
    assert stats["total_queries"] >= 3
    assert stats["recent_queries"]
    for query in stats["recent_queries"]:
        assert set(query) == {"latency_ms", "citations", "timestamp"}


def test_tracker_does_not_accept_question_text():
    import inspect

    from codeatlas.observability.tracker import QueryRecord, SessionTracker

    assert "question" not in inspect.signature(SessionTracker.record_query).parameters
    assert "question" not in QueryRecord.__dataclass_fields__
