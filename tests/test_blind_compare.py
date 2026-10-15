"""scripts/blind_compare.py against a fake server: the masking, the guards and the sheet.

The harness asks a real model through a real server, so what is worth testing is everything
except that: that the shuffle is seeded and reproducible, that the sheet a person fills in
cannot give the key away, that a resumed run does not pay twice for an answer it already has,
that the token cap stops the run while there is still room for the next question, and that a
429 ends it cleanly with what was already paid for still on disk.

The server here is a `ThreadingHTTPServer` on 127.0.0.1 that speaks the same SSE the app
speaks. It bumps a token ledger of its own after every request, so the cap is exercised
through the real `budget_guard` and the real ledger rather than through a stub. **No model, no
provider, no credential and nothing outside 127.0.0.1.**
"""

import json
import sys
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.blind_compare import (  # noqa: E402
    ARM_SETTINGS,
    DEFAULT_ARMS,
    LABELS,
    SCORE_COLUMNS,
    SHEET_COLUMNS,
    Paths,
    collect,
    complete_questions,
    inside_writable_root,
    key_for,
    labels_for,
    load_runs,
    parse_arm,
    read_sheet,
    sample_size_note,
    sheet_rows,
    summarise,
    unmask,
    write_key,
    write_sheet_csv,
    write_sheet_md,
)

ARMS = ["legacy", "grounded", "grounded-low"]

# One distinctive answer per arm that does not contain the arm's name, so a sheet carrying the
# answers can be searched for the arm names without the answers themselves matching.
ANSWER_OF = {
    "legacy": "Alpha explains the signer and derives a key first.",
    "grounded": "Bravo explains the signer, with the derivation step named.",
    "grounded-low": "Charlie explains the signer briefly.",
}
CITATIONS_OF = {
    "legacy": ["src/one.py (lines 1-4) | def sign(): ..."],
    "grounded": ["src/two.py (lines 5-9) | def unsign(): ..."],
    "grounded-low": ["src/three.py (lines 2-3) | def derive(): ..."],
}

QUESTIONS = [
    {"repo": "itsdangerous", "question_id": "itsdangerous-1", "question": "How does Signer derive its key?"},
    {"repo": "itsdangerous", "question_id": "itsdangerous-2", "question": "What does unsign check?"},
]

TOKENS_PER_CALL = 4_000


class FakeChat(BaseHTTPRequestHandler):
    """Answers /ask/stream as the app does, per the `X-Arm` header the client carries."""

    ledger: Path
    calls: list[dict] = []
    status_for: dict[str, int] = {}
    runs_seen: list[int] = []
    runs_path: Path | None = None

    def do_POST(self) -> None:  # noqa: N802 - the name BaseHTTPRequestHandler dispatches to
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        arm = self.headers.get("X-Arm", "")
        cls = FakeChat
        # How many runs were on disk when this request arrived; the harness claims to write
        # after every single one, and this is what proves it did.
        cls.runs_seen.append(len(load_runs(cls.runs_path)) if cls.runs_path else -1)
        cls.calls.append({"arm": arm, "question": body.get("question", "")})

        status = cls.status_for.get(arm, 200)
        if status != 200:
            payload = json.dumps({"detail": "slow down"}).encode()
            self._head(status, "application/json", len(payload))
            self.wfile.write(payload)
            return

        cls.bump_ledger()
        payload = self._sse(arm).encode()
        self._head(200, "text/event-stream", len(payload))
        self.wfile.write(payload)

    @staticmethod
    def _sse(arm: str) -> str:
        events = [
            {"type": "status", "stage": "retrieving", "message": "Retrieving context..."},
            {"type": "token", "content": ANSWER_OF.get(arm, "no arm")},
            {
                "type": "done",
                "citations": CITATIONS_OF.get(arm, []),
                "reasoning_steps": [],
                "agents_used": ["mentor"],
                "retrieval_note": "",
                "timings_ms": {"llm": 12.5, "total": 20.0},
            },
        ]
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def _head(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.end_headers()

    @classmethod
    def bump_ledger(cls) -> None:
        """What a model call costs, written where the real ledger would be."""
        spent = TOKENS_PER_CALL * (len([c for c in cls.calls if cls.status_for.get(c["arm"], 200) == 200]))
        cls.ledger.write_text(
            json.dumps(
                {"day": date.today().isoformat(), "models": {"fake": {"input": spent, "output": 0, "calls": 1}}}
            ),
            encoding="utf-8",
        )

    def log_message(self, *args) -> None:
        """Silence: the access log is noise in the test output."""


@pytest.fixture
def server(tmp_path):
    FakeChat.ledger = tmp_path / "token-spend.json"
    FakeChat.calls = []
    FakeChat.status_for = {}
    FakeChat.runs_seen = []
    FakeChat.runs_path = None
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def clients(server):
    host, port = server.server_address[0], server.server_address[1]
    base_url = f"http://{host}:{port}"
    made = {arm: httpx.Client(base_url=base_url, headers={"X-Arm": arm}, timeout=30) for arm in ARMS}
    try:
        yield made
    finally:
        for client in made.values():
            client.close()


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """A run folder in a temporary directory, with the writable root moved to match."""
    root = tmp_path / "blind-compare" / "a-run"
    monkeypatch.setattr("scripts.blind_compare.WRITES_UNDER", tmp_path)
    made = Paths(root)
    made.prepare()
    FakeChat.runs_path = made.runs
    return made


def _collect(clients, paths, items=None, token_cap=1_000_000, **kwargs):
    return collect(
        clients,
        "repo-1",
        items if items is not None else QUESTIONS,
        ARMS,
        paths,
        token_cap=token_cap,
        ledger=FakeChat.ledger,
        tpm_limit=10_000_000,
        report=lambda *a: None,
        **kwargs,
    )


# ---------- the seeded mask ----------


def test_the_shuffle_is_seeded_and_two_runs_of_it_agree():
    once = {item["question_id"]: labels_for(item["question_id"], ARMS, 1315) for item in QUESTIONS}
    again = {item["question_id"]: labels_for(item["question_id"], ARMS, 1315) for item in QUESTIONS}

    assert once == again
    for labels in once.values():
        assert sorted(labels) == list(LABELS[: len(ARMS)])
        assert sorted(labels.values()) == sorted(ARMS)


def test_a_different_seed_masks_the_questions_differently():
    """Not every question need move -- there are only six orders of three arms -- but the set
    of orders across a run must not be identical, or the seed would be doing nothing."""
    ids = [f"q-{n}" for n in range(20)]
    one = [tuple(labels_for(q, ARMS, 1315).values()) for q in ids]
    other = [tuple(labels_for(q, ARMS, 99).values()) for q in ids]

    assert one != other


def test_the_label_an_arm_gets_depends_on_the_question_as_well_as_the_seed():
    """Otherwise A is the same arm throughout and a grader stops being blind after one look."""
    orders = {tuple(labels_for(f"q-{n}", ARMS, 1315).values()) for n in range(30)}

    assert len(orders) > 1


def test_the_key_names_every_question_and_carries_its_seed():
    key = key_for([item["question_id"] for item in QUESTIONS], ARMS, 1315)

    assert key["seed"] == 1315
    assert key["arms"] == ARMS
    assert set(key["labels"]) == {item["question_id"] for item in QUESTIONS}


# ---------- the sheet gives nothing away ----------


def test_the_key_never_appears_in_the_sheet(clients, paths):
    rows, stopped = _collect(clients, paths)
    assert stopped == ""
    write_key(paths.key, rows, ARMS, 1315)
    write_sheet_md(paths.sheet_md, rows, ARMS, 1315)
    write_sheet_csv(paths.sheet_csv, sheet_rows(rows, ARMS, 1315))

    sheet_text = paths.sheet_md.read_text(encoding="utf-8") + paths.sheet_csv.read_text(encoding="utf-8")
    for arm in ARMS:
        assert arm not in sheet_text, f"{arm} is named in the sheet, so the comparison is not blind"
    # ...while the answers themselves are all there to be read and scored.
    for arm in ARMS:
        assert ANSWER_OF[arm] in sheet_text
    # The key is where the arms live, and it is a different file in a different folder.
    key_text = paths.key.read_text(encoding="utf-8")
    assert all(arm in key_text for arm in ARMS)
    assert paths.key.parent != paths.sheet_csv.parent


def test_the_sheet_has_one_row_per_question_and_label_with_empty_score_columns(clients, paths):
    rows, _ = _collect(clients, paths)
    write_sheet_csv(paths.sheet_csv, sheet_rows(rows, ARMS, 1315))

    sheet = read_sheet(paths.sheet_csv)

    assert list(sheet[0]) == list(SHEET_COLUMNS)
    assert len(sheet) == len(QUESTIONS) * len(ARMS)
    assert {row["label"] for row in sheet} == set(LABELS[: len(ARMS)])
    assert all(row[column] == "" for row in sheet for column in (*SCORE_COLUMNS, "note"))
    # The cited files are on the sheet, because one of the two scores is about them.
    assert any("src/one.py" in row["cited_files"] for row in sheet)


def test_a_question_only_one_arm_answered_is_left_off_the_sheet(clients, paths):
    """An arm scored on a question the others never saw is not a comparison."""
    FakeChat.status_for = {"grounded": 429}

    rows, stopped = _collect(clients, paths)

    assert stopped
    assert complete_questions(rows, ARMS) == []
    assert sheet_rows(rows, ARMS, 1315) == []


# ---------- resuming ----------


def test_a_resumed_run_asks_nothing_it_already_has(clients, paths):
    first, _ = _collect(clients, paths)
    assert len(first) == len(QUESTIONS) * len(ARMS)
    calls_after_first = len(FakeChat.calls)

    second, stopped = _collect(clients, paths)

    assert len(FakeChat.calls) == calls_after_first, "a resumed run asked the server again"
    assert stopped == ""
    assert [(r["question_id"], r["arm"]) for r in second] == [(r["question_id"], r["arm"]) for r in first]


def test_a_resumed_run_finishes_what_a_stop_left_undone(clients, paths):
    FakeChat.status_for = {"grounded-low": 429}
    first, stopped = _collect(clients, paths)
    assert stopped and len(first) < len(QUESTIONS) * len(ARMS)

    # The provider recovers, and the run picks up where it stopped.
    FakeChat.status_for = {}
    second, stopped = _collect(clients, paths)

    assert stopped == ""
    assert len(second) == len(QUESTIONS) * len(ARMS)
    assert len(complete_questions(second, ARMS)) == len(QUESTIONS)


def test_every_run_is_on_disk_before_the_next_one_is_asked(clients, paths):
    """Writing "after every run", as the server sees it: n runs on disk when run n+1 arrives."""
    _collect(clients, paths)

    assert FakeChat.runs_seen == list(range(len(QUESTIONS) * len(ARMS)))


# ---------- the guards ----------


def test_the_token_cap_stops_the_run_with_room_for_the_question_it_did_not_ask(clients, paths):
    """The reserve is the point: 8,000 spent is under a 10,000 cap, and a ninth thousand
    would not have been. The run stops rather than discovering the cost after paying it."""
    rows, stopped = _collect(clients, paths, token_cap=10_000)

    assert "token cap" in stopped
    assert len(rows) == 2, rows
    spent = sum(row["tokens"] for row in rows)
    assert spent == 2 * TOKENS_PER_CALL
    assert spent + TOKENS_PER_CALL > 10_000, "the run stopped exactly because the next one would not fit"
    assert load_runs(paths.runs) == rows


def test_a_429_ends_the_run_cleanly_and_what_was_paid_for_survives(clients, paths):
    FakeChat.status_for = {"grounded-low": 429}

    rows, stopped = _collect(clients, paths)

    assert "429" in stopped
    assert load_runs(paths.runs) == rows
    assert [row["refused"] for row in rows if row["refused"]] == ["http 429"]
    assert any(row["answer"] == ANSWER_OF["legacy"] for row in rows), "the answers already paid for are kept"


def test_a_refused_run_is_done_again_rather_than_scored(clients, paths):
    FakeChat.status_for = {"grounded-low": 429}
    _collect(clients, paths)
    FakeChat.status_for = {}

    rows, _ = _collect(clients, paths)

    assert [row for row in rows if row["refused"]] == []
    assert len(rows) == len(QUESTIONS) * len(ARMS)


# ---------- what the run writes, and where ----------


def test_a_run_records_the_answer_the_cited_files_and_the_timings(clients, paths):
    rows, _ = _collect(clients, paths)

    row = next(r for r in rows if r["arm"] == "grounded")
    assert row["answer"] == ANSWER_OF["grounded"]
    assert row["cited_files"] == ["src/two.py"]
    assert row["timings_ms"] == {"llm": 12.5, "total": 20.0}
    assert row["total_seconds"] >= 0


def test_nothing_may_be_written_outside_codeatlas(tmp_path):
    root = tmp_path / ".codeatlas"
    root.mkdir()

    assert inside_writable_root(root / "blind-compare" / "x", root) == (root / "blind-compare" / "x").resolve()
    with pytest.raises(SystemExit) as raised:
        inside_writable_root(root / ".." / "elsewhere", root)
    assert "writes only under" in str(raised.value)


def test_the_folder_everything_is_written_to_is_gitignored():
    """The answers are a model's prose and change on every run; committing them is noise."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".codeatlas/" in [line.strip() for line in ignored]


# ---------- arms ----------


def test_an_arm_is_a_set_of_environment_overrides():
    name, overrides = parse_arm("fast=CODEATLAS_CHAT_MODE=grounded,CODEATLAS_REASONING_EFFORT=low")

    assert name == "fast"
    assert overrides == {"CODEATLAS_CHAT_MODE": "grounded", "CODEATLAS_REASONING_EFFORT": "low"}


def test_an_override_that_would_not_be_applied_is_refused():
    """Two arms that differ only in a setting nobody applies are one arm, and the run would
    report a dead heat it had not measured."""
    with pytest.raises(SystemExit) as raised:
        parse_arm("x=CODEATLAS_EMBEDDING_PROVIDER=hash")
    assert "not an arm setting" in str(raised.value)

    for bad in ("", "no-equals", "x=", "x=JUST_A_NAME"):
        with pytest.raises(SystemExit):
            parse_arm(bad)


def test_the_default_arms_only_use_settings_a_server_could_be_started_with():
    for overrides in DEFAULT_ARMS.values():
        assert set(overrides) <= set(ARM_SETTINGS)


# ---------- scoring ----------


def _filled(rows, scores: dict[tuple[str, str], tuple[int, int]]) -> list[dict]:
    """The sheet as a person would hand it back, scored by (question_id, arm)."""
    sheet = sheet_rows(rows, ARMS, 1315)
    for row in sheet:
        arm = labels_for(row["question_id"], ARMS, 1315)[row["label"]]
        correctness, support = scores[(row["question_id"], arm)]
        row["correctness"], row["cited_support"] = str(correctness), str(support)
    return sheet


def test_scoring_puts_the_arms_back_and_averages_them(clients, paths):
    rows, _ = _collect(clients, paths)
    key = key_for(complete_questions(rows, ARMS), ARMS, 1315)
    scores = {
        ("itsdangerous-1", "legacy"): (2, 2),
        ("itsdangerous-1", "grounded"): (1, 1),
        ("itsdangerous-1", "grounded-low"): (0, 1),
        ("itsdangerous-2", "legacy"): (2, 1),
        ("itsdangerous-2", "grounded"): (2, 2),
        ("itsdangerous-2", "grounded-low"): (1, 0),
    }

    summary = summarise(unmask(_filled(rows, scores), key), ARMS)

    assert summary["questions_scored"] == 2
    assert summary["per_arm"]["legacy"]["correctness_mean"] == 2.0
    assert summary["per_arm"]["legacy"]["total_mean"] == 3.5
    assert summary["per_arm"]["grounded-low"]["total_mean"] == 1.0
    # legacy wins question 1 (4 against 2 and 1); grounded wins question 2 (4 against 3 and 1).
    assert summary["per_arm"]["legacy"]["wins"] == 1
    assert summary["per_arm"]["grounded"]["wins"] == 1
    assert summary["per_arm"]["grounded-low"]["wins"] == 0
    assert summary["ties"] == 0


def test_a_question_where_two_arms_tie_at_the_top_is_a_tie_and_not_a_win(clients, paths):
    rows, _ = _collect(clients, paths, items=QUESTIONS[:1])
    key = key_for(complete_questions(rows, ARMS), ARMS, 1315)
    scores = {
        ("itsdangerous-1", "legacy"): (2, 2),
        ("itsdangerous-1", "grounded"): (2, 2),
        ("itsdangerous-1", "grounded-low"): (0, 0),
    }

    summary = summarise(unmask(_filled(rows, scores), key), ARMS)

    assert summary["ties"] == 1
    assert sum(summary["per_arm"][arm]["wins"] for arm in ARMS) == 0


def test_a_question_not_scored_under_every_arm_is_left_out_of_the_averages(clients, paths):
    rows, _ = _collect(clients, paths)
    key = key_for(complete_questions(rows, ARMS), ARMS, 1315)
    sheet = _filled(rows, dict.fromkeys([(q["question_id"], a) for q in QUESTIONS for a in ARMS], (2, 2)))
    for row in sheet:
        if row["question_id"] == "itsdangerous-2" and row["label"] == "A":
            row["correctness"] = ""

    summary = summarise(unmask(sheet, key), ARMS)

    assert summary["questions_scored"] == 1
    assert summary["skipped"] == ["itsdangerous-2"]
    assert all(summary["per_arm"][arm]["questions"] == 1 for arm in ARMS)


def test_a_score_outside_zero_to_two_is_refused_by_name(clients, paths):
    rows, _ = _collect(clients, paths, items=QUESTIONS[:1])
    key = key_for(complete_questions(rows, ARMS), ARMS, 1315)
    sheet = _filled(rows, dict.fromkeys([("itsdangerous-1", a) for a in ARMS], (2, 2)))
    sheet[1]["correctness"] = "5"

    with pytest.raises(SystemExit) as raised:
        unmask(sheet, key)
    assert "itsdangerous-1/" in str(raised.value) and "must be 0, 1 or 2" in str(raised.value)


def test_a_sheet_the_key_does_not_know_about_is_refused(clients, paths):
    rows, _ = _collect(clients, paths, items=QUESTIONS[:1])
    key = key_for(complete_questions(rows, ARMS), ARMS, 1315)
    sheet = _filled(rows, dict.fromkeys([("itsdangerous-1", a) for a in ARMS], (2, 2)))
    sheet[0]["question_id"] = "itsdangerous-9"

    with pytest.raises(SystemExit) as raised:
        unmask(sheet, key)
    assert "the sheet and the key disagree" in str(raised.value)


def test_the_sample_size_is_always_said_plainly():
    note = sample_size_note(10)

    assert note.startswith("10 question(s)")
    assert "cannot separate small differences" in note
    assert "one person" in note
