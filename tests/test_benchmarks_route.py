"""GET /eval/benchmarks: a read-only route that serves a committed file and nothing else.

The route is open, like /eval/stats, so these tests also pin down what it must never carry:
no key, no account id, no budget or neuron figure, and no visitor question text.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeatlas.app.main import create_app
from codeatlas.controllers.eval_controller import HEADING_NOTE
from codeatlas.utils.config import AppConfig

ROOT = Path(__file__).resolve().parent.parent

CONFIG = AppConfig(
    embedding_provider="hash",
    embedding_model="",
    index_dir=".codeatlas/indexes",
    state_dir=".codeatlas/state",
    llm_provider="none",
    llm_model="",
    llm_temperature=0.2,
    api_key=None,
    auth_enabled=False,
)

SUMMARY = {"schema": 1, "retrieval": {"providers": {"hash": {"embedder": "hash (...)"}}}}


def _client(**overrides) -> TestClient:
    return TestClient(create_app(replace(CONFIG, **overrides)))


@pytest.fixture
def summary_file(tmp_path) -> Path:
    path = tmp_path / "benchmarks.json"
    path.write_text(json.dumps(SUMMARY), encoding="utf-8")
    return path


def test_serves_the_summary_file_as_it_is(summary_file):
    response = _client(benchmarks_file=str(summary_file)).get("/eval/benchmarks")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["summary"] == SUMMARY


def test_the_heading_note_says_recorded_not_live(summary_file):
    body = _client(benchmarks_file=str(summary_file)).get("/eval/benchmarks").json()

    assert body["heading_note"] == HEADING_NOTE
    assert "not live accuracy" in body["heading_note"]


def test_a_missing_file_is_answered_not_refused(tmp_path):
    response = _client(benchmarks_file=str(tmp_path / "nothing-here.json")).get("/eval/benchmarks")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "not available" in body["note"]
    assert "summary" not in body
    # The page still has the heading note and the live facts to show beside the empty panels.
    assert body["heading_note"] == HEADING_NOTE
    assert body["live"]["embedding_provider"] == "hash"


def test_an_unreadable_file_is_treated_as_missing(tmp_path):
    broken = tmp_path / "benchmarks.json"
    broken.write_text("{ this is not json", encoding="utf-8")

    body = _client(benchmarks_file=str(broken)).get("/eval/benchmarks").json()

    assert body["available"] is False


def test_the_live_block_is_the_provider_and_the_cap(summary_file):
    body = (
        _client(benchmarks_file=str(summary_file), embedding_provider="cloudflare", cloudflare_max_documents=600)
        .get("/eval/benchmarks")
        .json()
    )

    assert body["live"] == {"embedding_provider": "cloudflare", "semantic_document_cap": 600}


def test_nothing_secret_or_private_is_served(summary_file):
    text = _client(benchmarks_file=str(summary_file)).get("/eval/benchmarks").text.lower()

    for banned in ("api_key", "api-key", "account_id", "token", "neuron", "budget", "question_text"):
        assert banned not in text, f"the benchmarks route must not carry {banned}"


def test_the_route_needs_no_key_when_auth_is_on(summary_file):
    response = _client(benchmarks_file=str(summary_file), auth_enabled=True, api_key="secret").get("/eval/benchmarks")

    assert response.status_code == 200


def test_the_route_reads_the_file_per_request(summary_file):
    client = _client(benchmarks_file=str(summary_file))
    assert client.get("/eval/benchmarks").json()["available"] is True

    summary_file.unlink()
    assert client.get("/eval/benchmarks").json()["available"] is False


def test_the_default_path_is_the_committed_summary():
    assert CONFIG.benchmarks_file == "eval/benchmarks.json"
