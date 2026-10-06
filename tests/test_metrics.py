import json
import re
from dataclasses import replace
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from test_analyze_indexing import Env
from test_unknown_repo import CONFIG, KNOWN, FakeStateStore, ModelSpy

from codeatlas.app.di import get_agent_orchestrator, get_llm_provider, get_repo_state_store
from codeatlas.app.main import create_app
from codeatlas.observability import metrics
from codeatlas.observability.metrics import load_eval_results, observe_stages

ROOT = Path(__file__).resolve().parent.parent


def _value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_model_stages_are_summed_and_total_is_recorded():
    before_model = _value("codeatlas_stage_duration_seconds_sum", endpoint="t_sum", stage="model")
    before_count = _value("codeatlas_stage_duration_seconds_count", endpoint="t_sum", stage="model")
    observe_stages("t_sum", {"llm": 1500.0, "rewrite": 500.0, "search": 2.0}, total_ms=2100.0)
    assert _value("codeatlas_stage_duration_seconds_sum", endpoint="t_sum", stage="model") - before_model == 2.0
    # One observation per request, not one per model call.
    assert _value("codeatlas_stage_duration_seconds_count", endpoint="t_sum", stage="model") - before_count == 1
    assert _value("codeatlas_stage_duration_seconds_sum", endpoint="t_sum", stage="total") >= 2.1
    assert _value("codeatlas_stage_duration_seconds_bucket", endpoint="t_sum", stage="search", le="0.0025") >= 1


def test_indexing_request_records_its_stages(tmp_path):
    env = Env(tmp_path)
    before = {
        s: _value("codeatlas_stage_duration_seconds_count", endpoint="analyze", stage=s)
        for s in ("clone", "parse", "embed", "total")
    }
    assert env.analyze().status_code == 200
    for stage, count in before.items():
        assert _value("codeatlas_stage_duration_seconds_count", endpoint="analyze", stage=stage) == count + 1


def _client(spy=None) -> TestClient:
    app = create_app(CONFIG)
    app.dependency_overrides[get_repo_state_store] = FakeStateStore
    app.dependency_overrides[get_agent_orchestrator] = lambda: spy or ModelSpy()
    app.dependency_overrides[get_llm_provider] = lambda: spy or ModelSpy()
    return TestClient(app, raise_server_exceptions=False)


def test_requests_are_labelled_by_route_template_not_raw_url():
    client = _client()
    before = _value("codeatlas_request_total", method="GET", path="unmatched", status="404")
    client.get("/no-such-page-12345")
    client.get("/another/random/url")
    assert _value("codeatlas_request_total", method="GET", path="unmatched", status="404") == before + 2
    assert (
        REGISTRY.get_sample_value(
            "codeatlas_request_total", {"method": "GET", "path": "/no-such-page-12345", "status": "404"}
        )
        is None
    )
    before = _value("codeatlas_request_total", method="GET", path="/docs", status="200")
    client.get("/docs")
    assert _value("codeatlas_request_total", method="GET", path="/docs", status="200") == before + 1


class FailingSpy(ModelSpy):
    def handle_question(self, question, repo_id):
        raise RuntimeError("boom")

    def handle_question_fast(self, question, repo_id, **kwargs):
        raise RuntimeError("boom")


def test_unhandled_exception_and_stream_error_are_counted():
    client = _client(FailingSpy())
    before = _value("codeatlas_errors_total", path="/ask", kind="exception")
    before_500 = _value("codeatlas_request_total", method="POST", path="/ask", status="500")
    assert client.post("/ask", json={"question": "q", "repo_id": KNOWN}).status_code == 500
    assert _value("codeatlas_errors_total", path="/ask", kind="exception") == before + 1
    assert _value("codeatlas_request_total", method="POST", path="/ask", status="500") == before_500 + 1

    before = _value("codeatlas_errors_total", path="/ask/stream", kind="stream_error")
    resp = client.post("/ask/stream", json={"question": "q", "repo_id": KNOWN})
    assert resp.status_code == 200 and '"type": "error"' in resp.text
    assert _value("codeatlas_errors_total", path="/ask/stream", kind="stream_error") == before + 1


def test_5xx_response_is_counted(tmp_path, monkeypatch):
    env = Env(tmp_path)
    monkeypatch.setattr(env.index, "index_repository", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    before = _value("codeatlas_errors_total", path="/analyze-repo", kind="http_5xx")
    assert env.analyze().status_code == 500
    assert _value("codeatlas_errors_total", path="/analyze-repo", kind="http_5xx") == before + 1


def test_eval_results_feed_the_retrieval_gauges(tmp_path):
    result = {
        "codementor_commit": "abc1234",
        "embedder": "hash (HashEmbeddingService, 384 dims)",
        "split": "unit",
        "overall": {"questions": 2, "hit_rate": {"@1": 0.5, "@3": 1.0, "@5": 1.0}, "mrr@10": 0.75},
    }
    (tmp_path / "unit.json").write_text(json.dumps(result))
    (tmp_path / "broken.json").write_text("{not json")
    assert load_eval_results(str(tmp_path)) == 1
    assert _value("codeatlas_retrieval_hit_rate", split="unit", k="1") == 0.5
    assert _value("codeatlas_retrieval_mrr", split="unit") == 0.75
    assert _value("codeatlas_retrieval_eval_info", split="unit", commit="abc1234", embedder="hash") == 1


def test_checked_in_eval_results_reach_metrics_with_the_key():
    create_app(replace(CONFIG, eval_results_dir=str(ROOT / "eval" / "results")))
    recorded = json.loads((ROOT / "eval" / "results" / "test.json").read_text())
    assert _value("codeatlas_retrieval_hit_rate", split="test", k="5") == recorded["overall"]["hit_rate"]["@5"]
    app = create_app(replace(CONFIG, auth_enabled=True, api_key="k"))
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401
    body = client.get("/metrics", headers={"X-API-Key": "k"}).text
    for name in ("codeatlas_stage_duration_seconds", "codeatlas_errors_total", "codeatlas_retrieval_hit_rate"):
        assert name in body


def test_dashboard_only_queries_metrics_that_exist():
    dashboard = json.loads((ROOT / "ops" / "grafana" / "dashboards" / "codementor.json").read_text())
    known = {
        m._name
        for m in vars(metrics).values()
        if hasattr(m, "_name") and getattr(m, "_name", "").startswith("codeatlas_")
    }
    exprs = [t["expr"] for p in dashboard["panels"] for t in p["targets"]]
    assert exprs
    for expr in exprs:
        for name in re.findall(r"codeatlas_[a-z_]+", expr):
            base = re.sub(r"_(bucket|sum|count|total)$", "", name)
            assert name in known or base in known, f"{name} in {expr}"


def test_compose_gives_prometheus_the_key_file_it_reads():
    compose = yaml.safe_load((ROOT / "docker-compose.monitoring.yml").read_text())
    prometheus = yaml.safe_load((ROOT / "ops" / "prometheus.yml").read_text())
    files = prometheus["scrape_configs"][0]["http_headers"]["X-API-Key"]["files"]
    secret = compose["services"]["prometheus"]["secrets"][0]
    assert files == [f"/run/secrets/{secret}"]
    assert compose["secrets"][secret] == {"environment": "CODEATLAS_API_KEY"}
    app_env = compose["services"]["codeatlas"]["environment"]
    assert app_env["CODEATLAS_AUTH_ENABLED"] == "true"
    assert app_env["CODEATLAS_EMBEDDING_PROVIDER"] == "hash"
    assert "eval/results" in (ROOT / "Dockerfile").read_text()
