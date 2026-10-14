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


def _dashboard() -> dict:
    return json.loads((ROOT / "ops" / "grafana" / "dashboards" / "codementor.json").read_text())


def _panels(dashboard: dict) -> list[dict]:
    """Every panel, including the ones nested inside a row."""
    out = []
    for panel in dashboard["panels"]:
        out.append(panel)
        out.extend(panel.get("panels", []))
    return out


def _targets(dashboard: dict) -> list[dict]:
    return [target for panel in _panels(dashboard) for target in panel.get("targets", [])]


def _panel(dashboard: dict, title: str) -> dict:
    return next(p for p in _panels(dashboard) if p["title"] == title)


def test_dashboard_only_queries_metrics_that_exist():
    dashboard = _dashboard()
    known = {
        m._name
        for m in vars(metrics).values()
        if hasattr(m, "_name") and getattr(m, "_name", "").startswith("codeatlas_")
    }
    exprs = [t["expr"] for t in _targets(dashboard)]
    assert exprs
    for expr in exprs:
        for name in re.findall(r"codeatlas_[a-z_]+", expr):
            base = re.sub(r"_(bucket|sum|count|total)$", "", name)
            assert name in known or base in known, f"{name} in {expr}"


# ---------- Prometheus's own scrapes are not traffic ----------

# The three panels /metrics would otherwise dominate. Prometheus scrapes every 15 s whether
# anyone is using the site or not, which is 0.067 req/s of requests nobody made; on a quiet
# server that is most of the request graph, all of the status graph, and a latency line for
# the one route whose behaviour says nothing about the service.
SCRAPE_FREE_PANELS = (
    "Requests per second by route",
    "Response start latency p95 by route",
    "Responses by status",
)


def test_the_scrape_endpoint_is_left_out_of_the_traffic_panels():
    dashboard = _dashboard()
    for title in SCRAPE_FREE_PANELS:
        panel = _panel(dashboard, title)
        for target in panel["targets"]:
            assert 'path!="/metrics"' in target["expr"], title


def test_the_excluded_panels_still_ask_for_every_other_route():
    """The filter must be a negative match, not a list of routes that will go stale."""
    dashboard = _dashboard()
    for title in SCRAPE_FREE_PANELS:
        for target in _panel(dashboard, title)["targets"]:
            assert 'path="' not in target["expr"].replace('path!="', ""), title


def test_each_excluded_panel_says_so_in_its_description():
    dashboard = _dashboard()
    for title in SCRAPE_FREE_PANELS:
        assert "/metrics" in _panel(dashboard, title).get("description", ""), title


def test_the_errors_panel_still_counts_every_route():
    """Deliberately not filtered: a 5xx on /metrics is a real failure worth seeing."""
    panel = _panel(_dashboard(), "Errors per second")
    assert 'path!="/metrics"' not in panel["targets"][0]["expr"]


# ---------- the Semantic search row ----------

SEMANTIC_PANELS = {
    "Semantic index attempts per minute, by outcome": "codeatlas_semantic_index_total",
    "Estimated neurons spent per minute": "codeatlas_semantic_index_estimated_neurons_total",
    "Questions answered from the fallback index per minute, by reason": "codeatlas_retrieval_fallback_total",
}


def test_the_semantic_search_row_exists():
    dashboard = _dashboard()
    rows = [p for p in dashboard["panels"] if p["type"] == "row"]
    assert [r["title"] for r in rows] == ["Semantic search"]


def test_each_semantic_panel_queries_its_own_metric():
    dashboard = _dashboard()
    for title, metric in SEMANTIC_PANELS.items():
        panel = _panel(dashboard, title)
        expr = panel["targets"][0]["expr"]
        assert metric in expr, title


def test_the_semantic_panels_use_the_metrics_the_app_already_records():
    """No new metric was added for this row; all three were being recorded and never drawn.

    A Counter's `_name` has no `_total`, which prometheus_client appends when it renders the
    sample - so the dashboard's name is stripped before the comparison, as the check above
    does for `_bucket` and the rest.
    """
    known = {getattr(m, "_name", "") for m in vars(metrics).values()}
    for metric in SEMANTIC_PANELS.values():
        assert re.sub(r"_total$", "", metric) in known, metric


def test_the_semantic_panels_sit_below_everything_else_and_do_not_overlap():
    dashboard = _dashboard()
    row = _panel(dashboard, "Semantic search")
    others = [p for p in _panels(dashboard) if p["id"] < row["id"]]
    assert row["gridPos"]["y"] >= max(p["gridPos"]["y"] + p["gridPos"]["h"] for p in others)
    below = [_panel(dashboard, title) for title in SEMANTIC_PANELS]
    assert all(p["gridPos"]["y"] == row["gridPos"]["y"] + row["gridPos"]["h"] for p in below)
    assert sum(p["gridPos"]["w"] for p in below) == 24


def test_every_semantic_panel_explains_what_it_shows():
    dashboard = _dashboard()
    for title in SEMANTIC_PANELS:
        assert len(_panel(dashboard, title).get("description", "")) > 40, title


# ---------- a per-minute title means a per-minute axis ----------


def test_a_panel_titled_per_minute_multiplies_by_sixty_and_says_counts_per_minute():
    """rate() is per second. A per-second value under a per-minute title with a req/s axis
    reads 60 times too small, which is what these two panels used to do."""
    agents = json.loads((ROOT / "ops" / "grafana" / "dashboards" / "codeatlas-agents.json").read_text())
    for title in ("Agent runs per minute", "Model calls per minute, by agent"):
        panel = next(p for p in agents["panels"] if p["title"] == title)
        assert panel["targets"][0]["expr"].endswith("* 60"), title
        assert panel["fieldConfig"]["defaults"]["unit"] == "cpm", title


def test_no_semantic_panel_claims_a_unit_its_query_does_not_produce():
    dashboard = _dashboard()
    for title in SEMANTIC_PANELS:
        panel = _panel(dashboard, title)
        assert panel["targets"][0]["expr"].endswith("* 60"), title
        assert panel["fieldConfig"]["defaults"]["unit"] == "cpm", title


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
