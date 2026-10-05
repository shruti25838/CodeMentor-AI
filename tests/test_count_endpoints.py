import importlib.util
from pathlib import Path

from test_auth import ADMIN, USER_FACING

_spec = importlib.util.spec_from_file_location(
    "count_endpoints", Path(__file__).resolve().parent.parent / "scripts" / "count_endpoints.py"
)
count_endpoints = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(count_endpoints)

CONTROLLERS = Path(__file__).resolve().parent.parent / "codeatlas" / "controllers"


def test_counts_match_the_routes_the_auth_tests_check():
    result = count_endpoints.count()
    routes = {(r["method"], r["path"]) for r in result["list"]}
    assert routes == USER_FACING | ADMIN
    assert result["routes"] == len(USER_FACING | ADMIN)
    keyed = {(r["method"], r["path"]) for r in result["list"] if "key" in r["guards"]}
    assert keyed == ADMIN
    assert result["routes_open"] == len(USER_FACING)


def test_routers_are_the_controller_routers():
    result = count_endpoints.count()
    with_router = [p for p in CONTROLLERS.glob("*_controller.py") if "router = APIRouter(" in p.read_text()]
    assert result["routers"] == len(with_router)


def test_rate_limited_routes():
    limited = {r["path"] for r in count_endpoints.count()["list"] if {"clone", "llm"} & set(r["guards"])}
    assert limited == {"/analyze-repo", "/ask", "/ask/stream", "/explain", "/generate-code"}


def test_include_router_is_restored():
    from fastapi import FastAPI

    count_endpoints.count()
    assert FastAPI.include_router is count_endpoints._original_include
