"""Count the API's routers and its distinct (method, path) routes, from the app itself.

    python scripts/count_endpoints.py          # totals and one line per route
    python scripts/count_endpoints.py --json   # the same as JSON

Routers are the APIRouter objects included into the app by create_app(). Routes are every
(method, path) pair the app answers, including the docs pages; HEAD and OPTIONS answered
automatically by the framework are not counted. Each route is marked with what guards it:
"key" = needs X-API-Key (when CODEATLAS_AUTH_ENABLED is true, the default), "clone"/"llm" =
counted against that request limit. Nothing is started and no request is made.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402

_included: list = []
_original_include = FastAPI.include_router


def _counting_include(self, router, *args, **kwargs):
    _included.append(router)
    return _original_include(self, router, *args, **kwargs)


def count() -> dict:
    FastAPI.include_router = _counting_include
    try:
        from codeatlas.app.main import create_app

        _included.clear()
        app = create_app()
    finally:
        FastAPI.include_router = _original_include

    from codeatlas.app.rate_limit import limit_clone, limit_llm
    from codeatlas.app.security import require_admin_key

    guards = {require_admin_key: "key", limit_clone: "clone", limit_llm: "llm"}
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        if not methods:
            continue
        found = sorted(_guards(route.dependant, guards)) if isinstance(route, APIRoute) else []
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            routes.append({"method": method, "path": route.path, "guards": found})
    distinct = sorted({(r["method"], r["path"]) for r in routes})
    routes = sorted({(r["method"], r["path"]): r for r in routes}.values(), key=lambda r: (r["path"], r["method"]))
    return {
        "routers": len({id(r) for r in _included}),
        "routes": len(distinct),
        "routes_needing_key": sum(1 for r in routes if "key" in r["guards"]),
        "routes_open": sum(1 for r in routes if "key" not in r["guards"]),
        "routes_rate_limited": sum(1 for r in routes if {"clone", "llm"} & set(r["guards"])),
        "list": routes,
    }


def _guards(dependant, guards: dict) -> set[str]:
    found: set[str] = set()
    for dep in dependant.dependencies:
        if dep.call in guards:
            found.add(guards[dep.call])
        found |= _guards(dep, guards)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = count()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"routers: {result['routers']}")
    print(f"distinct method+path routes: {result['routes']}")
    print(f"  need the API key: {result['routes_needing_key']}")
    print(f"  open: {result['routes_open']}")
    print(f"  rate limited (clone or llm): {result['routes_rate_limited']}")
    for r in result["list"]:
        print(f"  {r['method']:<6} {r['path']:<28} {','.join(r['guards'])}")


if __name__ == "__main__":
    main()
