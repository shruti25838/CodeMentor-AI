import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import JSONResponse

from codeatlas.app.di import get_config
from codeatlas.app.rate_limit import RateLimits, limit_clone, limit_llm
from codeatlas.app.security import require_admin_key
from codeatlas.controllers.analyze_controller import router as analyze_router
from codeatlas.controllers.ask_controller import router as ask_router
from codeatlas.controllers.dependency_controller import router as dependency_router
from codeatlas.controllers.eval_controller import router as eval_router
from codeatlas.controllers.explain_controller import router as explain_router
from codeatlas.controllers.files_controller import router as files_router
from codeatlas.controllers.generate_controller import router as generate_router
from codeatlas.controllers.metrics_controller import router as metrics_router
from codeatlas.controllers.overview_controller import router as overview_router
from codeatlas.controllers.repos_controller import router as repos_router
from codeatlas.controllers.search_controller import router as search_router
from codeatlas.utils.config import AppConfig
from codeatlas.utils.logging import configure_logging

load_dotenv()


def _allowed_origins() -> list[str]:
    """CORS origins, comma-separated via CODEATLAS_ALLOWED_ORIGINS."""
    raw = os.getenv("CODEATLAS_ALLOWED_ORIGINS") or "http://localhost:3000,http://localhost:3001"
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(config: AppConfig | None = None) -> FastAPI:
    configure_logging()
    config = config or get_config()
    # Docs and schema are re-added below behind the admin key.
    app = FastAPI(title="CodeAtlas", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.config = config
    app.state.rate_limits = RateLimits(config)
    app.state.rate_limits.log_mode()

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Let browser code read how long to wait after a 429.
        expose_headers=["Retry-After"],
    )

    # The website only calls the user-facing endpoints, so they stay open for the
    # public demo. Admin/debug endpoints need X-API-Key (see codeatlas/app/security.py).
    admin_key = Depends(require_admin_key)

    # Limits only on endpoints that clone repositories or call the LLM; browsing stays unlimited.
    clone_limit = Depends(limit_clone)
    llm_limit = Depends(limit_llm)

    # User-facing (called by codementor-ui/lib/api.ts).
    app.include_router(analyze_router, dependencies=[clone_limit])
    app.include_router(ask_router, dependencies=[llm_limit])
    app.include_router(files_router)
    app.include_router(repos_router)
    app.include_router(overview_router)
    app.include_router(eval_router)
    # /dependencies/graph is user-facing; plain /dependencies is admin-only (set in the controller).
    app.include_router(dependency_router)

    # Admin/debug.
    app.include_router(explain_router, dependencies=[admin_key, llm_limit])
    app.include_router(search_router, dependencies=[admin_key])
    app.include_router(generate_router, dependencies=[admin_key, llm_limit])
    app.include_router(metrics_router, dependencies=[admin_key])

    @app.get("/openapi.json", include_in_schema=False, dependencies=[admin_key])
    def openapi_schema() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False, dependencies=[admin_key])
    def swagger_docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Docs")

    @app.get("/docs/oauth2-redirect", include_in_schema=False, dependencies=[admin_key])
    def swagger_oauth2_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/redoc", include_in_schema=False, dependencies=[admin_key])
    def redoc_docs():
        return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

    @app.middleware("http")
    async def record_metrics(request, call_next):
        from time import perf_counter

        from codeatlas.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY

        start = perf_counter()
        response = await call_next(request)
        elapsed = perf_counter() - start
        REQUEST_COUNT.labels(
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(path=request.url.path).observe(elapsed)
        return response

    return app


app = create_app()
