import os

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from codeatlas.app.di import get_config
from codeatlas.app.security import verify_api_key
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
from codeatlas.utils.logging import configure_logging
from dotenv import load_dotenv

load_dotenv()

def _allowed_origins() -> list[str]:
    """CORS origins, comma-separated via CODEATLAS_ALLOWED_ORIGINS."""
    raw = (
        os.getenv("CODEATLAS_ALLOWED_ORIGINS")
        or "http://localhost:3000,http://localhost:3001"
    )
    return [o.strip() for o in raw.split(",") if o.strip()]

def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="CodeAtlas", version="0.1.0")

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Simplified dependency to bypass API key check for local frontend
    def auth_dep(x_api_key: str | None = Header(default=None)) -> None:
        # verify_api_key(get_config(), x_api_key)
        pass 

    auth_dependency = Depends(auth_dep)
    
    app.include_router(analyze_router, dependencies=[auth_dependency])
    app.include_router(ask_router, dependencies=[auth_dependency])
    app.include_router(explain_router, dependencies=[auth_dependency])
    app.include_router(dependency_router, dependencies=[auth_dependency])
    app.include_router(files_router, dependencies=[auth_dependency])
    app.include_router(repos_router, dependencies=[auth_dependency])
    app.include_router(overview_router, dependencies=[auth_dependency])
    app.include_router(search_router, dependencies=[auth_dependency])
    app.include_router(generate_router, dependencies=[auth_dependency])
    app.include_router(eval_router, dependencies=[auth_dependency])
    app.include_router(metrics_router)

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
