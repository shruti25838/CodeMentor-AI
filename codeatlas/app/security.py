import hmac

from fastapi import Header, HTTPException, Request

from codeatlas.utils.config import AppConfig


def verify_api_key(config: AppConfig, x_api_key: str | None = Header(default=None)) -> None:
    if not config.auth_enabled:
        return
    if not config.api_key:
        # Fail closed: with no key configured, nobody gets in.
        raise HTTPException(
            status_code=503,
            detail="This endpoint is disabled because no API key is configured on the server.",
        )
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), config.api_key.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")


def require_admin_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Router dependency for admin/debug endpoints. The website never calls these."""
    verify_api_key(request.app.state.config, x_api_key)
