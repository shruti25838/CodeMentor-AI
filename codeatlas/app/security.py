from fastapi import Header, HTTPException

from codeatlas.utils.config import AppConfig


def verify_api_key(
    config: AppConfig, x_api_key: str | None = Header(default=None)
) -> None:
    if not config.auth_enabled:
        return
    if not config.api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
