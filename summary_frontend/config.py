"""Runtime configuration for the summary frontend service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name, "")
    return int(value) if value else default


@dataclass(frozen=True)
class FrontendSettings:
    host: str
    port: int
    backend_http_url: str
    backend_ws_url: str


def get_frontend_settings() -> FrontendSettings:
    backend_http_url = _env("SUMMARY_FRONTEND_BACKEND_HTTP_URL", "http://localhost:8088").rstrip("/")
    backend_ws_url = _env("SUMMARY_FRONTEND_BACKEND_WS_URL", "")
    if not backend_ws_url:
        backend_ws_url = backend_http_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    return FrontendSettings(
        host=_env("SUMMARY_FRONTEND_HOST", "0.0.0.0"),
        port=_env_int("SUMMARY_FRONTEND_PORT", 8090),
        backend_http_url=backend_http_url,
        backend_ws_url=backend_ws_url.rstrip("/"),
    )
