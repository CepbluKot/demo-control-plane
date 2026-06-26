"""Static frontend service for summary jobs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import get_frontend_settings

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Summary Job Frontend", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/config.js")
def config_js() -> Response:
    settings = get_frontend_settings()
    payload = {
        "backendHttpUrl": settings.backend_http_url,
        "backendWsUrl": settings.backend_ws_url,
    }
    body = "window.SUMMARY_FRONTEND_CONFIG = " + json.dumps(payload, ensure_ascii=False) + ";\n"
    return Response(content=body, media_type="application/javascript")
