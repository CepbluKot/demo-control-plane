"""Run the frontend service with uvicorn."""

from __future__ import annotations

import uvicorn

from .config import get_frontend_settings


def main() -> None:
    settings = get_frontend_settings()
    uvicorn.run("summary_frontend.app:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
