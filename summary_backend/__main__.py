"""Run the FastAPI app with uvicorn."""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run("summary_backend.api:app", host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
