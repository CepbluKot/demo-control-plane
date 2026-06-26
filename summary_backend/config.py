"""Runtime configuration for the summary backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    value = _env(name, "")
    if not value:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = _env(name, "")
    if not value:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "")
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path(name: str, default: str) -> Path:
    return Path(_env(name, default)).expanduser()


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in _env(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]
    websocket_poll_interval_seconds: float

    log_level: str
    log_dir: Path
    audit_dir: Path
    upload_staging_dir: Path

    clickhouse_host: str
    clickhouse_port: int
    clickhouse_username: str
    clickhouse_password: str
    clickhouse_database: str
    clickhouse_secure: bool

    source_clickhouse_host: str
    source_clickhouse_port: int
    source_clickhouse_username: str
    source_clickhouse_password: str
    source_clickhouse_database: str
    source_clickhouse_secure: bool

    broker_url: str

    openai_api_base: str
    openai_api_key: str
    llm_model: str
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float
    dry_run: bool

    chunk_target_estimated_tokens: int
    reduce_group_size: int
    max_enqueue_nodes_per_advance: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    openai_api_base = _env("SUMMARY_BACKEND_OPENAI_API_BASE", _env("OPENAI_API_BASE_DB", ""))
    openai_api_key = _env("SUMMARY_BACKEND_OPENAI_API_KEY", _env("OPENAI_API_KEY_DB", ""))
    llm_model = _env("SUMMARY_BACKEND_LLM_MODEL", _env("LLM_MODEL_ID", ""))
    has_llm_config = bool(openai_api_base and openai_api_key and llm_model)
    clickhouse_host = _env("SUMMARY_BACKEND_CLICKHOUSE_HOST", "localhost")
    clickhouse_port = _env_int("SUMMARY_BACKEND_CLICKHOUSE_PORT", 8123)
    clickhouse_username = _env("SUMMARY_BACKEND_CLICKHOUSE_USERNAME", "default")
    clickhouse_password = _env("SUMMARY_BACKEND_CLICKHOUSE_PASSWORD", "")
    clickhouse_database = _env("SUMMARY_BACKEND_CLICKHOUSE_DATABASE", "summary_test")
    clickhouse_secure = _env_bool("SUMMARY_BACKEND_CLICKHOUSE_SECURE", False)

    return Settings(
        api_host=_env("SUMMARY_BACKEND_API_HOST", "0.0.0.0"),
        api_port=_env_int("SUMMARY_BACKEND_API_PORT", 8088),
        cors_origins=_env_csv(
            "SUMMARY_BACKEND_CORS_ORIGINS",
            "http://localhost:8090,http://127.0.0.1:8090",
        ),
        websocket_poll_interval_seconds=_env_float("SUMMARY_BACKEND_WEBSOCKET_POLL_INTERVAL_SECONDS", 0.5),
        log_level=_env("SUMMARY_BACKEND_LOG_LEVEL", "INFO"),
        log_dir=_env_path("SUMMARY_BACKEND_LOG_DIR", "artifacts/summary_backend/logs"),
        audit_dir=_env_path("SUMMARY_BACKEND_AUDIT_DIR", "artifacts/summary_backend/audit"),
        upload_staging_dir=_env_path("SUMMARY_BACKEND_UPLOAD_STAGING_DIR", "artifacts/summary_backend/uploads"),
        clickhouse_host=clickhouse_host,
        clickhouse_port=clickhouse_port,
        clickhouse_username=clickhouse_username,
        clickhouse_password=clickhouse_password,
        clickhouse_database=clickhouse_database,
        clickhouse_secure=clickhouse_secure,
        source_clickhouse_host=_env("SUMMARY_BACKEND_SOURCE_CLICKHOUSE_HOST", clickhouse_host),
        source_clickhouse_port=_env_int("SUMMARY_BACKEND_SOURCE_CLICKHOUSE_PORT", clickhouse_port),
        source_clickhouse_username=_env("SUMMARY_BACKEND_SOURCE_CLICKHOUSE_USERNAME", clickhouse_username),
        source_clickhouse_password=_env("SUMMARY_BACKEND_SOURCE_CLICKHOUSE_PASSWORD", clickhouse_password),
        source_clickhouse_database=_env("SUMMARY_BACKEND_SOURCE_CLICKHOUSE_DATABASE", clickhouse_database),
        source_clickhouse_secure=_env_bool("SUMMARY_BACKEND_SOURCE_CLICKHOUSE_SECURE", clickhouse_secure),
        broker_url=_env("SUMMARY_BACKEND_BROKER_URL", "redis://localhost:6379/0"),
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        llm_timeout_seconds=_env_float("SUMMARY_BACKEND_LLM_TIMEOUT_SECONDS", 120.0),
        llm_max_retries=_env_int("SUMMARY_BACKEND_LLM_MAX_RETRIES", 3),
        llm_retry_backoff_seconds=_env_float("SUMMARY_BACKEND_LLM_RETRY_BACKOFF_SECONDS", 2.0),
        dry_run=_env_bool("SUMMARY_BACKEND_DRY_RUN", not has_llm_config),
        chunk_target_estimated_tokens=_env_int("SUMMARY_BACKEND_CHUNK_TARGET_ESTIMATED_TOKENS", 6000),
        reduce_group_size=_env_int("SUMMARY_BACKEND_REDUCE_GROUP_SIZE", 8),
        max_enqueue_nodes_per_advance=_env_int("SUMMARY_BACKEND_MAX_ENQUEUE_NODES_PER_ADVANCE", 500),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()
