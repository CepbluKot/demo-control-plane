"""Runtime configuration for the summary backend."""

from __future__ import annotations

import os
import json
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


def _load_llm_gateway_defaults() -> dict[str, object]:
    config_path = Path(__file__).with_name("llm_gateway_defaults.json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    worker_processes: int
    worker_threads: int

    openai_api_base: str
    openai_api_key: str
    llm_model: str
    llm_models: tuple[str, ...]
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float
    llm_max_concurrency: int
    llm_pool_acquire_timeout_seconds: float
    llm_pool_poll_interval_seconds: float
    llm_pool_retry_delay_seconds: float
    dry_run: bool

    chunk_target_estimated_tokens: int
    reduce_group_size: int
    max_enqueue_nodes_per_advance: int
    queued_node_requeue_after_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    llm_gateway_defaults = _load_llm_gateway_defaults()
    default_api_base = str(llm_gateway_defaults.get("api_base") or "").strip()
    default_api_key = str(llm_gateway_defaults.get("api_key") or "").strip()
    default_model = str(llm_gateway_defaults.get("default_model") or "").strip()
    raw_available_models = llm_gateway_defaults.get("available_models")
    if not isinstance(raw_available_models, list):
        raw_available_models = []
    default_available_models = tuple(
        str(model).strip()
        for model in raw_available_models
        if str(model).strip()
    )
    default_llm_models = default_available_models or ((default_model,) if default_model else ())

    openai_api_base = _env("SUMMARY_BACKEND_OPENAI_API_BASE", _env("OPENAI_API_BASE_DB", default_api_base))
    openai_api_key = _env("SUMMARY_BACKEND_OPENAI_API_KEY", _env("OPENAI_API_KEY_DB", default_api_key))
    llm_model = _env("SUMMARY_BACKEND_LLM_MODEL", _env("LLM_MODEL_ID", default_model))
    llm_models = _env_csv("SUMMARY_BACKEND_LLM_MODELS", ",".join(default_llm_models))
    if llm_model and llm_model not in llm_models:
        llm_models = (llm_model, *llm_models)
    has_llm_config = bool(openai_api_base and openai_api_key and llm_model)
    clickhouse_host = _env("SUMMARY_BACKEND_CLICKHOUSE_HOST", "localhost")
    clickhouse_port = _env_int("SUMMARY_BACKEND_CLICKHOUSE_PORT", 8123)
    clickhouse_username = _env("SUMMARY_BACKEND_CLICKHOUSE_USERNAME", "default")
    clickhouse_password = _env("SUMMARY_BACKEND_CLICKHOUSE_PASSWORD", "")
    clickhouse_database = _env("SUMMARY_BACKEND_CLICKHOUSE_DATABASE", "summary_test")
    clickhouse_secure = _env_bool("SUMMARY_BACKEND_CLICKHOUSE_SECURE", False)
    worker_processes = _env_int("SUMMARY_BACKEND_WORKER_PROCESSES", 1)
    worker_threads = _env_int("SUMMARY_BACKEND_WORKER_THREADS", 4)

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
        worker_processes=worker_processes,
        worker_threads=worker_threads,
        openai_api_base=openai_api_base,
        openai_api_key=openai_api_key,
        llm_model=llm_model,
        llm_models=llm_models,
        llm_timeout_seconds=_env_float("SUMMARY_BACKEND_LLM_TIMEOUT_SECONDS", 600.0),
        llm_max_retries=_env_int("SUMMARY_BACKEND_LLM_MAX_RETRIES", 3),
        llm_retry_backoff_seconds=_env_float("SUMMARY_BACKEND_LLM_RETRY_BACKOFF_SECONDS", 2.0),
        llm_max_concurrency=max(1, _env_int("SUMMARY_BACKEND_LLM_MAX_CONCURRENCY", min(worker_threads, 2))),
        llm_pool_acquire_timeout_seconds=max(0.1, _env_float("SUMMARY_BACKEND_LLM_POOL_ACQUIRE_TIMEOUT_SECONDS", 1.0)),
        llm_pool_poll_interval_seconds=max(0.05, _env_float("SUMMARY_BACKEND_LLM_POOL_POLL_INTERVAL_SECONDS", 0.25)),
        llm_pool_retry_delay_seconds=max(1.0, _env_float("SUMMARY_BACKEND_LLM_POOL_RETRY_DELAY_SECONDS", 5.0)),
        dry_run=_env_bool("SUMMARY_BACKEND_DRY_RUN", not has_llm_config),
        chunk_target_estimated_tokens=_env_int("SUMMARY_BACKEND_CHUNK_TARGET_ESTIMATED_TOKENS", 6000),
        reduce_group_size=_env_int("SUMMARY_BACKEND_REDUCE_GROUP_SIZE", 8),
        max_enqueue_nodes_per_advance=_env_int("SUMMARY_BACKEND_MAX_ENQUEUE_NODES_PER_ADVANCE", 500),
        queued_node_requeue_after_seconds=_env_float("SUMMARY_BACKEND_QUEUED_NODE_REQUEUE_AFTER_SECONDS", 30.0),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()
