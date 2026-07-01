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


@dataclass(frozen=True)
class LlmProfileSettings:
    profile_id: str
    label: str
    api_base: str
    api_key: str
    default_model: str
    available_models: tuple[str, ...]


@dataclass(frozen=True)
class LlmModelOption:
    value: str
    label: str
    profile_id: str
    profile_label: str
    model: str
    api_base: str


def _load_llm_gateway_defaults() -> dict[str, object]:
    config_path = Path(__file__).with_name("llm_gateway_defaults.json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _profile_env_token(profile_id: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in profile_id.strip())
    return sanitized.upper()


def _read_profile_defaults(payload: dict[str, object]) -> tuple[str, tuple[LlmProfileSettings, ...]]:
    raw_profiles = payload.get("profiles")
    profiles: list[LlmProfileSettings] = []
    default_profile = str(payload.get("default_profile") or "").strip()
    if isinstance(raw_profiles, list):
        for index, raw_profile in enumerate(raw_profiles):
            if not isinstance(raw_profile, dict):
                continue
            profile_id = str(raw_profile.get("id") or raw_profile.get("profile_id") or "").strip() or f"profile_{index + 1}"
            label = str(raw_profile.get("label") or profile_id).strip() or profile_id
            api_base = str(raw_profile.get("api_base") or "").strip()
            api_key = str(raw_profile.get("api_key") or "").strip()
            default_model = str(raw_profile.get("default_model") or raw_profile.get("model") or "").strip()
            raw_available_models = raw_profile.get("available_models")
            available_models = tuple(
                str(model).strip()
                for model in raw_available_models
                if str(model).strip()
            ) if isinstance(raw_available_models, list) else ()
            if default_model and default_model not in available_models:
                available_models = (default_model, *available_models)
            profiles.append(
                LlmProfileSettings(
                    profile_id=profile_id,
                    label=label,
                    api_base=api_base,
                    api_key=api_key,
                    default_model=default_model,
                    available_models=available_models or ((default_model,) if default_model else ()),
                )
            )
    if profiles:
        if not default_profile or default_profile not in {profile.profile_id for profile in profiles}:
            default_profile = profiles[0].profile_id
        return default_profile, tuple(profiles)

    default_api_base = str(payload.get("api_base") or "").strip()
    default_api_key = str(payload.get("api_key") or "").strip()
    default_model = str(payload.get("default_model") or "").strip()
    raw_available_models = payload.get("available_models")
    available_models = tuple(
        str(model).strip()
        for model in raw_available_models
        if str(model).strip()
    ) if isinstance(raw_available_models, list) else ()
    if default_model and default_model not in available_models:
        available_models = (default_model, *available_models)
    if default_api_base or default_api_key or default_model or available_models:
        profile = LlmProfileSettings(
            profile_id="default",
            label="Default LLM",
            api_base=default_api_base,
            api_key=default_api_key,
            default_model=default_model,
            available_models=available_models or ((default_model,) if default_model else ()),
        )
        return "default", (profile,)
    return "", ()


def _load_llm_profiles_from_env(default_profiles: tuple[LlmProfileSettings, ...], default_profile_id: str) -> tuple[str, tuple[LlmProfileSettings, ...]]:
    configured_profile_ids = _env_csv("SUMMARY_BACKEND_LLM_PROFILES", "")
    if not configured_profile_ids:
        return default_profile_id, default_profiles

    profiles: list[LlmProfileSettings] = []
    for profile_id in configured_profile_ids:
        token = _profile_env_token(profile_id)
        prefix = f"SUMMARY_BACKEND_LLM_PROFILE__{token}__"
        label = _env(f"{prefix}LABEL", profile_id)
        api_base = _env(f"{prefix}API_BASE", "")
        api_key = _env(f"{prefix}API_KEY", "")
        default_model = _env(f"{prefix}DEFAULT_MODEL", _env(f"{prefix}MODEL", ""))
        available_models = _env_csv(f"{prefix}AVAILABLE_MODELS", default_model)
        if default_model and default_model not in available_models:
            available_models = (default_model, *available_models)
        profiles.append(
            LlmProfileSettings(
                profile_id=profile_id,
                label=label or profile_id,
                api_base=api_base,
                api_key=api_key,
                default_model=default_model,
                available_models=available_models or ((default_model,) if default_model else ()),
            )
        )

    env_default_profile = _env("SUMMARY_BACKEND_LLM_PROFILE_DEFAULT", "").strip()
    if env_default_profile and env_default_profile in {profile.profile_id for profile in profiles}:
        return env_default_profile, tuple(profiles)
    return profiles[0].profile_id if profiles else default_profile_id, tuple(profiles)


def build_llm_model_options(
    *,
    llm_profiles: tuple[LlmProfileSettings, ...],
    default_selection: str = "",
) -> tuple[LlmModelOption, ...]:
    profile_count = len(llm_profiles)
    options: list[LlmModelOption] = []
    seen_values: set[str] = set()
    for profile in llm_profiles:
        models = profile.available_models or ((profile.default_model,) if profile.default_model else ())
        for model in models:
            clean_model = str(model).strip()
            if not clean_model:
                continue
            value = clean_model if profile_count <= 1 else f"{profile.profile_id}/{clean_model}"
            if value in seen_values:
                continue
            seen_values.add(value)
            options.append(
                LlmModelOption(
                    value=value,
                    label=clean_model if profile_count <= 1 else f"{profile.label} - {clean_model}",
                    profile_id=profile.profile_id,
                    profile_label=profile.label,
                    model=clean_model,
                    api_base=profile.api_base,
                )
            )

    if not options and default_selection:
        options.append(
            LlmModelOption(
                value=default_selection,
                label=default_selection,
                profile_id="",
                profile_label="",
                model=default_selection,
                api_base="",
            )
        )
    return tuple(options)


def build_settings_llm_model_options(settings: "Settings") -> tuple[LlmModelOption, ...]:
    if len(settings.llm_profiles) <= 1:
        profile = settings.llm_profiles[0] if settings.llm_profiles else LlmProfileSettings(
            profile_id=settings.llm_default_profile or "default",
            label="Default LLM",
            api_base=settings.openai_api_base,
            api_key=settings.openai_api_key,
            default_model=settings.llm_model,
            available_models=settings.llm_models,
        )
        models = settings.llm_models or ((settings.llm_model,) if settings.llm_model else ())
        options = [
            LlmModelOption(
                value=model,
                label=model,
                profile_id=profile.profile_id,
                profile_label=profile.label,
                model=model,
                api_base=profile.api_base,
            )
            for model in models
            if model
        ]
        if options:
            return tuple(options)
    return build_llm_model_options(llm_profiles=settings.llm_profiles, default_selection=settings.llm_model)


def resolve_llm_model_option(settings: "Settings", requested_model: str | None) -> LlmModelOption:
    options = build_settings_llm_model_options(settings)
    if not options:
        clean = (requested_model or settings.llm_model).strip()
        return LlmModelOption(
            value=clean,
            label=clean,
            profile_id="",
            profile_label="",
            model=clean,
            api_base=settings.openai_api_base,
        )

    clean_requested = (requested_model or "").strip()
    default_option = next((option for option in options if option.value == settings.llm_model), options[0])
    if not clean_requested:
        return default_option
    exact = next((option for option in options if option.value == clean_requested), None)
    if exact:
        return exact
    matching_models = [option for option in options if option.model == clean_requested]
    if len(matching_models) == 1:
        return matching_models[0]
    return default_option


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
    llm_default_profile: str
    llm_profiles: tuple[LlmProfileSettings, ...]
    llm_model: str
    llm_models: tuple[str, ...]
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_retry_backoff_seconds: float
    llm_max_concurrency: int
    llm_assistant_max_concurrency: int
    llm_pool_acquire_timeout_seconds: float
    llm_pool_poll_interval_seconds: float
    llm_pool_retry_delay_seconds: float
    dry_run: bool

    chunk_target_estimated_tokens: int
    reduce_target_estimated_tokens: int
    reduce_group_size: int
    max_enqueue_nodes_per_advance: int
    queued_node_requeue_after_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    llm_gateway_defaults = _load_llm_gateway_defaults()
    default_profile_id, default_profiles = _read_profile_defaults(llm_gateway_defaults)
    llm_default_profile, llm_profiles = _load_llm_profiles_from_env(default_profiles, default_profile_id)

    legacy_default_profile = llm_profiles[0] if llm_profiles else LlmProfileSettings(
        profile_id=llm_default_profile or "default",
        label="Default LLM",
        api_base="",
        api_key="",
        default_model="",
        available_models=(),
    )
    if len(llm_profiles) <= 1:
        openai_api_base = _env("SUMMARY_BACKEND_OPENAI_API_BASE", _env("OPENAI_API_BASE_DB", legacy_default_profile.api_base))
        openai_api_key = _env("SUMMARY_BACKEND_OPENAI_API_KEY", _env("OPENAI_API_KEY_DB", legacy_default_profile.api_key))
        legacy_default_model = legacy_default_profile.default_model
        llm_model = _env("SUMMARY_BACKEND_LLM_MODEL", _env("LLM_MODEL_ID", legacy_default_model))
        llm_models = _env_csv("SUMMARY_BACKEND_LLM_MODELS", ",".join(legacy_default_profile.available_models or ((legacy_default_model,) if legacy_default_model else ())))
        if llm_model and llm_model not in llm_models:
            llm_models = (llm_model, *llm_models)
        if llm_profiles:
            profile = llm_profiles[0]
            llm_profiles = (
                LlmProfileSettings(
                    profile_id=profile.profile_id,
                    label=profile.label,
                    api_base=openai_api_base,
                    api_key=openai_api_key,
                    default_model=llm_model or profile.default_model,
                    available_models=llm_models or profile.available_models,
                ),
            )
            llm_default_profile = profile.profile_id
        elif openai_api_base or openai_api_key or llm_model or llm_models:
            llm_profiles = (
                LlmProfileSettings(
                    profile_id=llm_default_profile or "default",
                    label="Default LLM",
                    api_base=openai_api_base,
                    api_key=openai_api_key,
                    default_model=llm_model,
                    available_models=llm_models,
                ),
            )
            llm_default_profile = llm_profiles[0].profile_id
    else:
        model_options = build_llm_model_options(llm_profiles=llm_profiles)
        configured_default_selection = _env("SUMMARY_BACKEND_LLM_MODEL", _env("LLM_MODEL_ID", "")).strip()
        resolved_default_option = next((option for option in model_options if option.value == configured_default_selection), None)
        if resolved_default_option is None and configured_default_selection:
            matching = [option for option in model_options if option.model == configured_default_selection]
            if len(matching) == 1:
                resolved_default_option = matching[0]
        if resolved_default_option is None:
            default_profile = next((profile for profile in llm_profiles if profile.profile_id == llm_default_profile), llm_profiles[0])
            default_model = default_profile.default_model or (default_profile.available_models[0] if default_profile.available_models else "")
            default_value = f"{default_profile.profile_id}/{default_model}" if default_model else ""
            resolved_default_option = next((option for option in model_options if option.value == default_value), model_options[0] if model_options else None)
        llm_model = resolved_default_option.value if resolved_default_option else ""
        llm_models = tuple(option.value for option in model_options)
        default_profile = next((profile for profile in llm_profiles if profile.profile_id == llm_default_profile), llm_profiles[0])
        openai_api_base = default_profile.api_base
        openai_api_key = default_profile.api_key

    has_llm_config = bool(
        any(
            profile.api_base and profile.api_key and (profile.default_model or profile.available_models)
            for profile in llm_profiles
        )
    ) if llm_profiles else bool(openai_api_base and openai_api_key and llm_model)
    clickhouse_host = _env("SUMMARY_BACKEND_CLICKHOUSE_HOST", "localhost")
    clickhouse_port = _env_int("SUMMARY_BACKEND_CLICKHOUSE_PORT", 8123)
    clickhouse_username = _env("SUMMARY_BACKEND_CLICKHOUSE_USERNAME", "default")
    clickhouse_password = _env("SUMMARY_BACKEND_CLICKHOUSE_PASSWORD", "")
    clickhouse_database = _env("SUMMARY_BACKEND_CLICKHOUSE_DATABASE", "summary_test")
    clickhouse_secure = _env_bool("SUMMARY_BACKEND_CLICKHOUSE_SECURE", False)
    worker_processes = _env_int("SUMMARY_BACKEND_WORKER_PROCESSES", 1)
    worker_threads = _env_int("SUMMARY_BACKEND_WORKER_THREADS", 4)
    llm_max_concurrency = max(1, _env_int("SUMMARY_BACKEND_LLM_MAX_CONCURRENCY", 5))
    llm_assistant_max_concurrency = max(
        1,
        _env_int("SUMMARY_BACKEND_LLM_ASSISTANT_MAX_CONCURRENCY", llm_max_concurrency),
    )

    chunk_target_estimated_tokens = _env_int("SUMMARY_BACKEND_CHUNK_TARGET_ESTIMATED_TOKENS", 6000)
    reduce_target_estimated_tokens = _env_int(
        "SUMMARY_BACKEND_REDUCE_TARGET_ESTIMATED_TOKENS",
        min(chunk_target_estimated_tokens, 4000),
    )

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
        llm_default_profile=llm_default_profile,
        llm_profiles=llm_profiles,
        llm_model=llm_model,
        llm_models=llm_models,
        llm_timeout_seconds=_env_float("SUMMARY_BACKEND_LLM_TIMEOUT_SECONDS", 600.0),
        llm_max_retries=_env_int("SUMMARY_BACKEND_LLM_MAX_RETRIES", 3),
        llm_retry_backoff_seconds=_env_float("SUMMARY_BACKEND_LLM_RETRY_BACKOFF_SECONDS", 2.0),
        llm_max_concurrency=llm_max_concurrency,
        llm_assistant_max_concurrency=llm_assistant_max_concurrency,
        llm_pool_acquire_timeout_seconds=max(0.1, _env_float("SUMMARY_BACKEND_LLM_POOL_ACQUIRE_TIMEOUT_SECONDS", 1.0)),
        llm_pool_poll_interval_seconds=max(0.05, _env_float("SUMMARY_BACKEND_LLM_POOL_POLL_INTERVAL_SECONDS", 0.25)),
        llm_pool_retry_delay_seconds=max(1.0, _env_float("SUMMARY_BACKEND_LLM_POOL_RETRY_DELAY_SECONDS", 5.0)),
        dry_run=_env_bool("SUMMARY_BACKEND_DRY_RUN", not has_llm_config),
        chunk_target_estimated_tokens=chunk_target_estimated_tokens,
        reduce_target_estimated_tokens=reduce_target_estimated_tokens,
        reduce_group_size=_env_int("SUMMARY_BACKEND_REDUCE_GROUP_SIZE", 8),
        max_enqueue_nodes_per_advance=_env_int("SUMMARY_BACKEND_MAX_ENQUEUE_NODES_PER_ADVANCE", 500),
        queued_node_requeue_after_seconds=_env_float("SUMMARY_BACKEND_QUEUED_NODE_REQUEUE_AFTER_SECONDS", 30.0),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()
