from __future__ import annotations

"""
Простой запуск полного пайплайна Logs Summarizer без UI.
Все параметры, которые обычно задаются в sidebar, можно задать ниже в SIDEBAR_PARAMS.

Запуск:
  ./venv/bin/python debug_logs_summarizer_simple.py
"""

from dataclasses import asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import socket
from types import SimpleNamespace
from typing import Any, Dict, List

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from settings import settings

from debug_logs_summarizer_pipeline import (
    _load_json,
    _normalize_summary_text,
    _parse_dt,
    _run_final_sections_stage,
    _run_map_reduce_stage,
    _run_reduce_only_stage,
    _save_json,
    _save_text,
    _setup_logging,
)

MSK = timezone(timedelta(hours=3))

OUTPUT_DIR = Path("artifacts/debug_logs_summarizer_simple")
ANOMALY_FILE = Path("artifacts/debug_logs_summarizer_simple/anomaly.json")
DEBUG_LOGS = True

def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _default_logs_queries() -> List[str]:
    query = (
        str(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_SQL", "") or "").strip()
        or str(getattr(settings, "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY", "") or "").strip()
    )
    return [query] if query else []


@pydantic_dataclass
class SidebarParams:
    # Аналог параметров из sidebar.
    logs_queries: List[str] = field(default_factory=_default_logs_queries)
    metrics_queries: List[str] = field(default_factory=list)
    alerts: List[Any] = field(default_factory=list)
    user_goal: str = ""
    service: str = str(getattr(settings, "CONTROL_PLANE_FORECAST_SERVICE", "") or "demo-service")

    # Период: "Явный диапазон (start/end)" или "Окно вокруг даты (±N минут)".
    period_mode: str = "Явный диапазон (start/end)"
    window_minutes: int = Field(default=120, ge=1)
    center_dt: str = ""
    start_dt: str = ""
    end_dt: str = ""

    # Параметры батчей/LLM.
    db_batch_size: int = Field(
        default=max(_safe_int(getattr(settings, "CONTROL_PLANE_LOGS_PAGE_LIMIT", 1000), 1000), 1),
        ge=1,
    )
    llm_batch_size: int = Field(
        default=max(
            _safe_int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE", 200), 200),
            1,
        ),
        ge=1,
    )
    llm_model_id: str = str(getattr(settings, "LLM_MODEL_ID", "") or "")
    use_instructor: bool = bool(getattr(settings, "CONTROL_PLANE_LLM_USE_INSTRUCTOR", True))
    model_supports_tool_calling: bool = bool(
        getattr(settings, "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING", True)
    )

    # Финальный отчёт.
    enable_final_structured: bool = True
    enable_final_freeform: bool = True
    enable_final_topics_sync: bool = True
    enable_final_instructor_report: bool = True

    # Reduce/compression.
    reduce_group_size: int = Field(
        default=max(
            _safe_int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_GROUP_SIZE", 2), 2),
            2,
        ),
        ge=2,
    )
    reduce_input_max_chars: int = Field(
        default=max(
            _safe_int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_INPUT_MAX_CHARS", 40000),
                40000,
            ),
            1000,
        ),
        ge=1000,
    )
    reduce_target_token_pct: int = Field(
        default=min(
            max(
                _safe_int(getattr(settings, "CONTROL_PLANE_LLM_REDUCE_TARGET_TOKEN_PCT", 50), 50),
                10,
            ),
            95,
        ),
        ge=10,
        le=95,
    )
    compression_target_pct: int = Field(
        default=min(
            max(
                _safe_int(getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_TARGET_PCT", 50), 50),
                10,
            ),
            95,
        ),
        ge=10,
        le=95,
    )
    compression_importance_threshold: float = Field(
        default=min(
            max(
                _safe_float(
                    getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_IMPORTANCE_THRESHOLD", 0.7),
                    0.7,
                ),
                0.0,
            ),
            1.0,
        ),
        ge=0.0,
        le=1.0,
    )

    max_retries: int = _safe_int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES", -1), -1)
    llm_timeout: float = Field(
        default=max(
            _safe_float(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT", 1200), 1200),
            1.0,
        ),
        gt=0.0,
    )

    # Прочее.
    enable_no_logs_hypothesis: bool = False
    demo_mode: bool = False
    demo_logs_count: int = Field(default=1000, ge=1)


SIDEBAR_PARAMS = SidebarParams()


def _resolve_period(params: SidebarParams) -> tuple[str, str]:
    now = datetime.now(MSK).replace(microsecond=0)
    period_mode = str(params.period_mode or "Явный диапазон (start/end)")
    if period_mode.startswith("Окно вокруг"):
        center_raw = str(params.center_dt or "").strip() or now.isoformat()
        center_dt = _parse_dt(center_raw)
        window_minutes = max(int(params.window_minutes or 120), 1)
        start_dt = center_dt - timedelta(minutes=window_minutes)
        end_dt = center_dt + timedelta(minutes=window_minutes)
    else:
        start_raw = str(params.start_dt or "").strip() or (now - timedelta(hours=2)).isoformat()
        end_raw = str(params.end_dt or "").strip() or now.isoformat()
        start_dt = _parse_dt(start_raw)
        end_dt = _parse_dt(end_raw)
    if end_dt <= start_dt:
        raise ValueError("period_end должен быть больше period_start")
    return start_dt.isoformat(), end_dt.isoformat()


def _runtime_args(params: SidebarParams) -> SimpleNamespace:
    llm_timeout = float(params.llm_timeout)
    return SimpleNamespace(
        db_batch=max(int(params.db_batch_size), 1),
        llm_batch=max(int(params.llm_batch_size), 1),
        min_llm_batch=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE", 20) or 20),
            1,
        ),
        auto_shrink_on_400=bool(
            getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_AUTO_SHRINK_ON_400", True)
        ),
        max_shrink_rounds=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SHRINK_ROUNDS", 6) or 6),
            0,
        ),
        max_cell_chars=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0) or 0),
            0,
        ),
        max_summary_chars=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0) or 0),
            0,
        ),
        reduce_prompt_max_chars=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0) or 0),
            0,
        ),
        reduce_group_size=max(int(params.reduce_group_size), 2),
        reduce_input_max_chars=max(int(params.reduce_input_max_chars), 1000),
        reduce_target_token_pct=max(min(int(params.reduce_target_token_pct), 95), 10),
        compression_target_pct=max(min(int(params.compression_target_pct), 95), 10),
        compression_importance_threshold=min(
            max(float(params.compression_importance_threshold), 0.0),
            1.0,
        ),
        use_instructor=bool(params.use_instructor),
        model_supports_tool_calling=bool(params.model_supports_tool_calling),
        llm_timeout=llm_timeout,
        max_retries=int(params.max_retries),
        final_max_retries=3,
        final_llm_timeout=llm_timeout,
        skip_structured=not bool(params.enable_final_structured),
        skip_freeform=not bool(params.enable_final_freeform),
    )


def _build_metrics_context(params: SidebarParams) -> str:
    rows = [str(item or "").strip() for item in list(params.metrics_queries or [])]
    rows = [item for item in rows if item]
    if not rows:
        return ""
    return "\n\n".join(f"[METRICS QUERY #{idx + 1}]\n{query}" for idx, query in enumerate(rows))


def _preflight_or_raise(selected_query: str) -> None:
    problems: List[str] = []
    api_base = str(getattr(settings, "OPENAI_API_BASE_DB", "") or "").strip()
    api_key = str(getattr(settings, "OPENAI_API_KEY_DB", "") or "").strip()
    query_template = str(selected_query or "").strip()
    ch_host = str(getattr(settings, "CONTROL_PLANE_LOGS_CLICKHOUSE_HOST", "") or "").strip()
    ch_port = int(getattr(settings, "CONTROL_PLANE_LOGS_CLICKHOUSE_PORT", 8123) or 8123)

    if not api_base:
        problems.append("Не задан OPENAI_API_BASE_DB")
    if not api_key:
        problems.append("Не задан OPENAI_API_KEY_DB")
    if not query_template:
        problems.append("Не задан SQL logs query (SIDEBAR_PARAMS['logs_queries'])")
    if not ch_host:
        problems.append("Не задан CONTROL_PLANE_LOGS_CLICKHOUSE_HOST")

    if ch_host:
        try:
            with socket.create_connection((ch_host, ch_port), timeout=2.0):
                pass
        except Exception as exc:
            problems.append(
                f"ClickHouse недоступен по {ch_host}:{ch_port} ({type(exc).__name__}: {exc})"
            )

    if problems:
        raise RuntimeError("Preflight не пройден:\n- " + "\n- ".join(problems))


def main() -> int:
    logger = _setup_logging(DEBUG_LOGS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    params = SIDEBAR_PARAMS
    args = _runtime_args(params)
    period_start_iso, period_end_iso = _resolve_period(params)
    logs_queries = [str(item or "").strip() for item in list(params.logs_queries or [])]
    logs_queries = [item for item in logs_queries if item]
    selected_logs_query = logs_queries[0] if logs_queries else str(
        getattr(settings, "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY", "") or ""
    ).strip()
    if len(logs_queries) > 1:
        logger.warning(
            "В debug runner используется только первый SQL logs query (из %s).",
            len(logs_queries),
        )

    model_id = str(params.llm_model_id or "").strip()
    if model_id:
        settings.LLM_MODEL_ID = model_id

    _preflight_or_raise(selected_logs_query)

    anomaly: Dict[str, Any] = _load_json(str(ANOMALY_FILE)) if ANOMALY_FILE.exists() else {}
    anomaly["service"] = str(params.service or anomaly.get("service") or "demo-service")
    user_goal = str(params.user_goal or anomaly.get("description") or anomaly.get("name") or "")
    if user_goal:
        anomaly["description"] = user_goal
    alerts = list(params.alerts or [])
    if alerts:
        anomaly["alerts"] = alerts
    metrics_context = _build_metrics_context(params)

    logger.info(
        "simple full run start | period=[%s, %s) | out=%s",
        period_start_iso,
        period_end_iso,
        OUTPUT_DIR.resolve(),
    )

    stage_payload = _run_map_reduce_stage(
        args=args,
        period_start_iso=period_start_iso,
        period_end_iso=period_end_iso,
        anomaly=anomaly,
        output_dir=OUTPUT_DIR,
        logger=logger,
        query_template_override=selected_logs_query,
        metrics_context=metrics_context,
    )
    map_summaries: List[str] = list(stage_payload.get("map_summaries") or [])
    base_summary = _normalize_summary_text(stage_payload.get("summary"))
    stats: Dict[str, Any] = dict(stage_payload.get("stats") or {})

    if not base_summary and map_summaries:
        logger.info("base summary empty -> rebuild reduce from map summaries")
        base_summary = _run_reduce_only_stage(
            args=args,
            map_summaries=map_summaries,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            logger=logger,
        )
        _save_text(OUTPUT_DIR / "summary_reduce.md", base_summary)

    final_payload = _run_final_sections_stage(
        args=args,
        base_summary=base_summary,
        map_summaries=map_summaries,
        user_goal=user_goal,
        period_start_iso=period_start_iso,
        period_end_iso=period_end_iso,
        stats=stats,
        logger=logger,
        metrics_context=metrics_context,
    )

    if final_payload.get("structured_summary"):
        _save_text(OUTPUT_DIR / "final_structured.md", str(final_payload["structured_summary"]))
    if final_payload.get("freeform_summary"):
        _save_text(OUTPUT_DIR / "final_freeform.md", str(final_payload["freeform_summary"]))
    _save_json(OUTPUT_DIR / "final_sections.json", final_payload)
    _save_json(
        OUTPUT_DIR / "run_meta.json",
        {
            "period_start": period_start_iso,
            "period_end": period_end_iso,
            "map_summaries_count": len(map_summaries),
            "base_summary_len": len(base_summary or ""),
            "structured_len": len(str(final_payload.get("structured_summary") or "")),
            "freeform_len": len(str(final_payload.get("freeform_summary") or "")),
            "logs_queries_count": len(logs_queries),
            "metrics_queries_count": len(list(params.metrics_queries or [])),
            "llm_model_id": str(getattr(settings, "LLM_MODEL_ID", "") or ""),
            "sidebar_params": asdict(params),
        },
    )
    logger.info("simple full run done | out=%s", OUTPUT_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
