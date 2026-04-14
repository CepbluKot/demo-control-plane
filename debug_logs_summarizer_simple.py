from __future__ import annotations

"""
Простой запуск полного пайплайна Logs Summarizer без UI.
Все параметры, которые обычно задаются в sidebar, можно задать ниже в RUN_CONFIG.

Запуск:
  ./venv/bin/python debug_logs_summarizer_simple.py
"""

from dataclasses import asdict, field
from datetime import datetime, timedelta, timezone
from enum import Enum
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

class PeriodMode(str, Enum):
    RANGE = "range"  # Явный диапазон через start_dt/end_dt
    WINDOW = "window"  # Окно вокруг center_dt ± window_minutes


@pydantic_dataclass
class DebugSimpleRunConfig:
    # Аналог параметров из sidebar.
    logs_queries: List[str] = field(
        default_factory=lambda: [
            (
                "SELECT timestamp, log "
                "FROM app.logs "
                "WHERE timestamp >= '{period_start}' AND timestamp < '{period_end}' "
                "ORDER BY timestamp "
                "LIMIT 1000"
            )
        ]
    )
    metrics_queries: List[str] = field(default_factory=list)
    alerts: List[Any] = field(default_factory=list)
    user_goal: str = ""
    service: str = "demo-service"

    # Режим периода: RANGE или WINDOW.
    period_mode: PeriodMode = PeriodMode.RANGE
    window_minutes: int = Field(default=120, ge=1)
    center_dt: str = ""
    start_dt: str = ""
    end_dt: str = ""

    # Параметры батчей/LLM.
    db_batch_size: int = Field(default=1000, ge=1)
    llm_batch_size: int = Field(default=200, ge=1)
    min_llm_batch: int = Field(default=20, ge=1)
    llm_model_id: str = "PNX.QWEN3 235b a22b instruct"
    use_instructor: bool = True
    model_supports_tool_calling: bool = True
    auto_shrink_on_400: bool = True
    auto_shrink_on_500: bool = True
    map_gateway_retry_cap: int = Field(default=3, ge=0)
    max_shrink_rounds: int = Field(default=6, ge=0)
    max_cell_chars: int = Field(default=0, ge=0)
    max_summary_chars: int = Field(default=0, ge=0)
    reduce_prompt_max_chars: int = Field(default=0, ge=0)

    # Финальный отчёт.
    enable_final_structured: bool = True
    enable_final_freeform: bool = True
    enable_final_topics_sync: bool = True
    enable_final_instructor_report: bool = True

    # Reduce/compression.
    reduce_group_size: int = Field(default=2, ge=2)
    reduce_input_max_chars: int = Field(default=40000, ge=1000)
    reduce_target_token_pct: int = Field(default=50, ge=10, le=95)
    compression_target_pct: int = Field(default=50, ge=10, le=95)
    compression_importance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    max_retries: int = 3
    llm_timeout: float = Field(default=1200.0, gt=0.0)
    final_max_retries: int = 3

    # Прочее.
    enable_no_logs_hypothesis: bool = False
    demo_mode: bool = False
    demo_logs_count: int = Field(default=1000, ge=1)


# Все параметры явно вынесены сюда, чтобы было удобно править под конкретный прогон.
RUN_CONFIG = DebugSimpleRunConfig(
    logs_queries=[
        (
            "SELECT timestamp, log "
            "FROM app.logs "
            "WHERE timestamp >= '{period_start}' AND timestamp < '{period_end}' "
            "ORDER BY timestamp "
            "LIMIT 1000"
        )
    ],
    metrics_queries=[],
    alerts=[],
    user_goal="",
    service="demo-service",
    period_mode=PeriodMode.RANGE,  # PeriodMode.RANGE | PeriodMode.WINDOW
    window_minutes=120,  # используется только в режиме PeriodMode.WINDOW
    center_dt="2026-04-13T19:00:00+03:00",  # MSK
    start_dt="2026-04-13T18:00:00+03:00",  # MSK
    end_dt="2026-04-13T20:00:00+03:00",  # MSK
    db_batch_size=1000,
    llm_batch_size=200,
    min_llm_batch=20,
    llm_model_id="PNX.QWEN3 235b a22b instruct",
    use_instructor=True,
    model_supports_tool_calling=True,
    auto_shrink_on_400=True,
    auto_shrink_on_500=True,
    map_gateway_retry_cap=3,
    max_shrink_rounds=6,
    max_cell_chars=0,
    max_summary_chars=0,
    reduce_prompt_max_chars=0,
    enable_final_structured=True,
    enable_final_freeform=True,
    enable_final_topics_sync=True,
    enable_final_instructor_report=True,
    reduce_group_size=2,
    reduce_input_max_chars=40000,
    reduce_target_token_pct=50,
    compression_target_pct=50,
    compression_importance_threshold=0.7,
    max_retries=3,
    llm_timeout=1200.0,
    final_max_retries=3,
    enable_no_logs_hypothesis=False,
    demo_mode=False,
    demo_logs_count=1000,
)


def _resolve_period(params: DebugSimpleRunConfig) -> tuple[str, str]:
    now = datetime.now(MSK).replace(microsecond=0)
    period_mode = params.period_mode or PeriodMode.RANGE
    if period_mode == PeriodMode.WINDOW:
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


def _runtime_args(params: DebugSimpleRunConfig) -> SimpleNamespace:
    llm_timeout = float(params.llm_timeout)
    return SimpleNamespace(
        db_batch=max(int(params.db_batch_size), 1),
        llm_batch=max(int(params.llm_batch_size), 1),
        min_llm_batch=max(int(params.min_llm_batch), 1),
        auto_shrink_on_400=bool(params.auto_shrink_on_400),
        auto_shrink_on_500=bool(params.auto_shrink_on_500),
        map_gateway_retry_cap=max(int(params.map_gateway_retry_cap), 0),
        max_shrink_rounds=max(int(params.max_shrink_rounds), 0),
        max_cell_chars=max(int(params.max_cell_chars), 0),
        max_summary_chars=max(int(params.max_summary_chars), 0),
        reduce_prompt_max_chars=max(int(params.reduce_prompt_max_chars), 0),
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
        final_max_retries=max(int(params.final_max_retries), 0),
        final_llm_timeout=llm_timeout,
        skip_structured=not bool(params.enable_final_structured),
        skip_freeform=not bool(params.enable_final_freeform),
    )


def _build_metrics_context(params: DebugSimpleRunConfig) -> str:
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
        problems.append("Не задан SQL logs query (RUN_CONFIG.logs_queries)")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(
        DEBUG_LOGS,
        log_file=OUTPUT_DIR / "debug_logs_summarizer_simple.log",
    )

    params = RUN_CONFIG
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
            output_dir=OUTPUT_DIR,
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
        output_dir=OUTPUT_DIR,
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
            "prompt_audit_dir": str((OUTPUT_DIR / "prompt_audit").resolve()),
            "run_config": asdict(params),
        },
    )
    logger.info("simple full run done | out=%s", OUTPUT_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
