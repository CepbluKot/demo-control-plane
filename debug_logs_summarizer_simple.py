from __future__ import annotations

"""
Максимально простой запуск полного пайплайна Logs Summarizer без UI.

Запуск:
  ./venv/bin/python debug_logs_summarizer_simple.py
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from settings import settings

from debug_logs_summarizer_pipeline import (
    _load_json,
    _normalize_summary_text,
    _run_final_sections_stage,
    _run_map_reduce_stage,
    _run_reduce_only_stage,
    _save_json,
    _save_text,
    _setup_logging,
)

MSK = timezone(timedelta(hours=3))

# Минимум ручных настроек:
ANOMALY_FILE = Path("artifacts/debug_logs_summarizer_simple/anomaly.json")
OUTPUT_DIR = Path("artifacts/debug_logs_summarizer_simple")
PERIOD_HOURS_BACK = 2
DEBUG_LOGS = True


def _runtime_args() -> SimpleNamespace:
    llm_timeout = float(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT", 1200.0))
    max_retries = int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES", -1))
    return SimpleNamespace(
        db_batch=max(int(getattr(settings, "CONTROL_PLANE_LOGS_PAGE_LIMIT", 1000) or 1000), 1),
        llm_batch=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE", 200) or 200),
            1,
        ),
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
        reduce_group_size=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_GROUP_SIZE", 2) or 2),
            2,
        ),
        reduce_input_max_chars=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_INPUT_MAX_CHARS", 40000) or 40000),
            1000,
        ),
        reduce_target_token_pct=max(
            min(int(getattr(settings, "CONTROL_PLANE_LLM_REDUCE_TARGET_TOKEN_PCT", 50) or 50), 95),
            10,
        ),
        compression_target_pct=max(
            min(int(getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_TARGET_PCT", 50) or 50), 95),
            10,
        ),
        compression_importance_threshold=min(
            max(float(getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_IMPORTANCE_THRESHOLD", 0.7) or 0.7), 0.0),
            1.0,
        ),
        use_instructor=bool(getattr(settings, "CONTROL_PLANE_LLM_USE_INSTRUCTOR", True)),
        model_supports_tool_calling=bool(
            getattr(settings, "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING", True)
        ),
        llm_timeout=llm_timeout,
        max_retries=max_retries,
        final_max_retries=3,
        final_llm_timeout=llm_timeout,
        skip_structured=False,
        skip_freeform=False,
    )


def main() -> int:
    logger = _setup_logging(DEBUG_LOGS)
    args = _runtime_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(MSK).replace(microsecond=0)
    start_iso = (now - timedelta(hours=PERIOD_HOURS_BACK)).isoformat()
    end_iso = now.isoformat()

    anomaly: Dict[str, Any] = _load_json(str(ANOMALY_FILE)) if ANOMALY_FILE.exists() else {}
    user_goal = str(anomaly.get("description") or anomaly.get("name") or "")

    logger.info(
        "simple full run start | period=[%s, %s) | out=%s",
        start_iso,
        end_iso,
        OUTPUT_DIR.resolve(),
    )

    stage_payload = _run_map_reduce_stage(
        args=args,
        period_start_iso=start_iso,
        period_end_iso=end_iso,
        anomaly=anomaly,
        output_dir=OUTPUT_DIR,
        logger=logger,
    )
    map_summaries: List[str] = list(stage_payload.get("map_summaries") or [])
    base_summary = _normalize_summary_text(stage_payload.get("summary"))
    stats: Dict[str, Any] = dict(stage_payload.get("stats") or {})

    if not base_summary and map_summaries:
        logger.info("base summary empty -> rebuild reduce from map summaries")
        base_summary = _run_reduce_only_stage(
            args=args,
            map_summaries=map_summaries,
            period_start_iso=start_iso,
            period_end_iso=end_iso,
            logger=logger,
        )
        _save_text(OUTPUT_DIR / "summary_reduce.md", base_summary)

    final_payload = _run_final_sections_stage(
        args=args,
        base_summary=base_summary,
        map_summaries=map_summaries,
        user_goal=user_goal,
        period_start_iso=start_iso,
        period_end_iso=end_iso,
        stats=stats,
        logger=logger,
    )

    if final_payload.get("structured_summary"):
        _save_text(OUTPUT_DIR / "final_structured.md", str(final_payload["structured_summary"]))
    if final_payload.get("freeform_summary"):
        _save_text(OUTPUT_DIR / "final_freeform.md", str(final_payload["freeform_summary"]))
    _save_json(OUTPUT_DIR / "final_sections.json", final_payload)
    _save_json(
        OUTPUT_DIR / "run_meta.json",
        {
            "period_start": start_iso,
            "period_end": end_iso,
            "map_summaries_count": len(map_summaries),
            "base_summary_len": len(base_summary or ""),
            "structured_len": len(str(final_payload.get("structured_summary") or "")),
            "freeform_len": len(str(final_payload.get("freeform_summary") or "")),
        },
    )
    logger.info("simple full run done | out=%s", OUTPUT_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

