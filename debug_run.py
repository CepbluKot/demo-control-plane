"""
debug_run.py — запуск пайплайна Logs Summarizer.
Всё в одном файле. Меняй CONFIG внизу и запускай:

    python debug_run.py
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
#  CONFIG — всё что нужно менять здесь
# ═══════════════════════════════════════════════════════════════

LLM_API_BASE = "http://localhost:8000/v1"        # OpenAI-compatible endpoint
LLM_API_KEY  = "sk-placeholder"
LLM_MODEL    = "PNX.QWEN3 235b a22b instruct"

CH_HOST     = "localhost"
CH_PORT     = 8123
CH_USER     = "default"
CH_PASSWORD = ""

# SQL: {period_start} и {period_end} подставляются автоматически
SQL = """
SELECT timestamp, log
FROM app.logs
WHERE timestamp >= '{period_start}' AND timestamp < '{period_end}'
ORDER BY timestamp
LIMIT 1000
"""

PERIOD_START = "2026-04-13T18:00:00+03:00"
PERIOD_END   = "2026-04-13T20:00:00+03:00"

SERVICE   = "demo-service"
USER_GOAL = ""           # описание инцидента / цель анализа

OUTPUT_DIR = "artifacts/debug_run"

# ── параметры батчей ──────────────────────────────────────────
DB_PAGE_SIZE    = 1000    # строк за один SELECT
LLM_BATCH_SIZE  = 200     # строк на один MAP-вызов LLM
MIN_LLM_BATCH   = 20      # минимальный батч при auto-shrink
REDUCE_GROUP    = 2       # сколько саммари сливать за раз в REDUCE
MAX_RETRIES     = 3
LLM_TIMEOUT     = 600.0   # секунд

# ── что генерировать в финальном отчёте ──────────────────────
FINAL_STRUCTURED = True
FINAL_FREEFORM   = True

# ═══════════════════════════════════════════════════════════════
#  КОНЕЦ CONFIG
# ═══════════════════════════════════════════════════════════════

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

# Патчим settings ДО любого импорта из приложения
from settings import settings
settings.OPENAI_API_BASE_DB            = LLM_API_BASE
settings.OPENAI_API_KEY_DB             = LLM_API_KEY
settings.LLM_MODEL_ID                  = LLM_MODEL
settings.CONTROL_PLANE_LOGS_CLICKHOUSE_HOST = CH_HOST
settings.CONTROL_PLANE_LOGS_CLICKHOUSE_PORT = CH_PORT
settings.CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY = SQL.strip()

from debug_logs_summarizer_pipeline import (
    _build_llm_call,
    _build_runtime_config,
    _normalize_summary_text,
    _run_final_sections_stage,
    _run_map_reduce_stage,
    _run_reduce_only_stage,
    _save_json,
    _save_text,
    _setup_logging,
)

OUT = Path(OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)

log = _setup_logging(True, log_file=OUT / "run.log")

# Все параметры в одном SimpleNamespace — никаких pydantic
args = SimpleNamespace(
    db_batch               = DB_PAGE_SIZE,
    llm_batch              = LLM_BATCH_SIZE,
    min_llm_batch          = MIN_LLM_BATCH,
    auto_shrink_on_400     = True,
    max_shrink_rounds      = 6,
    max_cell_chars         = 0,
    max_summary_chars      = 0,
    reduce_prompt_max_chars= 0,
    reduce_group_size      = REDUCE_GROUP,
    reduce_input_max_chars = 40_000,
    reduce_target_token_pct= 50,
    compression_target_pct = 50,
    compression_importance_threshold = 0.7,
    use_instructor         = True,
    model_supports_tool_calling = True,
    map_schema_repair_enabled   = False,
    map_schema_repair_attempts  = 0,
    llm_timeout            = LLM_TIMEOUT,
    max_retries            = MAX_RETRIES,
    final_max_retries      = MAX_RETRIES,
    final_llm_timeout      = LLM_TIMEOUT,
    skip_structured        = not FINAL_STRUCTURED,
    skip_freeform          = not FINAL_FREEFORM,
)

anomaly = {"service": SERVICE, "description": USER_GOAL}

log.info("=" * 60)
log.info("Запуск | период=[%s, %s) | out=%s", PERIOD_START, PERIOD_END, OUT.resolve())
log.info("=" * 60)

# ── Шаг 1: MAP + REDUCE ──────────────────────────────────────
stage = _run_map_reduce_stage(
    args              = args,
    period_start_iso  = PERIOD_START,
    period_end_iso    = PERIOD_END,
    anomaly           = anomaly,
    output_dir        = OUT,
    logger            = log,
    query_template_override = SQL.strip(),
)

map_summaries = list(stage.get("map_summaries") or [])
base_summary  = _normalize_summary_text(stage.get("summary"))
stats         = dict(stage.get("stats") or {})

log.info(
    "MAP+REDUCE готово | batches=%s | base_summary_len=%s",
    len(map_summaries),
    len(base_summary),
)

# ── Шаг 2: если REDUCE не дал результата — пересчитать ───────
if not base_summary and map_summaries:
    log.info("base_summary пустой — пересчитываем REDUCE из MAP-саммари")
    base_summary = _run_reduce_only_stage(
        args             = args,
        map_summaries    = map_summaries,
        period_start_iso = PERIOD_START,
        period_end_iso   = PERIOD_END,
        logger           = log,
        output_dir       = OUT,
    )
    _save_text(OUT / "summary_reduce.md", base_summary)

# ── Шаг 3: финальные секции ──────────────────────────────────
final = _run_final_sections_stage(
    args             = args,
    base_summary     = base_summary,
    map_summaries    = map_summaries,
    user_goal        = USER_GOAL,
    period_start_iso = PERIOD_START,
    period_end_iso   = PERIOD_END,
    stats            = stats,
    logger           = log,
    output_dir       = OUT,
)

# ── Сохранение результатов ───────────────────────────────────
if final.get("structured_summary"):
    _save_text(OUT / "final_structured.md", final["structured_summary"])
    log.info("Structured: %s", (OUT / "final_structured.md").resolve())

if final.get("freeform_summary"):
    _save_text(OUT / "final_freeform.md", final["freeform_summary"])
    log.info("Freeform: %s", (OUT / "final_freeform.md").resolve())

_save_json(OUT / "final_sections.json", final)
_save_json(OUT / "run_meta.json", {
    "period_start"        : PERIOD_START,
    "period_end"          : PERIOD_END,
    "map_summaries_count" : len(map_summaries),
    "base_summary_len"    : len(base_summary),
    "stats"               : stats,
    "output_dir"          : str(OUT.resolve()),
})

log.info("=" * 60)
log.info("Готово! Артефакты: %s", OUT.resolve())
log.info("=" * 60)
