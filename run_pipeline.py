"""
run_pipeline.py — запуск полного пайплайна log_summarizer.

Заполни CONFIG и запусти:
    python run_pipeline.py
    python run_pipeline.py --output report.md
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
#  CONFIG — меняй здесь
# ══════════════════════════════════════════════════════════════════════

# LLM
API_BASE = "http://localhost:8000"
API_KEY  = "sk-placeholder"
MODEL    = "PNX.QWEN3 235b a22b instruct"
MAX_CONTEXT_TOKENS = 100_000
MODEL_SUPPORTS_TOOL_CALLING = False   # False = JSON mode (безопасно для vLLM)

# ClickHouse
CH_HOST     = "localhost"
CH_PORT     = 8123
CH_USER     = "default"
CH_PASSWORD = ""
CH_DATABASE = "default"

# Инцидент
INCIDENT_CONTEXT = """
Airflow: массовые ошибки Pod creation failed (Forbidden) в kubernetes_executor.
DAG-раны зависают в state=running, воркеры убиваются SIGTERM.
"""
INCIDENT_START = datetime(2026, 3, 18, 2, 0, 0, tzinfo=timezone.utc)
INCIDENT_END   = datetime(2026, 3, 18, 3, 0, 0, tzinfo=timezone.utc)

# SQL для логов.
# Плейсхолдеры: {start_time}, {end_time}, {limit}, {offset}
# Колонку с текстом лога назови raw_line — тогда он передаётся в LLM дословно.
# Если колонка называется иначе — DataLoader попробует: message / msg / log / value.
LOGS_SQL = """
SELECT
    timestamp,
    log         AS raw_line
FROM airflow.logs
WHERE timestamp >= '{start_time}'
  AND timestamp <  '{end_time}'
ORDER BY timestamp
LIMIT {limit} OFFSET {offset}
"""

# SQL для метрик (опционально, можно оставить пустым)
METRICS_SQL = ""

# Параметры пайплайна
MAP_CONCURRENCY      = 5
BATCH_SIZE           = 1000   # строк за один SELECT
MAX_EVENTS_PER_MERGE = 30

# Куда писать отчёт (None = stdout)
OUTPUT_FILE = "report.md"

# Лог пайплайна (None = только stderr)
LOG_FILE  = "pipeline.log"
LOG_LEVEL = "INFO"   # INFO = основные шаги | DEBUG = токены/бюджеты/страницы CH

# Папка для артефактов прогонов: runs/{timestamp}/
# Внутри: llm/ map/ reduce/ chunks_meta.json report.md
# Пустая строка = не сохранять артефакты
RUNS_DIR = "runs"

# ══════════════════════════════════════════════════════════════════════


async def main(output_file: str | None) -> None:
    from log_summarizer.config import PipelineConfig
    from log_summarizer.orchestrator import PipelineOrchestrator
    from log_summarizer.utils.logging import setup_pipeline_logging

    setup_pipeline_logging(LOG_LEVEL, log_file=LOG_FILE or None)

    try:
        import clickhouse_connect
    except ImportError:
        raise SystemExit("clickhouse-connect не установлен: pip install clickhouse-connect")

    ch = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE,
    )

    config = PipelineConfig(
        logs_sql_template=LOGS_SQL.strip(),
        metrics_sql_template=METRICS_SQL.strip() or None,
        incident_context=INCIDENT_CONTEXT.strip(),
        incident_start=INCIDENT_START,
        incident_end=INCIDENT_END,
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        max_context_tokens=MAX_CONTEXT_TOKENS,
        model_supports_tool_calling=MODEL_SUPPORTS_TOOL_CALLING,
        map_concurrency=MAP_CONCURRENCY,
        batch_size=BATCH_SIZE,
        max_events_per_merge=MAX_EVENTS_PER_MERGE,
        runs_dir=RUNS_DIR,
    )

    report = await PipelineOrchestrator(ch, config).run()

    dest = output_file or OUTPUT_FILE
    if dest:
        Path(dest).write_text(report, encoding="utf-8")
        print(f"\nОтчёт сохранён: {dest}")
    else:
        print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", "-o", default=None, help="Файл для отчёта (по умолчанию из OUTPUT_FILE)")
    args = p.parse_args()
    asyncio.run(main(args.output))
