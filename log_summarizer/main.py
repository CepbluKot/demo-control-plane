"""CLI-точка входа для Log Summarizer.

Использование:
    python -m log_summarizer.main \\
        --model qwen2.5-72b-instruct \\
        --api-base http://localhost:8000 \\
        --api-key sk-placeholder \\
        --context "payments service OOM at ~14:30 UTC" \\
        --start "2024-01-15T14:00:00" \\
        --end   "2024-01-15T15:00:00" \\
        --logs-sql  "SELECT ... FROM logs WHERE ..." \\
        --ch-host   localhost \\
        --ch-port   8123 \\
        --output    report.md

Или короткий вариант через переменные окружения (см. --help).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="log-summarizer",
        description="LLM-powered incident log analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── ClickHouse ─────────────────────────────────────────────────────
    ch = p.add_argument_group("ClickHouse")
    ch.add_argument("--ch-host", default=os.getenv("CH_HOST", "localhost"))
    ch.add_argument("--ch-port", type=int, default=int(os.getenv("CH_PORT", "8123")))
    ch.add_argument("--ch-user", default=os.getenv("CH_USER", "default"))
    ch.add_argument("--ch-password", default=os.getenv("CH_PASSWORD", ""))
    ch.add_argument("--ch-database", default=os.getenv("CH_DATABASE", "default"))

    # ── SQL-шаблоны ────────────────────────────────────────────────────
    sql = p.add_argument_group("SQL templates")
    sql.add_argument(
        "--logs-sql",
        default=os.getenv("LOGS_SQL"),
        help=(
            "SQL template with {start_time}, {end_time}, {limit}, {offset} placeholders. "
            "Can also be a path to a .sql file."
        ),
    )
    sql.add_argument(
        "--metrics-sql",
        default=os.getenv("METRICS_SQL"),
        help="Optional SQL template for metrics. Same placeholders. Path or inline SQL.",
    )

    # ── Инцидент ───────────────────────────────────────────────────────
    inc = p.add_argument_group("Incident")
    inc.add_argument(
        "--context",
        default=os.getenv("INCIDENT_CONTEXT", ""),
        help="Free-text description of the incident",
    )
    inc.add_argument(
        "--start",
        default=os.getenv("INCIDENT_START"),
        help="Incident start time in ISO8601 format (e.g. 2024-01-15T14:00:00)",
    )
    inc.add_argument(
        "--end",
        default=os.getenv("INCIDENT_END"),
        help="Incident end time in ISO8601 format",
    )

    # ── LLM ────────────────────────────────────────────────────────────
    llm = p.add_argument_group("LLM")
    llm.add_argument("--model", default=os.getenv("LLM_MODEL", "default-model"))
    llm.add_argument(
        "--api-base",
        default=os.getenv("LLM_API_BASE", "http://localhost:8000"),
    )
    llm.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "sk-placeholder"))
    llm.add_argument(
        "--context-tokens",
        type=int,
        default=int(os.getenv("LLM_CONTEXT_TOKENS", "150000")),
        help="Model context window in tokens",
    )
    llm.add_argument(
        "--tool-calling",
        action="store_true",
        default=os.getenv("LLM_TOOL_CALLING", "").lower() in ("1", "true", "yes"),
        help="Enable TOOLS mode in instructor (disable for vLLM)",
    )

    # ── Параметры пайплайна ────────────────────────────────────────────
    pipe = p.add_argument_group("Pipeline tuning")
    pipe.add_argument("--map-concurrency", type=int, default=5)
    pipe.add_argument("--batch-size", type=int, default=1000, help="Log rows per ClickHouse page")
    pipe.add_argument("--max-events-per-merge", type=int, default=30)
    pipe.add_argument("--max-reduce-rounds", type=int, default=15)

    # ── Вывод ──────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write report to this file (default: stdout)",
    )
    out.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    out.add_argument("--log-file", default=None, help="Optional log file path")

    return p.parse_args(argv)


def _read_sql(value: str | None) -> str:
    """Если value — путь к файлу, читаем его; иначе возвращаем как есть."""
    if not value:
        return ""
    path = Path(value)
    if path.exists() and path.suffix in (".sql", ".txt"):
        return path.read_text(encoding="utf-8").strip()
    return value.strip()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    # Пробуем несколько форматов
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse datetime: {value!r}. Expected ISO8601, e.g. 2024-01-15T14:00:00"
    )


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Настраиваем логирование до любых импортов пайплайна
    from log_summarizer.utils.logging import setup_pipeline_logging

    setup_pipeline_logging(level=args.log_level, log_file=args.log_file)

    from log_summarizer.config import PipelineConfig
    from log_summarizer.orchestrator import PipelineOrchestrator

    # Подготовка конфига
    logs_sql = _read_sql(args.logs_sql)
    if not logs_sql:
        print("ERROR: --logs-sql is required", file=sys.stderr)
        return 2

    config = PipelineConfig(
        logs_sql_template=logs_sql,
        metrics_sql_template=_read_sql(args.metrics_sql) or None,
        incident_context=args.context,
        incident_start=_parse_dt(args.start),
        incident_end=_parse_dt(args.end),
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        max_context_tokens=args.context_tokens,
        batch_size=args.batch_size,
        map_concurrency=args.map_concurrency,
        max_events_per_merge=args.max_events_per_merge,
        max_reduce_rounds=args.max_reduce_rounds,
        model_supports_tool_calling=args.tool_calling,
    )

    # ClickHouse клиент
    try:
        import clickhouse_connect  # type: ignore[import-not-found]

        ch_client = clickhouse_connect.get_client(
            host=args.ch_host,
            port=args.ch_port,
            username=args.ch_user,
            password=args.ch_password,
            database=args.ch_database,
        )
    except ImportError:
        print(
            "ERROR: clickhouse_connect not installed. Run: pip install clickhouse-connect",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR: Cannot connect to ClickHouse: {exc}", file=sys.stderr)
        return 1

    # Запуск пайплайна
    orchestrator = PipelineOrchestrator(ch_client, config)
    report = await orchestrator.run()

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
