"""CLI-точка входа для Log Summarizer.

Конфиг среды (CH, LLM) берётся из .env или переменных окружения через Settings.
Run-специфичные параметры передаются через CLI.

Использование:
    python -m log_summarizer.main \\
        --context "payments service OOM at ~14:30 UTC" \\
        --start "2024-01-15T14:00:00" \\
        --end   "2024-01-15T15:00:00" \\
        --logs-sql /path/to/query.sql \\
        --output   report.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="log-summarizer",
        description="LLM-powered incident log analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── SQL-шаблоны ────────────────────────────────────────────────────
    sql = p.add_argument_group("SQL templates")
    sql.add_argument(
        "--logs-sql",
        default=None,
        help="SQL template or path to .sql file. Placeholders: {last_ts}, {period_end}, {limit}, {raw_limit}.",
    )
    sql.add_argument(
        "--metrics-sql",
        default=None,
        help="Optional SQL template for metrics (path or inline).",
    )

    # ── Инцидент ───────────────────────────────────────────────────────
    inc = p.add_argument_group("Incident")
    inc.add_argument("--context", default="", help="Free-text description of the incident")
    inc.add_argument("--start", default=None, help="Incident start ISO8601, e.g. 2024-01-15T14:00:00")
    inc.add_argument("--end",   default=None, help="Incident end ISO8601")

    # ── Параметры пайплайна ────────────────────────────────────────────
    pipe = p.add_argument_group("Pipeline tuning")
    pipe.add_argument("--map-concurrency",      type=int, default=None)
    pipe.add_argument("--batch-size",           type=int, default=None, help="Log rows per ClickHouse page")
    pipe.add_argument("--max-events-per-merge", type=int, default=None)
    pipe.add_argument("--context-tokens",       type=int, default=None, help="Override LLM context window size")

    # ── Вывод ──────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--output", "-o", default=None, help="Write report to file (default: stdout)")
    out.add_argument("--log-file", default=None, help="Optional log file path")

    return p.parse_args(argv)


def _read_sql(value: str | None) -> str:
    if not value:
        return ""
    if "\n" in value or " " in value:
        return value.strip()
    try:
        path = Path(value)
        if path.exists() and path.suffix in (".sql", ".txt"):
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return value.strip()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
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
    raise ValueError(f"Cannot parse datetime: {value!r}. Expected ISO8601, e.g. 2024-01-15T14:00:00")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from log_summarizer.settings import Settings
    s = Settings()

    from log_summarizer.utils.logging import setup_pipeline_logging
    setup_pipeline_logging(level=s.log_level, log_file=args.log_file or s.log_file)

    from log_summarizer.config import PipelineConfig
    from log_summarizer.orchestrator import PipelineOrchestrator

    logs_sql = _read_sql(args.logs_sql)
    if not logs_sql:
        print("ERROR: --logs-sql is required", file=sys.stderr)
        return 2

    config = PipelineConfig(
        logs_sql_templates=[logs_sql],
        metrics_sql_template=_read_sql(args.metrics_sql) or None,
        incident_context=args.context,
        incident_start=_parse_dt(args.start),
        incident_end=_parse_dt(args.end),
        model=s.llm_model,
        api_base=s.llm_api_base,
        api_key=s.llm_api_key,
        max_context_tokens=args.context_tokens or s.llm_context_tokens,
        batch_size=args.batch_size or 1000,
        map_concurrency=args.map_concurrency or 5,
        max_events_per_merge=args.max_events_per_merge or 30,
        model_supports_tool_calling=s.llm_tool_calling,
    )

    try:
        import clickhouse_connect  # type: ignore[import-not-found]
        ch_client = clickhouse_connect.get_client(
            host=s.ch_host,
            port=s.ch_port,
            username=s.ch_user,
            password=s.ch_password,
            database=s.ch_database,
        )
    except ImportError:
        print("ERROR: clickhouse_connect not installed. Run: pip install clickhouse-connect", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Cannot connect to ClickHouse: {exc}", file=sys.stderr)
        return 1

    orchestrator = PipelineOrchestrator(ch_client, config)
    report = await orchestrator.run()

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}", file=sys.stderr)
    else:
        print(report)

    if s.save_to_ch:
        from log_summarizer.ch_writer import save_report
        save_report(
            host=s.ch_host,
            port=s.ch_port,
            user=s.ch_user,
            password=s.ch_password,
            result_database=s.result_database,
            run_id=s.airflow_run_id,
            incident_context=args.context,
            incident_start=args.start or "",
            incident_end=args.end or "",
            report=report,
            logs_sql=logs_sql,
        )

    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
