from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from my_summarizer import (
    DEFAULT_SUMMARY_COLUMNS,
    PeriodLogSummarizer,
    SummarizerConfig,
    _build_db_fetch_page,
    _estimate_total_logs,
    _make_llm_call,
    _normalize_summary_text,
    _resolve_logs_fetch_mode,
    regenerate_reduce_summary_from_map_summaries,
)
from settings import settings
from ui.pages.logs_summary_page import (
    _generate_sectional_freeform_summary,
    _generate_sectional_structured_summary,
)

MSK = timezone(timedelta(hours=3))


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty datetime")
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid datetime: {text}")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(MSK)
    return parsed.to_pydatetime()


def _setup_logging(debug: bool) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger("debug_logs_summarizer_pipeline")
    logger.setLevel(level)
    return logger


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"anomaly file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _normalize_anomaly_for_logs(
    anomaly: Optional[Dict[str, Any]],
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(anomaly or {})
    service = str(normalized.get("service") or "").strip()
    if service:
        return normalized
    fallback_service = str(getattr(settings, "CONTROL_PLANE_FORECAST_SERVICE", "") or "").strip()
    normalized["service"] = fallback_service or "demo-service"
    if logger is not None:
        logger.info(
            "anomaly.service is empty -> fallback to %s",
            normalized["service"],
        )
    return normalized


def _load_map_summaries(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"map summaries file not found: {p}")
    if p.suffix.lower() == ".jsonl":
        summaries: List[str] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    continue
                summary = _normalize_summary_text(row.get("batch_summary") or row.get("summary"))
                if summary:
                    summaries.append(summary)
        return summaries
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_normalize_summary_text(item) for item in payload if _normalize_summary_text(item)]
    if isinstance(payload, dict):
        if isinstance(payload.get("map_summaries"), list):
            return [
                _normalize_summary_text(item)
                for item in payload["map_summaries"]
                if _normalize_summary_text(item)
            ]
        if isinstance(payload.get("summaries"), list):
            return [
                _normalize_summary_text(item)
                for item in payload["summaries"]
                if _normalize_summary_text(item)
            ]
    return []


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _build_runtime_config(args: argparse.Namespace) -> SummarizerConfig:
    return SummarizerConfig(
        page_limit=int(args.db_batch),
        llm_chunk_rows=int(args.llm_batch),
        min_llm_chunk_rows=max(int(args.min_llm_batch), 1),
        auto_shrink_on_400=bool(args.auto_shrink_on_400),
        max_shrink_rounds=max(int(args.max_shrink_rounds), 0),
        max_cell_chars=max(int(args.max_cell_chars), 0),
        max_summary_chars=max(int(args.max_summary_chars), 0),
        reduce_prompt_max_chars=max(int(args.reduce_prompt_max_chars), 0),
        keep_map_batches_in_memory=True,
        keep_map_summaries_in_result=True,
        map_workers=1,
        use_new_algorithm=True,
        reduce_group_size=max(int(args.reduce_group_size), 2),
        reduce_input_max_chars=max(int(args.reduce_input_max_chars), 1000),
        reduce_target_token_pct=max(min(int(args.reduce_target_token_pct), 95), 10),
        compression_target_pct=max(min(int(args.compression_target_pct), 95), 10),
        compression_importance_threshold=min(
            max(float(args.compression_importance_threshold), 0.0),
            1.0,
        ),
        use_instructor=bool(args.use_instructor),
        model_supports_tool_calling=bool(args.model_supports_tool_calling),
    )


def _build_llm_call(
    *,
    timeout: float,
    max_retries: int,
    fail_open_return_empty: bool,
    logger: logging.Logger,
) -> Any:
    def _on_retry(attempt: int, total: int, exc: Exception) -> None:
        logger.warning(
            "LLM retry callback | attempt=%s | total=%s | error=%s",
            attempt,
            total if total >= 0 else "∞",
            exc,
        )

    def _on_attempt(attempt: int, total: int, timeout_seconds: float) -> None:
        logger.info(
            "LLM attempt callback | attempt=%s | total=%s | timeout=%.1fs",
            attempt,
            total if total >= 0 else "∞",
            timeout_seconds,
        )

    def _on_result(
        attempt: int,
        total: int,
        success: bool,
        elapsed_sec: float,
        error_text: Optional[str],
    ) -> None:
        logger.info(
            "LLM result callback | attempt=%s | total=%s | success=%s | elapsed=%.2fs | error=%s",
            attempt,
            total if total >= 0 else "∞",
            success,
            elapsed_sec,
            error_text or "",
        )

    return _make_llm_call(
        max_retries=max_retries,
        on_retry=_on_retry,
        on_attempt=_on_attempt,
        on_result=_on_result,
        llm_timeout=float(timeout),
        fail_open_return_empty=bool(fail_open_return_empty),
    )


def _run_map_reduce_stage(
    *,
    args: argparse.Namespace,
    period_start_iso: str,
    period_end_iso: str,
    anomaly: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
    query_template_override: Optional[str] = None,
    metrics_context: str = "",
) -> Dict[str, Any]:
    anomaly = _normalize_anomaly_for_logs(anomaly, logger=logger)
    fetch_mode = _resolve_logs_fetch_mode()
    tail_limit = max(int(getattr(settings, "CONTROL_PLANE_LOGS_TAIL_LIMIT", 1000) or 1000), 1)
    fetch_errors: List[str] = []
    progress_events: List[Dict[str, Any]] = []
    original_query_template = str(getattr(settings, "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY", "") or "")
    if str(query_template_override or "").strip():
        settings.CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY = str(query_template_override or "").strip()

    def _on_progress(event: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(MSK).isoformat(),
            "event": event,
            "payload": payload,
        }
        progress_events.append(row)
        logger.info("progress | event=%s | payload=%s", event, json.dumps(payload, ensure_ascii=False))

    def _on_fetch_error(msg: str) -> None:
        fetch_errors.append(str(msg))
        _on_progress("fetch_error", {"error": str(msg)})

    try:
        total_rows_estimate = _estimate_total_logs(
            anomaly=anomaly,
            period_start=period_start_iso,
            period_end=period_end_iso,
            page_limit=int(args.db_batch),
            fetch_mode=fetch_mode,
            tail_limit=tail_limit,
        )
        logger.info(
            "estimate | rows_total=%s | fetch_mode=%s | tail_limit=%s",
            total_rows_estimate,
            fetch_mode,
            tail_limit,
        )

        db_fetch_page = _build_db_fetch_page(
            anomaly,
            fetch_mode=fetch_mode,
            tail_limit=tail_limit,
            on_error=_on_fetch_error,
        )
        llm_call = _build_llm_call(
            timeout=float(args.llm_timeout),
            max_retries=int(args.max_retries),
            fail_open_return_empty=False,
            logger=logger,
        )
        summarizer = PeriodLogSummarizer(
            db_fetch_page=db_fetch_page,
            llm_call=llm_call,
            config=_build_runtime_config(args),
            on_progress=_on_progress,
            prompt_context={
                "incident_start": period_start_iso,
                "incident_end": period_end_iso,
                "incident_description": str(anomaly.get("description") or ""),
                "alerts_list": json.dumps(anomaly, ensure_ascii=False),
                "metrics_context": str(metrics_context or ""),
                "source_name": str(anomaly.get("service") or "unknown"),
                "sql_query": str(getattr(settings, "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY", "") or ""),
                "time_column": str(getattr(settings, "CONTROL_PLANE_LOGS_TIMESTAMP_COLUMN", "timestamp")),
                "data_type": "",
                "llm_timeout": float(args.llm_timeout),
                "llm_max_retries": int(args.max_retries),
            },
        )

        result = summarizer.summarize_period(
            period_start=period_start_iso,
            period_end=period_end_iso,
            columns=list(DEFAULT_SUMMARY_COLUMNS),
            total_rows_estimate=total_rows_estimate,
        )
    finally:
        if str(query_template_override or "").strip():
            settings.CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY = original_query_template

    payload = {
        "summary": _normalize_summary_text(result.summary),
        "freeform_summary": _normalize_summary_text(result.freeform_summary),
        "map_summaries": list(result.map_summaries or []),
        "map_batches": list(result.map_batches or []),
        "fetch_errors": fetch_errors,
        "progress_events": progress_events,
        "stats": {
            "pages_fetched": int(result.pages_fetched),
            "rows_processed": int(result.rows_processed),
            "llm_calls": int(result.llm_calls),
            "reduce_rounds": int(result.reduce_rounds),
            "rows_total_estimate": total_rows_estimate,
        },
    }
    _save_json(output_dir / "stage_map_reduce.json", payload)
    _save_json(output_dir / "progress_events.json", progress_events)
    _save_text(output_dir / "summary_reduce.md", payload["summary"])
    return payload


def _run_reduce_only_stage(
    *,
    args: argparse.Namespace,
    map_summaries: Sequence[str],
    period_start_iso: str,
    period_end_iso: str,
    logger: logging.Logger,
) -> str:
    llm_call = _build_llm_call(
        timeout=float(args.llm_timeout),
        max_retries=int(args.max_retries),
        fail_open_return_empty=False,
        logger=logger,
    )
    summary = regenerate_reduce_summary_from_map_summaries(
        map_summaries=map_summaries,
        period_start=period_start_iso,
        period_end=period_end_iso,
        llm_call=llm_call,
        config=_build_runtime_config(args),
        prompt_context={
            "incident_start": period_start_iso,
            "incident_end": period_end_iso,
            "llm_timeout": float(args.llm_timeout),
            "llm_max_retries": int(args.max_retries),
        },
    )
    return _normalize_summary_text(summary)


def _run_final_sections_stage(
    *,
    args: argparse.Namespace,
    base_summary: str,
    map_summaries: Sequence[str],
    user_goal: str,
    period_start_iso: str,
    period_end_iso: str,
    stats: Dict[str, Any],
    logger: logging.Logger,
    metrics_context: str = "",
) -> Dict[str, Any]:
    map_summaries_text = "\n\n".join(
        f"[MAP SUMMARY #{idx + 1}]\n{_normalize_summary_text(item)}"
        for idx, item in enumerate(map_summaries)
        if _normalize_summary_text(item)
    )
    llm_call = _build_llm_call(
        timeout=float(args.final_llm_timeout),
        max_retries=int(args.final_max_retries),
        fail_open_return_empty=True,
        logger=logger,
    )
    out: Dict[str, Any] = {}
    if not bool(args.skip_structured):
        structured_summary, structured_sections = _generate_sectional_structured_summary(
            llm_call=llm_call,
            base_summary=base_summary,
            user_goal=user_goal,
            period_start=period_start_iso,
            period_end=period_end_iso,
            stats=stats,
            metrics_context=str(metrics_context or ""),
            map_summaries_text=map_summaries_text,
            logger=logger,
        )
        out["structured_summary"] = _normalize_summary_text(structured_summary)
        out["structured_sections"] = structured_sections
    if not bool(args.skip_freeform):
        freeform_summary, freeform_sections = _generate_sectional_freeform_summary(
            llm_call=llm_call,
            final_summary=base_summary,
            user_goal=user_goal,
            period_start=period_start_iso,
            period_end=period_end_iso,
            stats=stats,
            metrics_context=str(metrics_context or ""),
            map_summaries_text=map_summaries_text,
            logger=logger,
        )
        out["freeform_summary"] = _normalize_summary_text(freeform_summary)
        out["freeform_sections"] = freeform_sections
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    now = datetime.now(MSK).replace(microsecond=0)
    p = argparse.ArgumentParser(
        description="Standalone debugger for the real Logs Summarizer pipeline.",
    )
    p.add_argument("--mode", choices=["full", "reduce-only", "final-only"], default="full")
    p.add_argument("--period-start", default=(now - timedelta(hours=2)).isoformat())
    p.add_argument("--period-end", default=now.isoformat())
    p.add_argument("--anomaly-file", default="", help="JSON file with incident/anomaly payload.")
    p.add_argument("--map-summaries-file", default="", help="JSON/JSONL with MAP summaries.")
    p.add_argument("--base-summary-file", default="", help="Text file with prebuilt reduce summary.")
    p.add_argument("--output-dir", default="artifacts/debug_logs_summarizer")
    p.add_argument("--db-batch", type=int, default=int(getattr(settings, "CONTROL_PLANE_LOGS_PAGE_LIMIT", 1000)))
    p.add_argument(
        "--llm-batch",
        type=int,
        default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE", 200)),
    )
    p.add_argument(
        "--min-llm-batch",
        type=int,
        default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE", 20)),
    )
    p.add_argument("--max-retries", type=int, default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES", -1)))
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=float(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT", 1200.0)),
    )
    p.add_argument("--final-max-retries", type=int, default=3)
    p.add_argument(
        "--final-llm-timeout",
        type=float,
        default=float(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT", 1200.0)),
    )
    p.add_argument(
        "--reduce-group-size",
        type=int,
        default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_GROUP_SIZE", 2)),
    )
    p.add_argument(
        "--reduce-input-max-chars",
        type=int,
        default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_INPUT_MAX_CHARS", 40000)),
    )
    p.add_argument(
        "--reduce-target-token-pct",
        type=int,
        default=int(getattr(settings, "CONTROL_PLANE_LLM_REDUCE_TARGET_TOKEN_PCT", 50)),
    )
    p.add_argument(
        "--compression-target-pct",
        type=int,
        default=int(getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_TARGET_PCT", 50)),
    )
    p.add_argument(
        "--compression-importance-threshold",
        type=float,
        default=float(getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_IMPORTANCE_THRESHOLD", 0.7)),
    )
    p.add_argument("--max-shrink-rounds", type=int, default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SHRINK_ROUNDS", 6)))
    p.add_argument("--max-cell-chars", type=int, default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0)))
    p.add_argument("--max-summary-chars", type=int, default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0)))
    p.add_argument("--reduce-prompt-max-chars", type=int, default=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0)))
    p.add_argument("--auto-shrink-on-400", action=argparse.BooleanOptionalAction, default=bool(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_AUTO_SHRINK_ON_400", True)))
    p.add_argument("--use-instructor", action=argparse.BooleanOptionalAction, default=bool(getattr(settings, "CONTROL_PLANE_LLM_USE_INSTRUCTOR", True)))
    p.add_argument("--model-supports-tool-calling", action=argparse.BooleanOptionalAction, default=bool(getattr(settings, "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING", True)))
    p.add_argument("--skip-structured", action="store_true")
    p.add_argument("--skip-freeform", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = _setup_logging(bool(args.debug))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    period_start_iso = _parse_dt(args.period_start).isoformat()
    period_end_iso = _parse_dt(args.period_end).isoformat()
    if period_end_iso <= period_start_iso:
        raise ValueError("period_end must be greater than period_start")

    anomaly = _normalize_anomaly_for_logs(_load_json(args.anomaly_file), logger=logger)
    user_goal = str(anomaly.get("description") or anomaly.get("name") or "")
    stats: Dict[str, Any] = {}
    map_summaries: List[str] = []
    base_summary = ""

    logger.info(
        "run start | mode=%s | period=[%s, %s) | output_dir=%s",
        args.mode,
        period_start_iso,
        period_end_iso,
        output_dir,
    )

    if args.mode == "full":
        full_payload = _run_map_reduce_stage(
            args=args,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            anomaly=anomaly,
            output_dir=output_dir,
            logger=logger,
        )
        map_summaries = list(full_payload.get("map_summaries") or [])
        base_summary = _normalize_summary_text(full_payload.get("summary"))
        stats = dict(full_payload.get("stats") or {})
    elif args.mode == "reduce-only":
        map_summaries = _load_map_summaries(args.map_summaries_file)
        if not map_summaries:
            raise ValueError("reduce-only mode requires non-empty --map-summaries-file")
        base_summary = _run_reduce_only_stage(
            args=args,
            map_summaries=map_summaries,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            logger=logger,
        )
        _save_text(output_dir / "summary_reduce.md", base_summary)
    else:
        map_summaries = _load_map_summaries(args.map_summaries_file)
        if args.base_summary_file:
            base_summary = _normalize_summary_text(
                Path(args.base_summary_file).read_text(encoding="utf-8")
            )
        if not base_summary:
            raise ValueError("final-only mode requires --base-summary-file")

    if not base_summary and map_summaries:
        logger.info("base summary is empty, rebuilding from MAP summaries before final stage")
        base_summary = _run_reduce_only_stage(
            args=args,
            map_summaries=map_summaries,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            logger=logger,
        )
        _save_text(output_dir / "summary_reduce.md", base_summary)

    final_payload = _run_final_sections_stage(
        args=args,
        base_summary=base_summary,
        map_summaries=map_summaries,
        user_goal=user_goal,
        period_start_iso=period_start_iso,
        period_end_iso=period_end_iso,
        stats=stats,
        logger=logger,
    )

    _save_json(
        output_dir / "run_meta.json",
        {
            "mode": args.mode,
            "period_start": period_start_iso,
            "period_end": period_end_iso,
            "map_summaries_count": len(map_summaries),
            "base_summary_len": len(base_summary or ""),
            "structured_len": len(str(final_payload.get("structured_summary") or "")),
            "freeform_len": len(str(final_payload.get("freeform_summary") or "")),
        },
    )
    if final_payload.get("structured_summary"):
        _save_text(output_dir / "final_structured.md", str(final_payload["structured_summary"]))
    if final_payload.get("freeform_summary"):
        _save_text(output_dir / "final_freeform.md", str(final_payload["freeform_summary"]))
    _save_json(output_dir / "final_sections.json", final_payload)

    logger.info("run done | output_dir=%s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
