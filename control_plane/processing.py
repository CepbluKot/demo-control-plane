import logging
import importlib
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import time

from .config import (
    ALERT_CALLABLE,
    SUMMARY_LOG_CHARS,
    SUMMARIZER_CALLABLE,
    TEST_MODE,
    TEST_MODE_LOGS_PER_BATCH,
    TEST_MODE_SUMMARY_BATCHES,
)
from .summarizer import do_summary as default_do_summary
from .trace import log_event

logger = logging.getLogger(__name__)


def _build_test_mode_batches(
    *,
    period_start: datetime,
    period_end: datetime,
    service: str,
    batch_total: int,
    logs_per_batch: int,
) -> List[Dict[str, Any]]:
    safe_batches = max(int(batch_total), 1)
    safe_logs_per_batch = max(int(logs_per_batch), 1)
    total_logs = safe_batches * safe_logs_per_batch
    window_seconds = max(int((period_end - period_start).total_seconds()), total_logs + 1)
    batch_titles = (
        "Контекст перед деградацией",
        "Рост latency и retry",
        "Сбой зависимостей",
        "Критическая фаза перед выбросом",
    )

    def _level_for(batch_idx: int, row_idx: int) -> str:
        phase = (batch_idx + 1) / safe_batches
        if phase < 0.35:
            levels = ("INFO", "INFO", "WARN")
        elif phase < 0.75:
            levels = ("INFO", "WARN", "ERROR")
        else:
            levels = ("WARN", "ERROR", "ERROR", "CRITICAL")
        return levels[(row_idx + batch_idx) % len(levels)]

    def _message_for(level: str, event_idx: int) -> str:
        if level == "CRITICAL":
            return f"critical incident #{event_idx}: request queue overflow and timeout storm"
        if level == "ERROR":
            return f"error #{event_idx}: upstream timeout while processing request"
        if level == "WARN":
            return f"warn #{event_idx}: retry rate increased above baseline"
        return f"info #{event_idx}: background processing in progress"

    batches: List[Dict[str, Any]] = []
    for batch_idx in range(safe_batches):
        title = batch_titles[min(batch_idx, len(batch_titles) - 1)]
        batch_rows: List[Dict[str, Any]] = []
        for row_idx in range(safe_logs_per_batch):
            global_idx = batch_idx * safe_logs_per_batch + row_idx + 1
            sec_offset = int(global_idx * window_seconds / (total_logs + 1))
            ts = period_start + timedelta(seconds=sec_offset)
            level = _level_for(batch_idx, row_idx)
            batch_rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "level": level,
                    "message": _message_for(level, global_idx),
                    "service": service,
                    "pod": f"{service}-pod-{1 + (global_idx % 3)}",
                    "container": "app",
                    "node": f"node-{1 + (global_idx % 5):02d}",
                    "cluster": "demo-cluster",
                }
            )

        batches.append(
            {
                "batch_summary": f"[Batch {batch_idx + 1}] {title}",
                "batch_logs": batch_rows,
            }
        )
    return batches


def _extract_batch_summaries(summary_result: Any, fallback_text: str) -> List[str]:
    candidates: List[Any] = []
    if isinstance(summary_result, dict):
        for key in (
            "map_summaries",
            "batch_summaries",
            "chunk_summaries",
            "partial_summaries",
            "map_results",
            "chunks",
        ):
            value = summary_result.get(key)
            if isinstance(value, list):
                candidates = value
                break
    else:
        for key in (
            "map_summaries",
            "batch_summaries",
            "chunk_summaries",
            "partial_summaries",
            "map_results",
            "chunks",
        ):
            value = getattr(summary_result, key, None)
            if isinstance(value, list):
                candidates = value
                break

    normalized: List[str] = []
    for item in candidates:
        if isinstance(item, str):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            text = (
                item.get("summary")
                or item.get("text")
                or item.get("content")
                or item.get("chunk_summary")
                or item.get("map_summary")
            )
            if text:
                normalized.append(str(text))
            continue
        text = getattr(item, "summary", None) or getattr(item, "text", None)
        if text:
            normalized.append(str(text))

    if normalized:
        return normalized
    return [fallback_text] if fallback_text else []


def _extract_map_batches(summary_result: Any, fallback_text: str) -> List[Dict[str, Any]]:
    candidates: List[Any] = []
    if isinstance(summary_result, dict):
        for key in (
            "map_batches",
            "batch_details",
            "chunk_details",
            "map_results",
            "chunks",
        ):
            value = summary_result.get(key)
            if isinstance(value, list):
                candidates = value
                break
    else:
        for key in (
            "map_batches",
            "batch_details",
            "chunk_details",
            "map_results",
            "chunks",
        ):
            value = getattr(summary_result, key, None)
            if isinstance(value, list):
                candidates = value
                break

    out: List[Dict[str, Any]] = []
    for item in candidates:
        if isinstance(item, str):
            out.append(
                {
                    "batch_summary": item,
                    "batch_logs_count": 0,
                    "batch_logs": [],
                }
            )
            continue

        if isinstance(item, dict):
            summary = (
                item.get("batch_summary")
                or item.get("summary")
                or item.get("text")
                or item.get("content")
                or item.get("chunk_summary")
                or item.get("map_summary")
                or ""
            )
            logs = item.get("batch_logs")
            if logs is None:
                logs = item.get("logs")
            if logs is None:
                logs = item.get("rows")
            if not isinstance(logs, list):
                logs = []
            normalized_logs: List[Dict[str, Any]] = []
            for row in logs:
                if isinstance(row, dict):
                    normalized_logs.append(dict(row))
                else:
                    normalized_logs.append({"value": row})
            logs_count = item.get("batch_logs_count")
            if logs_count is None:
                logs_count = item.get("logs_count")
            if logs_count is None:
                logs_count = item.get("rows_count")
            if logs_count is None:
                logs_count = len(normalized_logs)
            out.append(
                {
                    "batch_summary": str(summary),
                    "batch_logs_count": int(logs_count),
                    "batch_logs": normalized_logs,
                }
            )
            continue

        summary = getattr(item, "summary", None) or getattr(item, "text", None) or str(item)
        rows = getattr(item, "rows", None)
        normalized_logs = rows if isinstance(rows, list) else []
        out.append(
            {
                "batch_summary": str(summary),
                "batch_logs_count": len(normalized_logs),
                "batch_logs": normalized_logs,
            }
        )

    if out:
        return out

    fallback_summaries = _extract_batch_summaries(summary_result, fallback_text)
    return [
        {
            "batch_summary": str(text),
            "batch_logs_count": 0,
            "batch_logs": [],
        }
        for text in fallback_summaries
    ]


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" in path:
        module_path, attr_name = path.split(":", 1)
    else:
        module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, attr_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Callable '{path}' не найден или не является callable")
    return fn


def _payload_for_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        target_key = key
        if key == "event":
            target_key = "payload_event"
        elif key == "level":
            target_key = "payload_level"
        if isinstance(value, list):
            out[f"{target_key}_len"] = len(value)
            continue
        if isinstance(value, dict):
            out[f"{target_key}_keys"] = list(value.keys())[:10]
            continue
        out[target_key] = value
    return out


def _call_summarizer_adapter(
    fn: Callable[..., Any],
    *,
    period_start_dt: datetime,
    period_end_dt: datetime,
    anomaly: Dict[str, Any],
    on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Any:
    attempts = [
        {
            "period_start": period_start_dt.isoformat(),
            "period_end": period_end_dt.isoformat(),
            "anomaly": anomaly,
            "on_progress": on_progress,
        },
        {
            "start_dt": period_start_dt,
            "end_dt": period_end_dt,
            "anomaly": anomaly,
            "on_progress": on_progress,
        },
        {
            "start_dt": period_start_dt,
            "end_dt": period_end_dt,
            "anomaly": anomaly,
        },
        {
            "start_dt": period_start_dt,
            "end_dt": period_end_dt,
        },
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return fn(**kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return fn()


def _call_alert_adapter(
    fn: Callable[..., Any],
    *,
    alert_text: str,
) -> Any:
    # Single supported signature for alert sender: fn(text: str)
    return fn(alert_text)


def process_anomalies(
    anomalies: List[Dict[str, Any]],
    lookback_minutes: int = 30,
    continue_on_error: bool = True,
    test_mode: Optional[bool] = None,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Control plane для обработки выбросов и генерации алертов.
    Args:
        anomalies: Список выбросов с минимум 'timestamp' полем
        lookback_minutes: Количество минут для lookback периода перед timestamp
        continue_on_error: Продолжать обработку при ошибке одного из выбросов
    Returns:
        Список результатов обработки каждого выброса
    """
    def _emit(event: str, payload: Dict[str, Any]) -> None:
        log_event(
            logger,
            "process_anomalies.emit",
            emitted_event=event,
            **_payload_for_log(payload),
        )
        if on_event is not None:
            on_event(event, payload)

    effective_test_mode = TEST_MODE if test_mode is None else bool(test_mode)
    _emit(
        "process_start",
        {
            "total": len(anomalies),
            "lookback_minutes": lookback_minutes,
            "test_mode": effective_test_mode,
        },
    )
    log_event(
        logger,
        "process_anomalies.start",
        anomalies=len(anomalies),
        lookback_minutes=lookback_minutes,
        continue_on_error=continue_on_error,
        test_mode=effective_test_mode,
    )
    results = []
    for i, anomaly in enumerate(anomalies):
        log_event(logger, "process_anomalies.item.start", index=i, anomaly=anomaly)
        result = {
            "anomaly": anomaly,
            "success": False,
            "summary": None,
            "alert_result": None,
            "error": None,
        }
        try:
            # Извлекаем timestamp из выброса
            timestamp_str = anomaly["timestamp"]
            timestamp_dt = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            )
            # Определяем период для сбора логов
            period_end = timestamp_dt
            period_start = period_end - timedelta(minutes=lookback_minutes)
            _emit(
                "anomaly_start",
                {
                    "index": i,
                    "total": len(anomalies),
                    "timestamp": timestamp_str,
                    "window_start": period_start.isoformat(),
                    "window_end": period_end.isoformat(),
                },
            )
            if effective_test_mode:
                _emit(
                    "summary_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                    },
                )
                _emit(
                    "map_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                    },
                )
                test_batches = _build_test_mode_batches(
                    period_start=period_start,
                    period_end=period_end,
                    service=str(anomaly.get("service") or "demo-service"),
                    batch_total=TEST_MODE_SUMMARY_BATCHES,
                    logs_per_batch=TEST_MODE_LOGS_PER_BATCH,
                )
                total_mock_logs = sum(len(batch.get("batch_logs", [])) for batch in test_batches)
                processed_mock_logs = 0
                for batch_idx, batch in enumerate(test_batches):
                    batch_logs = batch.get("batch_logs", [])
                    processed_mock_logs += len(batch_logs)
                    _emit(
                        "map_batch",
                        {
                            "index": i,
                            "timestamp": timestamp_str,
                            "mode": "mock",
                            "batch_index": batch_idx,
                            "batch_total": len(test_batches),
                            "batch_summary": batch.get("batch_summary", ""),
                            "batch_logs_count": len(batch_logs),
                            "batch_logs": batch_logs,
                            "rows_processed": processed_mock_logs,
                            "rows_total": total_mock_logs,
                        },
                    )
                _emit(
                    "map_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                        "batch_total": len(test_batches),
                        "rows_processed": processed_mock_logs,
                        "rows_total": total_mock_logs,
                    },
                )
                _emit(
                    "reduce_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                    },
                )
                summary = (
                    "[TEST MODE] Summary for anomaly at "
                    f"{timestamp_str} (window {period_start.isoformat()}..{period_end.isoformat()}, "
                    f"batches={len(test_batches)}, logs={sum(len(b.get('batch_logs', [])) for b in test_batches)})"
                )
                result["summary"] = summary
                preview = summary[:SUMMARY_LOG_CHARS]
                logger.info(
                    "do_summary.mock: anomaly_ts=%s, summary_len=%s, preview=%s",
                    timestamp_str,
                    len(summary),
                    preview,
                )
                _emit(
                    "summary_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                        "summary_len": len(summary),
                        "summary_preview": preview,
                        "rows_processed": processed_mock_logs,
                        "rows_total": total_mock_logs,
                    },
                )
                _emit(
                    "reduce_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                        "summary": summary,
                    },
                )
                alert_text = f"Время выброса: {timestamp_str}\n\n{summary}"
                result["alert_text"] = alert_text
                _emit(
                    "notification_ready",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                        "notification_text": alert_text,
                    },
                )
                _emit(
                    "alert_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                    },
                )
                result["alert_result"] = {
                    "status": "mocked",
                    "message_preview": alert_text[:120],
                }
                log_event(logger, "process_anomalies.mock_alert", index=i, alert=result["alert_result"])
                _emit(
                    "alert_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                        "alert_result": result["alert_result"],
                    },
                )
            else:
                # Генерируем summary логов через адаптер из env или существующий суммаризатор
                logger.info(
                    "do_summary.start: anomaly_ts=%s, window_start=%s, window_end=%s",
                    timestamp_str,
                    period_start.isoformat(),
                    period_end.isoformat(),
                )
                _emit(
                    "summary_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                    },
                )
                t0 = time.time()
                live_state: Dict[str, Any] = {
                    "map_started": False,
                    "map_done": False,
                    "reduce_started": False,
                    "reduce_done": False,
                    "reduce_summary": None,
                    "batch_count": 0,
                    "rows_processed": 0,
                    "rows_total": None,
                }

                def _on_summary_progress(event: str, payload: Dict[str, Any]) -> None:
                    if not isinstance(payload, dict):
                        payload = {}
                    if event == "map_start":
                        live_state["map_started"] = True
                    elif event == "map_batch":
                        live_state["map_started"] = True
                        live_state["batch_count"] = int(live_state["batch_count"]) + 1
                    elif event == "map_done":
                        live_state["map_done"] = True
                    elif event == "reduce_start":
                        live_state["reduce_started"] = True
                    elif event == "reduce_done":
                        live_state["reduce_done"] = True
                        if payload.get("summary") is not None:
                            live_state["reduce_summary"] = payload.get("summary")

                    if payload.get("rows_processed") is not None:
                        live_state["rows_processed"] = payload.get("rows_processed")
                    if payload.get("rows_total") is not None:
                        live_state["rows_total"] = payload.get("rows_total")

                    event_payload = {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                    }
                    event_payload.update(payload)
                    if event in ("map_start", "map_batch", "map_done", "reduce_start"):
                        _emit(event, event_payload)

                    progress_rows = payload.get("rows_processed")
                    if progress_rows is None:
                        progress_rows = payload.get("rows_fetched")
                    if progress_rows is not None:
                        _emit(
                            "summary_progress",
                            {
                                "index": i,
                                "timestamp": timestamp_str,
                                "mode": "real",
                                "rows_processed": progress_rows,
                                "rows_total": payload.get("rows_total"),
                            },
                        )

                if SUMMARIZER_CALLABLE:
                    logger.info("Using custom summarizer callable: %s", SUMMARIZER_CALLABLE)
                    summarizer_fn = _load_callable(SUMMARIZER_CALLABLE)
                    summary_result = _call_summarizer_adapter(
                        summarizer_fn,
                        period_start_dt=period_start,
                        period_end_dt=period_end,
                        anomaly=anomaly,
                        on_progress=_on_summary_progress,
                    )
                else:
                    summary_result = default_do_summary(
                        start_dt=period_start,
                        end_dt=period_end,
                        anomaly=anomaly,
                        on_progress=_on_summary_progress,
                    )
                elapsed = time.time() - t0

                # Извлекаем итоговый summary
                if isinstance(summary_result, dict):
                    summary = summary_result.get("summary", str(summary_result))
                    rows_processed_result = summary_result.get("rows_processed")
                    rows_total_result = summary_result.get("rows_total_estimate")
                else:
                    summary = getattr(summary_result, "summary", str(summary_result))
                    rows_processed_result = getattr(summary_result, "rows_processed", None)
                    rows_total_result = getattr(summary_result, "rows_total_estimate", None)
                summary = str(summary)
                result["summary"] = summary
                preview = summary[:SUMMARY_LOG_CHARS] if isinstance(summary, str) else str(summary)
                map_batches = _extract_map_batches(summary_result, preview)
                if rows_processed_result is None:
                    rows_processed_result = live_state.get("rows_processed")
                if rows_total_result is None:
                    rows_total_result = live_state.get("rows_total")

                processed_rows = int(rows_processed_result or 0)
                if not live_state.get("map_started"):
                    _emit(
                        "map_start",
                        {
                            "index": i,
                            "timestamp": timestamp_str,
                            "mode": "real",
                            "rows_processed": 0,
                            "rows_total": rows_total_result,
                        },
                    )
                if int(live_state.get("batch_count", 0)) == 0:
                    for batch_idx, batch_info in enumerate(map_batches):
                        batch_logs = batch_info.get("batch_logs", [])
                        batch_logs_count = int(batch_info.get("batch_logs_count", len(batch_logs)))
                        processed_rows += batch_logs_count
                        _emit(
                            "map_batch",
                            {
                                "index": i,
                                "timestamp": timestamp_str,
                                "mode": "real",
                                "batch_index": batch_idx,
                                "batch_total": len(map_batches),
                                "batch_summary": batch_info.get("batch_summary", ""),
                                "batch_logs_count": batch_logs_count,
                                "batch_logs": batch_logs,
                                "rows_processed": processed_rows,
                                "rows_total": rows_total_result,
                            },
                        )
                if not live_state.get("map_done"):
                    _emit(
                        "map_done",
                        {
                            "index": i,
                            "timestamp": timestamp_str,
                            "mode": "real",
                            "batch_total": len(map_batches),
                            "rows_processed": processed_rows,
                            "rows_total": rows_total_result,
                        },
                    )
                if not live_state.get("reduce_started"):
                    _emit(
                        "reduce_start",
                        {
                            "index": i,
                            "timestamp": timestamp_str,
                            "mode": "real",
                            "rows_processed": processed_rows,
                            "rows_total": rows_total_result,
                        },
                    )
                logger.info(
                    "do_summary.done: anomaly_ts=%s, seconds=%.2f, summary_len=%s, preview=%s",
                    timestamp_str,
                    elapsed,
                    len(summary) if isinstance(summary, str) else "n/a",
                    preview,
                )
                _emit(
                    "summary_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                        "elapsed_sec": round(elapsed, 3),
                        "summary_len": len(summary) if isinstance(summary, str) else None,
                        "summary_preview": preview,
                        "rows_processed": processed_rows,
                        "rows_total": rows_total_result,
                    },
                )
                reduce_summary = (
                    live_state.get("reduce_summary")
                    if live_state.get("reduce_summary") is not None
                    else summary
                )
                _emit(
                    "reduce_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                        "summary": reduce_summary,
                        "rows_processed": processed_rows,
                        "rows_total": rows_total_result,
                    },
                )
                # Формируем текст алерта с summary и timestamp
                alert_text = f"Время выброса: {timestamp_str}\n\n{summary}"
                result["alert_text"] = alert_text
                _emit(
                    "notification_ready",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                        "notification_text": alert_text,
                    },
                )
                # Генерируем и отправляем алерт через существующий генератор
                _emit(
                    "alert_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                    },
                )
                if ALERT_CALLABLE:
                    logger.info("Using custom alert callable: %s", ALERT_CALLABLE)
                    alert_fn = _load_callable(ALERT_CALLABLE)
                    alert_result = _call_alert_adapter(
                        alert_fn,
                        alert_text=alert_text,
                    )
                else:
                    from .alerts import make_alert

                    alert_result = make_alert(alert_text)
                result["alert_result"] = alert_result
                log_event(logger, "process_anomalies.alert_sent", index=i, alert_result=alert_result)
                _emit(
                    "alert_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                        "alert_result": alert_result,
                    },
                )
            result["success"] = True
            logger.info(f"Successfully processed anomaly {i+1}/{len(anomalies)}")
            _emit(
                "anomaly_done",
                {
                    "index": i,
                    "timestamp": timestamp_str,
                    "success": True,
                },
            )
        except Exception as exc:
            error_msg = f"Error processing anomaly {i+1}: {str(exc)}"
            result["error"] = error_msg
            logger.exception(error_msg)
            _emit(
                "anomaly_error",
                {
                    "index": i,
                    "timestamp": anomaly.get("timestamp"),
                    "error": error_msg,
                },
            )
            if not continue_on_error:
                raise
        finally:
            results.append(result)
            log_event(
                logger,
                "process_anomalies.item.done",
                index=i,
                success=result["success"],
                has_error=bool(result["error"]),
            )
    log_event(logger, "process_anomalies.done", processed=len(results))
    _emit("process_done", {"processed": len(results)})
    return results
