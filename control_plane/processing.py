import logging
import importlib
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import time

from .config import ALERT_CALLABLE, SUMMARY_LOG_CHARS, SUMMARIZER_CALLABLE, TEST_MODE
from .summarizer import do_summary as default_do_summary
from .trace import log_event

logger = logging.getLogger(__name__)


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


def _call_summarizer_adapter(
    fn: Callable[..., Any],
    *,
    period_start_dt: datetime,
    period_end_dt: datetime,
    anomaly: Dict[str, Any],
) -> Any:
    attempts = [
        {
            "period_start": period_start_dt.isoformat(),
            "period_end": period_end_dt.isoformat(),
            "anomaly": anomaly,
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
    summary: str,
    anomaly: Dict[str, Any],
) -> Any:
    attempts = [
        {
            "summary_text": alert_text,
            "summary": summary,
            "anomaly": anomaly,
        },
        {
            "summary_text": alert_text,
        },
        {
            "message": alert_text,
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
                test_batches = [
                    f"[Batch 1] Подготовка контекста перед {timestamp_str}",
                    f"[Batch 2] Аномальный рост метрики и сопутствующие события",
                    f"[Batch 3] Краткий вывод по вероятной причине выброса",
                ]
                for batch_idx, batch_summary in enumerate(test_batches):
                    _emit(
                        "map_batch",
                        {
                            "index": i,
                            "timestamp": timestamp_str,
                            "mode": "mock",
                            "batch_index": batch_idx,
                            "batch_total": len(test_batches),
                            "batch_summary": batch_summary,
                        },
                    )
                _emit(
                    "map_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "mock",
                        "batch_total": len(test_batches),
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
                    f"{timestamp_str} (window {period_start.isoformat()}..{period_end.isoformat()})"
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
                if SUMMARIZER_CALLABLE:
                    logger.info("Using custom summarizer callable: %s", SUMMARIZER_CALLABLE)
                    summarizer_fn = _load_callable(SUMMARIZER_CALLABLE)
                    summary_result = _call_summarizer_adapter(
                        summarizer_fn,
                        period_start_dt=period_start,
                        period_end_dt=period_end,
                        anomaly=anomaly,
                    )
                else:
                    summary_result = default_do_summary(
                        start_dt=period_start,
                        end_dt=period_end,
                        anomaly=anomaly,
                    )
                elapsed = time.time() - t0

                # Извлекаем итоговый summary
                summary = getattr(summary_result, "summary", str(summary_result))
                result["summary"] = summary
                preview = summary[:SUMMARY_LOG_CHARS] if isinstance(summary, str) else str(summary)
                map_batches = _extract_batch_summaries(summary_result, preview)
                _emit(
                    "map_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                    },
                )
                for batch_idx, batch_summary in enumerate(map_batches):
                    _emit(
                        "map_batch",
                        {
                            "index": i,
                            "timestamp": timestamp_str,
                            "mode": "real",
                            "batch_index": batch_idx,
                            "batch_total": len(map_batches),
                            "batch_summary": batch_summary,
                        },
                    )
                _emit(
                    "map_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                        "batch_total": len(map_batches),
                    },
                )
                _emit(
                    "reduce_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
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
                    },
                )
                _emit(
                    "reduce_done",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                        "summary": summary,
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
                        summary=summary,
                        anomaly=anomaly,
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
