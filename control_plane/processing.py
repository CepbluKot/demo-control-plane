import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
import time

from .config import SUMMARY_LOG_CHARS, TEST_MODE
from .trace import log_event

logger = logging.getLogger(__name__)


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
                alert_text = f"Время выброса: {timestamp_str}\n\n{summary}"
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
                from llm_log_summarizer import do_summary
                from make_alert import make_alert
                # Генерируем summary логов через существующий суммаризатор
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
                summary_result = do_summary.do_summary(
                    start_dt=period_start, end_dt=period_end
                )
                elapsed = time.time() - t0

                # Извлекаем итоговый summary
                summary = getattr(summary_result, "summary", str(summary_result))
                result["summary"] = summary
                preview = summary[:SUMMARY_LOG_CHARS] if isinstance(summary, str) else str(summary)
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
                # Формируем текст алерта с summary и timestamp
                alert_text = f"Время выброса: {timestamp_str}\n\n{summary}"
                # Генерируем и отправляем алерт через существующий генератор
                _emit(
                    "alert_start",
                    {
                        "index": i,
                        "timestamp": timestamp_str,
                        "mode": "real",
                    },
                )
                alert_result = make_alert(summary_text=alert_text)
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
