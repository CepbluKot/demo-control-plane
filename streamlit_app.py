import json
import logging
import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from my_summarizer import PeriodLogSummarizer, SummarizerConfig, _make_llm_call, _query_logs_df
from ui.pages import (
    ControlPlanePageDeps,
    LogsSummaryPageDeps,
    render_control_plane_page,
    render_logs_summary_page,
)

from control_plane.actuals import fetch_actual_metrics_df
from control_plane.api_client import build_api_response
from control_plane.anomaly_detectors import get_anomaly_detector
from control_plane.config import (
    ANALYZE_TOP_N_ANOMALIES,
    ANOMALY_DETECTOR,
    ANOMALY_IQR_MIN_PERIODS,
    ANOMALY_IQR_SCALE,
    ANOMALY_IQR_WINDOW,
    ANOMALY_PYOD_CONTAMINATION,
    ANOMALY_PYOD_RANDOM_STATE,
    ANOMALY_RUPTURES_MODEL,
    ANOMALY_RUPTURES_PENALTY,
    ANOMALY_ZSCORE,
    ARTIFACTS_DIR,
    CLICKHOUSE_METRICS_HOST,
    CLICKHOUSE_METRICS_PORT,
    CLICKHOUSE_METRICS_QUERY,
    CLICKHOUSE_METRICS_SECURE,
    CLICKHOUSE_PREDICTIONS_HOST,
    CLICKHOUSE_PREDICTIONS_PORT,
    CLICKHOUSE_PREDICTIONS_QUERY,
    CLICKHOUSE_PREDICTIONS_SECURE,
    DATA_LOOKBACK_MINUTES,
    FORECAST_METRIC_NAME,
    FORECAST_SERVICE,
    FORECAST_TYPE,
    LOGS_DIR,
    LOOPBACK_MINUTES,
    LOGS_FETCH_MODE,
    LOGS_TAIL_LIMIT,
    METRICS_SOURCE,
    PLOTS_DIR,
    PROCESS_ALERTS,
    PREDICTION_KIND,
    PREDICTION_LOOKAHEAD_MINUTES,
    PROM_MAX_POINTS,
    PROM_QUERY,
    PROM_SERIES_INDEX,
    PROM_STEP,
    TEST_MODE,
)
from control_plane.logging_config import configure_logging
from control_plane.predictions_db import fetch_predictions_from_db
from control_plane.processing import process_anomalies
from control_plane.test_mode import generate_mock_data
from control_plane.trace import log_dataframe, log_event
from control_plane.utils import to_iso_z
from control_plane.visualization import visualize, visualize_combined
from settings import settings

logger = logging.getLogger(__name__)

LOGS_BATCH_TABLE_HEIGHT = 240
ANOMALIES_TABLE_HEIGHT = 220
SUMMARY_TEXT_HEIGHT = 180
FINAL_TEXT_HEIGHT = 230
LOGS_SQL_TEXTAREA_HEIGHT = 220


def _ensure_runtime() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()


def _query_metrics_df(query: str) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError("Для запросов метрик нужен пакет clickhouse-connect") from exc

    host = str(settings.CONTROL_PLANE_CLICKHOUSE_METRICS_HOST).strip()
    if not host:
        raise ValueError(
            "Укажи CONTROL_PLANE_CLICKHOUSE_METRICS_HOST в .env "
            "для SQL-запросов метрик на странице Logs Summarizer"
        )

    client = clickhouse_connect.get_client(
        host=host,
        port=int(settings.CONTROL_PLANE_CLICKHOUSE_METRICS_PORT),
        username=str(settings.CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME).strip() or None,
        password=str(settings.CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD).strip() or None,
        secure=bool(settings.CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE),
    )
    try:
        return client.query_df(query)
    finally:
        try:
            client.close()
        except Exception:
            logger.warning("ClickHouse client close failed for metrics query")


def _apply_large_text_forms_style() -> None:
    st.markdown(
        """
        <style>
        label[data-testid="stWidgetLabel"] p {
            font-size: 1.02rem !important;
        }
        div[data-baseweb="input"] input {
            font-size: 1.02rem !important;
            min-height: 2.8rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }
        div[data-baseweb="textarea"] textarea {
            font-size: 1.0rem !important;
            line-height: 1.45 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _prepare_anomalies(
    anomalies_df: pd.DataFrame,
    service: str,
) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    if anomalies_df.empty:
        return anomalies
    for _, row in anomalies_df.iterrows():
        if row.get("source") != "actual":
            continue
        anomalies.append(
            {
                "timestamp": to_iso_z(pd.to_datetime(row["timestamp"], utc=True)),
                "service": service,
                "value": float(row["value"]),
                "predicted": float(row.get("predicted")) if row.get("predicted") is not None else None,
                "residual": float(row.get("residual")) if row.get("residual") is not None else None,
                "is_anomaly": True,
                "source": row.get("source"),
            }
        )
    return anomalies


def _payload_for_log(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in payload.items():
        target_key = key
        if key == "event":
            target_key = "payload_event"
        elif key == "level":
            target_key = "payload_level"
        if isinstance(value, pd.DataFrame):
            compact[f"{target_key}_rows"] = len(value)
            continue
        if isinstance(value, dict):
            compact[f"{target_key}_keys"] = list(value.keys())[:10]
            continue
        if isinstance(value, list):
            compact[f"{target_key}_len"] = len(value)
            continue
        compact[target_key] = value
    return compact


def _render_scrollable_text(value: Any, *, height: int) -> None:
    safe_text = html.escape(str(value) if value is not None else "")
    st.markdown(
        (
            f"<div style='max-height:{int(height)}px; overflow-y:auto; padding:0.55rem; "
            "border:1px solid rgba(128,128,128,0.35); border-radius:0.45rem; "
            "background:rgba(249,250,251,0.9);'>"
            "<pre style='margin:0; white-space:pre-wrap; font-family:ui-monospace, SFMono-Regular, "
            "Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; "
            "font-size:0.93rem; line-height:1.42;'>"
            f"{safe_text}</pre></div>"
        ),
        unsafe_allow_html=True,
    )


def _render_pretty_summary_text(value: Any, *, height: int) -> None:
    text = str(value) if value is not None else ""
    lines = text.replace("\r\n", "\n").split("\n")
    html_lines: List[str] = []
    in_list = False

    section_re = re.compile(r"^(?:\d+\s*[\)\.\-:]\s*)?([A-Za-zА-Яа-я][A-Za-zА-Яа-я0-9_ \-]{2,}):?$")
    bullet_re = re.compile(r"^(?:[-*]|•)\s+(.+)$")
    numbered_re = re.compile(r"^\d+\s*[\)\.\-:]\s+(.+)$")

    def _close_list_if_needed() -> None:
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            _close_list_if_needed()
            html_lines.append("<div style='height:0.35rem;'></div>")
            continue

        section_match = section_re.match(line)
        if section_match and len(line) <= 90:
            _close_list_if_needed()
            section_title = html.escape(section_match.group(1).replace("_", " "))
            html_lines.append(
                (
                    "<div style='font-weight:700; color:#111827; margin-top:0.25rem; "
                    "margin-bottom:0.18rem;'>"
                    f"{section_title}"
                    "</div>"
                )
            )
            continue

        bullet_match = bullet_re.match(line) or numbered_re.match(line)
        if bullet_match:
            if not in_list:
                html_lines.append(
                    "<ul style='margin:0.14rem 0 0.38rem 1rem; padding-left:0.52rem;'>"
                )
                in_list = True
            html_lines.append(f"<li style='margin:0.1rem 0;'>{html.escape(bullet_match.group(1))}</li>")
            continue

        _close_list_if_needed()
        html_lines.append(f"<div style='margin:0.1rem 0;'>{html.escape(line)}</div>")

    _close_list_if_needed()

    st.markdown(
        (
            f"<div style='max-height:{int(height)}px; overflow-y:auto; padding:0.7rem 0.75rem; "
            "border:1px solid rgba(128,128,128,0.35); border-radius:0.55rem; "
            "background:rgba(249,250,251,0.95); font-size:0.98rem; line-height:1.56;'>"
            f"{''.join(html_lines)}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def run_single_iteration(
    *,
    test_mode: bool,
    query: str,
    detector_name: str,
    data_lookback_minutes: int,
    prediction_lookahead_minutes: int,
    analyze_top_n: int,
    process_lookback_minutes: int,
    process_alerts: bool,
    on_stage: Optional[Callable[[str, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if not logging.getLogger().handlers:
        _ensure_runtime()
        log_event(logger, "run_single_iteration.logging_autoconfigured")

    def _emit(stage: str, progress: int, payload: Dict[str, Any]) -> None:
        log_event(
            logger,
            "run_single_iteration.emit",
            stage=stage,
            progress=progress,
            **_payload_for_log(payload),
        )
        if on_stage is not None:
            on_stage(stage, progress, payload)

    def _emit_stage_error(stage_name: str, progress: int, exc: Exception) -> None:
        error_text = f"{exc.__class__.__name__}: {exc}"
        logger.exception("run_single_iteration.%s.failed", stage_name)
        _emit(
            "stage_error",
            progress,
            {
                "stage_name": stage_name,
                "message": f"Ошибка на этапе {stage_name}",
                "error": error_text,
            },
        )

    log_event(
        logger,
        "run_single_iteration.start",
        test_mode=test_mode,
        detector=detector_name,
        lookback_minutes=data_lookback_minutes,
        prediction_lookahead_minutes=prediction_lookahead_minutes,
        analyze_top_n=analyze_top_n,
        process_lookback_minutes=process_lookback_minutes,
        process_alerts=process_alerts,
    )
    _emit("init", 2, {"message": "Инициализация запуска"})
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=data_lookback_minutes)
    if not test_mode and METRICS_SOURCE == "prometheus":
        # Requirement: for Prometheus mode always pull full month of points.
        start_time = end_time - timedelta(days=30)
        log_event(
            logger,
            "run_single_iteration.prometheus_month_window_applied",
            start=start_time.isoformat(),
            end=end_time.isoformat(),
        )

    _emit(
        "fetch_start",
        10,
        {
            "message": "Загрузка данных",
            "window_start": start_time,
            "window_end": end_time,
            "test_mode": test_mode,
        },
    )
    try:
        if test_mode:
            logger.info("TEST_MODE=true: внешние запросы (Prometheus/DB) пропущены")
            actual_df, predictions_df = generate_mock_data(
                start_time=start_time,
                end_time=end_time,
                step=PROM_STEP,
                lookahead_minutes=prediction_lookahead_minutes,
            )
        else:
            actual_df = fetch_actual_metrics_df(
                query=query,
                start_time=start_time,
                end_time=end_time,
                step=PROM_STEP,
                max_points=PROM_MAX_POINTS,
                series_index=PROM_SERIES_INDEX,
            )
            predictions_df = fetch_predictions_from_db(
                service=FORECAST_SERVICE,
                metric_name=FORECAST_METRIC_NAME,
                start_time=start_time,
                end_time=end_time + timedelta(minutes=prediction_lookahead_minutes),
                forecast_type=FORECAST_TYPE,
                prediction_kind=PREDICTION_KIND,
            )
    except Exception as exc:
        _emit_stage_error("fetch", 10, exc)
        raise
    log_dataframe(logger, "run_single_iteration.actual_df", actual_df)
    log_dataframe(logger, "run_single_iteration.predictions_df", predictions_df)
    _emit(
        "fetch_done",
        35,
        {
            "message": "Данные получены",
            "actual_df": actual_df,
            "predictions_df": predictions_df,
        },
    )

    _emit("detect_start", 45, {"message": "Поиск аномалий"})
    try:
        detector = get_anomaly_detector(
            detector_name,
            iqr_window=ANOMALY_IQR_WINDOW,
            iqr_scale=ANOMALY_IQR_SCALE,
            min_periods=ANOMALY_IQR_MIN_PERIODS,
            zscore_threshold=ANOMALY_ZSCORE,
            pyod_contamination=ANOMALY_PYOD_CONTAMINATION,
            pyod_random_state=ANOMALY_PYOD_RANDOM_STATE,
            ruptures_penalty=ANOMALY_RUPTURES_PENALTY,
            ruptures_model=ANOMALY_RUPTURES_MODEL,
        )
        detection_result = detector.detect(actual_df, predictions_df, step=PROM_STEP)
        merged_df = detection_result.merged_df
        anomalies_df = detection_result.anomalies_df
        if not anomalies_df.empty and "source" not in anomalies_df.columns:
            anomalies_df["source"] = "actual"
    except Exception as exc:
        _emit_stage_error("detect", 45, exc)
        raise
    log_dataframe(logger, "run_single_iteration.merged_df", merged_df)
    log_dataframe(logger, "run_single_iteration.anomalies_df", anomalies_df)
    _emit(
        "detect_done",
        60,
        {
            "message": "Аномалии найдены",
            "merged_df": merged_df,
            "anomalies_df": anomalies_df,
            "detector": getattr(detector, "name", detector.__class__.__name__),
        },
    )

    anomalies = _prepare_anomalies(anomalies_df, FORECAST_SERVICE)
    api_response = build_api_response(merged_df, predictions_df, anomalies_df)

    log_filename = LOGS_DIR / f"api_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(api_response, f, ensure_ascii=False, indent=2, default=str)
    log_event(
        logger,
        "run_single_iteration.api_response_saved",
        path=str(log_filename),
        merged=len(api_response.get("data", {}).get("merged_data", [])),
        predictions=len(api_response.get("data", {}).get("predictions", [])),
        anomalies=len(api_response.get("data", {}).get("anomalies", [])),
    )

    _emit("viz_start", 70, {"message": "Построение графиков"})
    try:
        plot_info = visualize(query, api_response)
    except Exception as exc:
        _emit_stage_error("visualization", 70, exc)
        raise
    _emit("viz_done", 82, {"message": "Графики готовы", "plot_info": plot_info})

    processing_results: List[Dict[str, Any]] = []
    if process_alerts and anomalies:
        _emit(
            "process_start",
            88,
            {
                "message": "Обработка аномалий",
                "anomalies_total": len(anomalies),
            },
        )
        sorted_anomalies = sorted(
            anomalies,
            key=lambda x: datetime.fromisoformat(x["timestamp"].replace("Z", "+00:00")),
            reverse=True,
        )
        recent_anomalies = sorted_anomalies[:analyze_top_n]
        _emit(
            "process_selected",
            89,
            {
                "message": "Выбраны аномалии для суммаризации",
                "recent_anomalies": recent_anomalies,
                "selected_total": len(recent_anomalies),
            },
        )

        def _on_processing_event(event: str, payload: Dict[str, Any]) -> None:
            idx = payload.get("index")
            selected_total = max(len(recent_anomalies), 1)
            if event == "summary_start":
                progress = 90
                message = f"Суммаризация аномалии {int(idx) + 1}/{selected_total}"
            elif event == "summary_done":
                progress = 92
                message = f"Summary готов для аномалии {int(idx) + 1}/{selected_total}"
            elif event == "alert_start":
                progress = 93
                message = f"Отправка alert для аномалии {int(idx) + 1}/{selected_total}"
            elif event == "alert_done":
                progress = 94
                message = f"Alert отправлен для аномалии {int(idx) + 1}/{selected_total}"
            elif event == "anomaly_done":
                progress = 95
                message = f"Аномалия {int(idx) + 1}/{selected_total} обработана"
            elif event == "anomaly_error":
                progress = 95
                message = f"Ошибка на аномалии {int(idx) + 1}/{selected_total}"
            elif event == "process_done":
                progress = 96
                message = "Суммаризация завершена"
            else:
                progress = 89
                message = f"Событие: {event}"

            _emit(
                "process_live",
                progress,
                {
                    "message": message,
                    "event": event,
                    "selected_total": len(recent_anomalies),
                    **payload,
                },
            )

        try:
            processing_results = process_anomalies(
                anomalies=recent_anomalies,
                lookback_minutes=process_lookback_minutes,
                continue_on_error=True,
                test_mode=test_mode,
                on_event=_on_processing_event,
            )
        except Exception as exc:
            _emit_stage_error("process", 88, exc)
            raise
        log_event(
            logger,
            "run_single_iteration.process_anomalies.done",
            processed=len(processing_results),
            success=sum(1 for item in processing_results if item.get("success")),
            errors=sum(1 for item in processing_results if item.get("error")),
        )
        _emit(
            "process_done",
            96,
            {
                "message": "Обработка завершена",
                "processing_results": processing_results,
            },
        )
    else:
        if not process_alerts:
            _emit(
                "process_skip",
                96,
                {
                    "message": "Обработка отключена настройкой",
                    "reason": "alerts_disabled",
                },
            )
        else:
            _emit(
                "process_skip",
                96,
                {
                    "message": "Аномалии не найдены, обработка не требуется",
                    "reason": "no_anomalies",
                },
            )

    result = {
        "window_start": start_time,
        "window_end": end_time,
        "actual_df": actual_df,
        "predictions_df": predictions_df,
        "merged_df": merged_df,
        "anomalies_df": anomalies_df,
        "anomalies": anomalies,
        "api_response": api_response,
        "plot_info": plot_info,
        "processing_results": processing_results,
        "api_log_path": str(log_filename),
        "detector": getattr(detector, "name", detector.__class__.__name__),
    }
    log_event(
        logger,
        "run_single_iteration.done",
        anomalies=len(anomalies),
        processing_results=len(processing_results),
        detector=result["detector"],
    )
    _emit("done", 100, {"message": "Итерация завершена", "result": result})
    return result


def _read_log_tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.readlines()
    return "".join(content[-lines:])


def _format_table_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    timestamp_like = {
        "timestamp",
        "ts",
        "time",
        "datetime",
        "period_start",
        "period_end",
        "start",
        "end",
    }
    for col in out.columns:
        col_name = str(col).strip().lower()
        if col_name not in timestamp_like and "timestamp" not in col_name:
            continue
        parsed = pd.to_datetime(out[col], utc=True, errors="coerce")
        mask = parsed.notna()
        if not bool(mask.any()):
            continue
        out.loc[mask, col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC").loc[mask]
    return out


def _only_future_predictions(
    predictions_df: Optional[pd.DataFrame],
    actual_end_ts: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty or actual_end_ts is None:
        return pd.DataFrame(columns=["timestamp", "predicted"])
    out = predictions_df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    return out[out["timestamp"] > actual_end_ts].copy()


def _infer_batch_period(batch: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    explicit_start = batch.get("batch_period_start")
    explicit_end = batch.get("batch_period_end")
    if explicit_start or explicit_end:
        return (
            str(explicit_start) if explicit_start else None,
            str(explicit_end) if explicit_end else None,
        )

    logs = batch.get("batch_logs", [])
    if not isinstance(logs, list) or not logs:
        return None, None

    timestamps: List[pd.Timestamp] = []
    for row in logs:
        if not isinstance(row, dict):
            continue
        raw_ts = None
        for key in ("timestamp", "ts", "time", "datetime"):
            if row.get(key) is not None:
                raw_ts = row.get(key)
                break
        if raw_ts is None:
            continue
        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        timestamps.append(ts)

    if not timestamps:
        return None, None
    return to_iso_z(min(timestamps)), to_iso_z(max(timestamps))


def _build_anomaly_window_figure(
    merged_df: pd.DataFrame,
    predictions_df: Optional[pd.DataFrame],
    anomaly: Dict[str, Any],
    window_minutes: int,
    actual_end_ts: Optional[pd.Timestamp],
) -> Optional[plt.Figure]:
    if merged_df is None or merged_df.empty:
        return None
    ts = pd.to_datetime(anomaly.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    start = ts - pd.Timedelta(minutes=window_minutes)
    end = ts + pd.Timedelta(minutes=window_minutes)
    window = merged_df[(merged_df["timestamp"] >= start) & (merged_df["timestamp"] <= end)]
    if window.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(window["timestamp"], window["value"], label="actual", color="#2563eb", linewidth=2)
    future_predictions = _only_future_predictions(predictions_df, actual_end_ts)
    if not future_predictions.empty:
        pred_window = future_predictions[
            (future_predictions["timestamp"] >= start)
            & (future_predictions["timestamp"] <= end)
        ]
        if not pred_window.empty:
            ax.plot(
                pred_window["timestamp"],
                pred_window["predicted"],
                label="predicted (future)",
                color="#ef4444",
                linewidth=1.8,
                linestyle="--",
            )
    anomaly_value = anomaly.get("value")
    if anomaly_value is not None:
        ax.scatter([ts], [float(anomaly_value)], color="#16a34a", marker="x", s=120, label="anomaly")
    ax.axvline(ts, color="#6b7280", linestyle=":", linewidth=1.2)
    ax.set_title(f"Аномалия @ {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _build_predictions_focus_figure(
    actual_df: Optional[pd.DataFrame],
    predictions_df: Optional[pd.DataFrame],
) -> Optional[plt.Figure]:
    if actual_df is None or predictions_df is None or actual_df.empty or predictions_df.empty:
        return None
    if "timestamp" not in actual_df.columns or "value" not in actual_df.columns:
        return None
    if "timestamp" not in predictions_df.columns:
        return None

    pred_col = "predicted" if "predicted" in predictions_df.columns else "value"
    if pred_col not in predictions_df.columns:
        return None

    actual = actual_df.copy()
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True, errors="coerce")
    actual["value"] = pd.to_numeric(actual["value"], errors="coerce")
    actual = actual.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    if actual.empty:
        return None

    preds = predictions_df.copy()
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True, errors="coerce")
    preds[pred_col] = pd.to_numeric(preds[pred_col], errors="coerce")
    preds = preds.dropna(subset=["timestamp", pred_col]).sort_values("timestamp")
    if preds.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.plot(
        actual["timestamp"],
        actual["value"],
        color="#2563eb",
        linewidth=2,
        label="actual",
    )
    ax.plot(
        preds["timestamp"],
        preds[pred_col],
        color="#ef4444",
        linewidth=2,
        linestyle="--",
        label="predicted",
    )

    actual_end_ts = actual["timestamp"].max()
    ax.axvline(
        actual_end_ts,
        color="#6b7280",
        linestyle=":",
        linewidth=1.2,
        label="history/prediction boundary",
    )

    ax.set_title("Фокус-график: история + все предикты (единое полотно)")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best")

    fig.tight_layout()
    return fig


def _render_anomaly_cards(
    container,
    analysis_state: Dict[str, Any],
    merged_df: Optional[pd.DataFrame],
    predictions_df: Optional[pd.DataFrame],
    actual_end_ts: Optional[pd.Timestamp],
    window_minutes: int,
) -> None:
    status_titles = {
        "queued": "В очереди",
        "summarizing": "Подготовка summary",
        "map": "Map этап",
        "reduce": "Reduce этап",
        "summary_ready": "Summary готов",
        "sending_alert": "Готовим уведомление",
        "done": "Завершено",
        "error": "Ошибка",
    }

    with container.container():
        st.subheader("3. Пошаговый анализ Top-N аномалий")
        rows = list(analysis_state.get("by_idx", {}).values())
        if not rows:
            empty_message = analysis_state.get("empty_message")
            if empty_message:
                st.info(str(empty_message))
            return
        rows = sorted(rows, key=lambda x: x.get("anomaly_idx", 0))
        for row in rows:
            anomaly = row.get("anomaly", {})
            title_ts = anomaly.get("timestamp", "-")
            st.markdown("---")
            with st.chat_message("user"):
                st.markdown(f"**Аномалия #{row.get('anomaly_idx')}**")
                st.markdown(
                    "\n".join(
                        [
                            f"Время: `{anomaly.get('timestamp', title_ts)}`",
                            f"Значение: `{anomaly.get('value', '-')}`",
                            f"Источник: `{anomaly.get('source', 'actual')}`",
                        ]
                    )
                )

                fig = _build_anomaly_window_figure(
                    merged_df=merged_df,
                    predictions_df=predictions_df,
                    anomaly=anomaly,
                    window_minutes=window_minutes,
                    actual_end_ts=actual_end_ts,
                )
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            with st.chat_message("assistant"):
                status = row.get("status", "queued")
                st.markdown(f"Статус: **{status_titles.get(status, status)}**")
                logs_processed = row.get("logs_processed")
                logs_total = row.get("logs_total")
                processed_num = pd.to_numeric(logs_processed, errors="coerce")
                total_num = pd.to_numeric(logs_total, errors="coerce")
                if not pd.isna(processed_num):
                    if not pd.isna(total_num) and float(total_num) > 0:
                        ratio = min(max(float(processed_num) / float(total_num), 0.0), 1.0)
                        st.progress(
                            ratio,
                            text=(
                                f"Прогресс суммаризации логов: "
                                f"{int(processed_num)}/{int(total_num)}"
                            ),
                        )
                    else:
                        st.caption(
                            f"Прогресс суммаризации логов: обработано {int(processed_num)} строк"
                        )

            batches = row.get("map_batches", [])
            if batches:
                for batch in batches:
                    idx = int(batch.get("batch_index", 0)) + 1
                    total = batch.get("batch_total", len(batches))
                    batch_logs = batch.get("batch_logs", [])
                    if not isinstance(batch_logs, list):
                        batch_logs = []
                    batch_logs_count = batch.get("batch_logs_count")
                    if batch_logs_count is None:
                        batch_logs_count = len(batch_logs)
                    batch_period_start, batch_period_end = _infer_batch_period(batch)
                    with st.chat_message("assistant"):
                        st.markdown(f"Map summary {idx}/{total}")
                        _render_pretty_summary_text(
                            batch.get("batch_summary", ""),
                            height=SUMMARY_TEXT_HEIGHT,
                        )
                        st.caption(f"Логов в батче: {batch_logs_count}")
                        if batch_period_start and batch_period_end:
                            st.caption(
                                f"Период логов батча: `{batch_period_start}` -> `{batch_period_end}`"
                            )
                        elif batch_period_start:
                            st.caption(f"Период логов батча: с `{batch_period_start}`")
                        elif batch_period_end:
                            st.caption(f"Период логов батча: до `{batch_period_end}`")
                        if batch_logs:
                            logs_df = pd.DataFrame(batch_logs)
                            st.dataframe(
                                _format_table_timestamps(logs_df),
                                use_container_width=True,
                                hide_index=True,
                                height=LOGS_BATCH_TABLE_HEIGHT,
                            )
                        else:
                            st.caption("Логи батча не переданы.")

            if row.get("final_summary"):
                with st.chat_message("assistant"):
                    st.markdown("Итоговый Reduce summary")
                    _render_pretty_summary_text(
                        row["final_summary"],
                        height=FINAL_TEXT_HEIGHT,
                    )

            if row.get("notification_text"):
                with st.chat_message("assistant"):
                    st.markdown("Уведомление для SRE")
                    _render_scrollable_text(
                        row["notification_text"],
                        height=SUMMARY_TEXT_HEIGHT,
                    )

            if row.get("error"):
                with st.chat_message("assistant"):
                    st.error(f"Ошибка обработки: {row['error']}")


def _ui_runtime_config() -> Dict[str, Any]:
    ch_query = (
        CLICKHOUSE_METRICS_QUERY
        if len(CLICKHOUSE_METRICS_QUERY) <= 140
        else f"{CLICKHOUSE_METRICS_QUERY[:140]}..."
    )
    ch_predictions_query = (
        CLICKHOUSE_PREDICTIONS_QUERY
        if len(CLICKHOUSE_PREDICTIONS_QUERY) <= 140
        else f"{CLICKHOUSE_PREDICTIONS_QUERY[:140]}..."
    )
    return {
        "CONTROL_PLANE_TEST_MODE": TEST_MODE,
        "CONTROL_PLANE_METRICS_SOURCE": METRICS_SOURCE,
        "CONTROL_PLANE_PROM_METRICS_QUERY": PROM_QUERY,
        "CONTROL_PLANE_PROM_METRICS_STEP": PROM_STEP,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY": ch_query,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_HOST": CLICKHOUSE_METRICS_HOST,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_PORT": CLICKHOUSE_METRICS_PORT,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE": CLICKHOUSE_METRICS_SECURE,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST": CLICKHOUSE_PREDICTIONS_HOST,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PORT": CLICKHOUSE_PREDICTIONS_PORT,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_SECURE": CLICKHOUSE_PREDICTIONS_SECURE,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY": ch_predictions_query,
        "CONTROL_PLANE_ANOMALY_DETECTOR": ANOMALY_DETECTOR,
        "CONTROL_PLANE_ANOMALY_PYOD_CONTAMINATION": ANOMALY_PYOD_CONTAMINATION,
        "CONTROL_PLANE_ANOMALY_PYOD_RANDOM_STATE": ANOMALY_PYOD_RANDOM_STATE,
        "CONTROL_PLANE_ANOMALY_RUPTURES_PENALTY": ANOMALY_RUPTURES_PENALTY,
        "CONTROL_PLANE_ANOMALY_RUPTURES_MODEL": ANOMALY_RUPTURES_MODEL,
        "CONTROL_PLANE_DATA_LOOKBACK_MINUTES": DATA_LOOKBACK_MINUTES,
        "CONTROL_PLANE_PREDICTION_LOOKAHEAD_MINUTES": PREDICTION_LOOKAHEAD_MINUTES,
        "CONTROL_PLANE_ANALYZE_TOP_N_ANOMALIES": ANALYZE_TOP_N_ANOMALIES,
        "CONTROL_PLANE_LOOPBACK_MINUTES": LOOPBACK_MINUTES,
        "CONTROL_PLANE_LOGS_FETCH_MODE": LOGS_FETCH_MODE,
        "CONTROL_PLANE_LOGS_TAIL_LIMIT": LOGS_TAIL_LIMIT,
        "CONTROL_PLANE_PROCESS_ALERTS": PROCESS_ALERTS,
        "CONTROL_PLANE_PROM_METRICS_MAX_POINTS": PROM_MAX_POINTS,
        "CONTROL_PLANE_PROM_METRICS_SERIES_INDEX": PROM_SERIES_INDEX,
        "CONTROL_PLANE_FORECAST_SERVICE": FORECAST_SERVICE,
        "CONTROL_PLANE_FORECAST_METRIC_NAME": FORECAST_METRIC_NAME,
        "CONTROL_PLANE_FORECAST_TYPE": FORECAST_TYPE,
        "CONTROL_PLANE_PREDICTION_KIND": PREDICTION_KIND,
    }


def _render_logs_summary_page() -> None:
    deps = LogsSummaryPageDeps(
        logger=logger,
        db_batch_size=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_DB_BATCH_SIZE", settings.CONTROL_PLANE_UI_LOGS_SUMMARY_BATCH_SIZE)),
            1,
        ),
        llm_batch_size=max(
            int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_BATCH_SIZE", settings.CONTROL_PLANE_UI_LOGS_SUMMARY_BATCH_SIZE)),
            1,
        ),
        map_workers=max(int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAP_WORKERS", 1)), 1),
        max_retries=int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_RETRIES", -1)),
        llm_timeout=max(int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_LLM_TIMEOUT", 600)), 10),
        test_mode=TEST_MODE,
        loopback_minutes=LOOPBACK_MINUTES,
        logs_tail_limit=LOGS_TAIL_LIMIT,
        period_log_summarizer_cls=PeriodLogSummarizer,
        summarizer_config_cls=SummarizerConfig,
        make_llm_call=_make_llm_call,
        query_logs_df=_query_logs_df,
        query_metrics_df=_query_metrics_df,
        render_scrollable_text=_render_scrollable_text,
        render_pretty_summary_text=_render_pretty_summary_text,
        infer_batch_period=_infer_batch_period,
        summary_text_height=SUMMARY_TEXT_HEIGHT,
        final_text_height=FINAL_TEXT_HEIGHT,
        logs_batch_table_height=LOGS_BATCH_TABLE_HEIGHT,
        sql_textarea_height=LOGS_SQL_TEXTAREA_HEIGHT,
        default_sql_query=(
            str(settings.CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_SQL).strip()
            or str(settings.CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY).strip()
        ),
        default_metrics_query=str(
            getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_DEFAULT_METRICS_SQL", "")
        ).strip(),
        output_dir=LOGS_DIR,
        logs_fetch_mode=LOGS_FETCH_MODE,
    )
    render_logs_summary_page(deps)


def _render_control_plane_page() -> None:
    deps = ControlPlanePageDeps(
        logger=logger,
        log_event=log_event,
        payload_for_log=_payload_for_log,
        run_single_iteration=run_single_iteration,
        visualize_combined=visualize_combined,
        build_predictions_focus_figure=_build_predictions_focus_figure,
        only_future_predictions=_only_future_predictions,
        render_anomaly_cards=_render_anomaly_cards,
        plots_dir=PLOTS_DIR,
        anomalies_table_height=ANOMALIES_TABLE_HEIGHT,
        loopback_minutes=LOOPBACK_MINUTES,
        test_mode=TEST_MODE,
        prom_query=PROM_QUERY,
        anomaly_detector=ANOMALY_DETECTOR,
        data_lookback_minutes=DATA_LOOKBACK_MINUTES,
        prediction_lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES,
        analyze_top_n=ANALYZE_TOP_N_ANOMALIES,
        process_alerts=PROCESS_ALERTS,
        logs_fetch_mode=LOGS_FETCH_MODE,
        logs_tail_limit=LOGS_TAIL_LIMIT,
    )
    render_control_plane_page(deps)


def main() -> None:
    st.set_page_config(page_title="Control Plane", layout="wide")
    _ensure_runtime()
    _apply_large_text_forms_style()

    with st.sidebar:
        page = st.radio(
            "Страницы",
            ["Control Plane", "Logs Summarizer"],
            index=0,
            key="cp_page_selector",
        )

    if page == "Logs Summarizer":
        _render_logs_summary_page()
        return

    _render_control_plane_page()


if __name__ == "__main__":
    main()
