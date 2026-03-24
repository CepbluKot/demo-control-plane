import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from control_plane.api_client import build_api_response
from control_plane.anomaly_detectors import get_anomaly_detector
from control_plane.config import (
    ANALYZE_TOP_N_ANOMALIES,
    ANOMALY_DETECTOR,
    ANOMALY_IQR_MIN_PERIODS,
    ANOMALY_IQR_SCALE,
    ANOMALY_IQR_WINDOW,
    ANOMALY_ZSCORE,
    ARTIFACTS_DIR,
    DATA_LOOKBACK_MINUTES,
    FORECAST_METRIC_NAME,
    FORECAST_SERVICE,
    FORECAST_TYPE,
    LOGS_DIR,
    LOOPBACK_MINUTES,
    PLOTS_DIR,
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
from control_plane.prometheus_io import fetch_prometheus_df
from control_plane.test_mode import generate_mock_data
from control_plane.utils import to_iso_z
from control_plane.visualization import visualize

logger = logging.getLogger(__name__)


def _ensure_runtime() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()


def _prepare_anomalies(anomalies_df: pd.DataFrame) -> List[Dict[str, Any]]:
    anomalies: List[Dict[str, Any]] = []
    if anomalies_df.empty:
        return anomalies
    for _, row in anomalies_df.iterrows():
        if row.get("source") != "actual":
            continue
        anomalies.append(
            {
                "timestamp": to_iso_z(pd.to_datetime(row["timestamp"], utc=True)),
                "value": float(row["value"]),
                "predicted": float(row.get("predicted")) if row.get("predicted") is not None else None,
                "residual": float(row.get("residual")) if row.get("residual") is not None else None,
                "is_anomaly": True,
                "source": row.get("source"),
            }
        )
    return anomalies


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
    def _emit(stage: str, progress: int, payload: Dict[str, Any]) -> None:
        if on_stage is not None:
            on_stage(stage, progress, payload)

    _emit("init", 2, {"message": "Инициализация запуска"})
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=data_lookback_minutes)

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
    if test_mode:
        actual_df, predictions_df = generate_mock_data(
            start_time=start_time,
            end_time=end_time,
            step=PROM_STEP,
            lookahead_minutes=prediction_lookahead_minutes,
        )
    else:
        actual_df = fetch_prometheus_df(
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
    detector = get_anomaly_detector(
        detector_name,
        iqr_window=ANOMALY_IQR_WINDOW,
        iqr_scale=ANOMALY_IQR_SCALE,
        min_periods=ANOMALY_IQR_MIN_PERIODS,
        zscore_threshold=ANOMALY_ZSCORE,
    )
    detection_result = detector.detect(actual_df, predictions_df, step=PROM_STEP)
    merged_df = detection_result.merged_df
    anomalies_df = detection_result.anomalies_df
    if not anomalies_df.empty and "source" not in anomalies_df.columns:
        anomalies_df["source"] = "actual"
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

    anomalies = _prepare_anomalies(anomalies_df)
    api_response = build_api_response(merged_df, predictions_df, anomalies_df)

    log_filename = LOGS_DIR / f"api_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(api_response, f, ensure_ascii=False, indent=2, default=str)

    _emit("viz_start", 70, {"message": "Построение графиков"})
    plot_info = visualize(query, api_response)
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

        processing_results = process_anomalies(
            anomalies=recent_anomalies,
            lookback_minutes=process_lookback_minutes,
            continue_on_error=True,
            test_mode=test_mode,
            on_event=_on_processing_event,
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
        _emit("process_skip", 96, {"message": "Обработка пропущена"})

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
    _emit("done", 100, {"message": "Итерация завершена", "result": result})
    return result


def _read_log_tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.readlines()
    return "".join(content[-lines:])


def main() -> None:
    st.set_page_config(page_title="Control Plane", layout="wide")
    st.title("Control Plane UI")
    st.caption("Запуск одного цикла пайплайна, просмотр аномалий, графиков и логов")

    _ensure_runtime()

    with st.sidebar:
        st.header("Параметры")
        test_mode = st.checkbox("TEST MODE", value=TEST_MODE)
        query = st.text_area("PromQL query", value=PROM_QUERY, height=120)
        detector_options = ["rolling_iqr", "residual_zscore"]
        default_detector_idx = detector_options.index(ANOMALY_DETECTOR) if ANOMALY_DETECTOR in detector_options else 0
        detector_name = st.selectbox("Детектор", options=detector_options, index=default_detector_idx)
        data_lookback_minutes = st.slider(
            "Окно фактических данных (мин)",
            min_value=10,
            max_value=240,
            value=DATA_LOOKBACK_MINUTES,
            step=5,
        )
        prediction_lookahead_minutes = st.slider(
            "Горизонт прогноза (мин)",
            min_value=10,
            max_value=240,
            value=PREDICTION_LOOKAHEAD_MINUTES,
            step=5,
        )
        analyze_top_n = st.slider(
            "Сколько аномалий анализировать",
            min_value=1,
            max_value=10,
            value=ANALYZE_TOP_N_ANOMALIES,
            step=1,
        )
        process_lookback_minutes = st.slider(
            "Lookback для логов (мин)",
            min_value=5,
            max_value=180,
            value=LOOPBACK_MINUTES,
            step=5,
        )
        process_alerts = st.checkbox("Запускать summary/alert обработку", value=True)
        run_clicked = st.button("Запустить 1 итерацию", type="primary", use_container_width=True)

    if not run_clicked:
        st.info('Нажми "Запустить 1 итерацию", чтобы выполнить пайплайн.')
        return

    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    live_placeholder = st.empty()
    summary_placeholder = st.empty()
    summary_table_placeholder = st.empty()
    summary_preview_placeholder = st.empty()
    summary_state: Dict[str, Any] = {
        "by_idx": {},
        "last_preview": "",
    }

    def _on_stage(stage: str, progress: int, payload: Dict[str, Any]) -> None:
        message = payload.get("message", stage)
        progress_placeholder.info(f"[{stage}] {message}")
        progress_bar.progress(progress)
        actual_df = payload.get("actual_df")
        predictions_df = payload.get("predictions_df")
        anomalies_df = payload.get("anomalies_df")
        with live_placeholder.container():
            cols = st.columns(3)
            if isinstance(actual_df, pd.DataFrame):
                cols[0].metric("Actual rows (live)", len(actual_df))
            if isinstance(predictions_df, pd.DataFrame):
                cols[1].metric("Prediction rows (live)", len(predictions_df))
            if isinstance(anomalies_df, pd.DataFrame):
                cols[2].metric("Anomalies (live)", len(anomalies_df))
            if isinstance(actual_df, pd.DataFrame) and not actual_df.empty:
                st.caption("Последние actual точки (live)")
                st.dataframe(actual_df.tail(10), use_container_width=True)
            if isinstance(anomalies_df, pd.DataFrame) and not anomalies_df.empty:
                st.caption("Аномалии (live)")
                st.dataframe(anomalies_df.tail(10), use_container_width=True)

        if stage == "process_selected":
            selected_total = payload.get("selected_total", 0)
            summary_placeholder.info(f"Суммаризация: выбрано аномалий {selected_total}")

        if stage == "process_live":
            event = payload.get("event")
            idx = payload.get("index")
            timestamp = payload.get("timestamp")
            if idx is not None:
                key = str(idx)
                row = summary_state["by_idx"].setdefault(
                    key,
                    {
                        "anomaly_idx": int(idx) + 1,
                        "timestamp": timestamp,
                        "status": "queued",
                        "elapsed_sec": None,
                        "summary_len": None,
                        "error": None,
                    },
                )
                if timestamp:
                    row["timestamp"] = timestamp
                if event == "summary_start":
                    row["status"] = "summarizing"
                elif event == "summary_done":
                    row["status"] = "summary_ready"
                    row["elapsed_sec"] = payload.get("elapsed_sec")
                    row["summary_len"] = payload.get("summary_len")
                    preview = payload.get("summary_preview")
                    if preview:
                        summary_state["last_preview"] = str(preview)
                elif event == "alert_start":
                    row["status"] = "sending_alert"
                elif event == "alert_done":
                    row["status"] = "done"
                elif event == "anomaly_done":
                    row["status"] = "done"
                elif event == "anomaly_error":
                    row["status"] = "error"
                    row["error"] = payload.get("error")

            summary_placeholder.info(f"Суммаризация live: {message}")
            rows = list(summary_state["by_idx"].values())
            if rows:
                summary_df = pd.DataFrame(rows).sort_values("anomaly_idx")
                summary_table_placeholder.dataframe(summary_df, use_container_width=True)
            if summary_state["last_preview"]:
                with summary_preview_placeholder.container():
                    st.caption("Последний preview summary (live)")
                    st.code(summary_state["last_preview"])

    try:
        result = run_single_iteration(
            test_mode=test_mode,
            query=query,
            detector_name=detector_name,
            data_lookback_minutes=data_lookback_minutes,
            prediction_lookahead_minutes=prediction_lookahead_minutes,
            analyze_top_n=analyze_top_n,
            process_lookback_minutes=process_lookback_minutes,
            process_alerts=process_alerts,
            on_stage=_on_stage,
        )
    except Exception as exc:
        logger.exception("Streamlit iteration failed")
        st.error(f"Ошибка выполнения: {exc}")
        st.stop()

    st.success("Итерация завершена")
    st.write(
        f"Окно: {result['window_start'].isoformat()} .. {result['window_end'].isoformat()} | "
        f"детектор: {result['detector']}"
    )

    metrics_col_1, metrics_col_2, metrics_col_3, metrics_col_4 = st.columns(4)
    metrics_col_1.metric("Actual rows", len(result["actual_df"]))
    metrics_col_2.metric("Prediction rows", len(result["predictions_df"]))
    metrics_col_3.metric("Merged rows", len(result["merged_df"]))
    metrics_col_4.metric("Anomalies", len(result["anomalies"]))

    st.subheader("Графики")
    if result["plot_info"]:
        for plot in result["plot_info"]:
            plot_path = Path(plot["path"])
            if plot_path.exists():
                st.image(str(plot_path), caption=str(plot_path))
    else:
        st.warning("Графики не сгенерированы")

    st.subheader("Данные")
    tab1, tab2, tab3 = st.tabs(["Actual", "Predictions", "Anomalies"])
    with tab1:
        st.dataframe(result["actual_df"], use_container_width=True)
    with tab2:
        st.dataframe(result["predictions_df"], use_container_width=True)
    with tab3:
        st.dataframe(result["anomalies_df"], use_container_width=True)

    st.subheader("Обработка аномалий")
    if result["processing_results"]:
        st.json(result["processing_results"])
    else:
        st.caption("Обработка не выполнялась или аномалии отсутствуют")

    st.subheader("Артефакты")
    st.code(f"API response: {result['api_log_path']}")

    st.subheader("Лог (tail)")
    log_path = LOGS_DIR / "control-plane.log"
    log_tail = _read_log_tail(log_path, lines=150)
    if log_tail:
        st.text_area("Последние строки", value=log_tail, height=360)
    else:
        st.caption("Лог еще пуст")


if __name__ == "__main__":
    main()
