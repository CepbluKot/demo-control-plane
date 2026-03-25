import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from control_plane.actuals import fetch_actual_metrics_df
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
    CLICKHOUSE_METRICS_HOST,
    CLICKHOUSE_METRICS_PASSWORD,
    CLICKHOUSE_METRICS_PORT,
    CLICKHOUSE_METRICS_QUERY,
    CLICKHOUSE_METRICS_SECURE,
    CLICKHOUSE_METRICS_USERNAME,
    CLICKHOUSE_PREDICTIONS_HOST,
    CLICKHOUSE_PREDICTIONS_PASSWORD,
    CLICKHOUSE_PREDICTIONS_PORT,
    CLICKHOUSE_PREDICTIONS_QUERY,
    CLICKHOUSE_PREDICTIONS_SECURE,
    CLICKHOUSE_PREDICTIONS_USERNAME,
    DATA_LOOKBACK_MINUTES,
    FORECAST_METRIC_NAME,
    FORECAST_SERVICE,
    FORECAST_TYPE,
    LOGS_DIR,
    LOOPBACK_MINUTES,
    METRICS_SOURCE,
    PLOTS_DIR,
    PROCESS_ALERTS,
    PREDICTION_KIND,
    PREDICTION_LOOKAHEAD_MINUTES,
    PROM_MAX_POINTS,
    PROM_PASSWORD,
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

logger = logging.getLogger(__name__)


def _ensure_runtime() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()


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

            batches = row.get("map_batches", [])
            if batches:
                for batch in batches:
                    idx = int(batch.get("batch_index", 0)) + 1
                    total = batch.get("batch_total", len(batches))
                    with st.chat_message("assistant"):
                        st.markdown(f"Map summary {idx}/{total}")
                        st.code(str(batch.get("batch_summary", "")))

            if row.get("final_summary"):
                with st.chat_message("assistant"):
                    st.markdown("Итоговый Reduce summary")
                    st.code(str(row["final_summary"]))

            if row.get("notification_text"):
                with st.chat_message("assistant"):
                    st.markdown("Уведомление для SRE")
                    st.code(str(row["notification_text"]))

            if row.get("error"):
                with st.chat_message("assistant"):
                    st.error(f"Ошибка обработки: {row['error']}")


def _ui_runtime_config() -> Dict[str, Any]:
    masked_password = "***" if bool(PROM_PASSWORD) else ""
    masked_ch_metrics_password = "***" if bool(CLICKHOUSE_METRICS_PASSWORD) else ""
    masked_ch_predictions_password = "***" if bool(CLICKHOUSE_PREDICTIONS_PASSWORD) else ""
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
        "CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY": ch_query,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_HOST": CLICKHOUSE_METRICS_HOST,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_PORT": CLICKHOUSE_METRICS_PORT,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_USERNAME": CLICKHOUSE_METRICS_USERNAME,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_PASSWORD": masked_ch_metrics_password,
        "CONTROL_PLANE_CLICKHOUSE_METRICS_SECURE": CLICKHOUSE_METRICS_SECURE,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST": CLICKHOUSE_PREDICTIONS_HOST,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PORT": CLICKHOUSE_PREDICTIONS_PORT,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_USERNAME": CLICKHOUSE_PREDICTIONS_USERNAME,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PASSWORD": masked_ch_predictions_password,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_SECURE": CLICKHOUSE_PREDICTIONS_SECURE,
        "CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_QUERY": ch_predictions_query,
        "CONTROL_PLANE_ANOMALY_DETECTOR": ANOMALY_DETECTOR,
        "CONTROL_PLANE_DATA_LOOKBACK_MINUTES": DATA_LOOKBACK_MINUTES,
        "CONTROL_PLANE_PREDICTION_LOOKAHEAD_MINUTES": PREDICTION_LOOKAHEAD_MINUTES,
        "CONTROL_PLANE_ANALYZE_TOP_N_ANOMALIES": ANALYZE_TOP_N_ANOMALIES,
        "CONTROL_PLANE_LOOPBACK_MINUTES": LOOPBACK_MINUTES,
        "CONTROL_PLANE_PROCESS_ALERTS": PROCESS_ALERTS,
        "CONTROL_PLANE_PROM_METRICS_MAX_POINTS": PROM_MAX_POINTS,
        "CONTROL_PLANE_PROM_METRICS_SERIES_INDEX": PROM_SERIES_INDEX,
        "CONTROL_PLANE_FORECAST_SERVICE": FORECAST_SERVICE,
        "CONTROL_PLANE_FORECAST_METRIC_NAME": FORECAST_METRIC_NAME,
        "CONTROL_PLANE_FORECAST_TYPE": FORECAST_TYPE,
        "CONTROL_PLANE_PREDICTION_KIND": PREDICTION_KIND,
        "CONTROL_PLANE_PROM_METRICS_PASSWORD": masked_password,
    }


def main() -> None:
    st.set_page_config(page_title="Control Plane", layout="wide")
    _ensure_runtime()

    with st.sidebar:
        st.subheader("Параметры (.env)")
        params = _ui_runtime_config()
        params_df = pd.DataFrame(
            [{"param": key, "value": str(value)} for key, value in params.items()]
        )
        st.dataframe(params_df, hide_index=True, use_container_width=True, height=520)
        st.caption("Параметры редактируются через файл .env")
        run_clicked = st.button("Запустить 1 итерацию", type="primary", use_container_width=True)

    runtime_error_placeholder = st.empty()
    graph_placeholder = st.empty()
    anomalies_table_placeholder = st.empty()
    anomaly_cards_placeholder = st.empty()
    had_stage_error = False

    summary_state: Dict[str, Any] = {
        "by_idx": {},
        "selected_anomalies": {},
        "merged_df": None,
        "predictions_df": None,
        "actual_end_ts": None,
        "empty_message": None,
    }
    ui_data: Dict[str, Any] = {
        "actual_df": None,
        "anomalies_df": pd.DataFrame(columns=["timestamp", "value", "source"]),
    }

    def _render_graph(
        actual_df: Optional[pd.DataFrame],
        predictions_df: Optional[pd.DataFrame],
        anomalies_df: Optional[pd.DataFrame],
    ) -> None:
        with graph_placeholder.container():
            st.markdown("1. График прошлое + будущее с отмеченными аномалиями")
            if (
                actual_df is None
                or predictions_df is None
                or anomalies_df is None
                or actual_df.empty
            ):
                return
            try:
                fig_path = visualize_combined(
                    actual_df,
                    predictions_df,
                    anomalies_df,
                    output_dir=PLOTS_DIR,
                )
                if Path(fig_path).exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.caption("Файл графика не найден")
            except Exception as exc:
                st.error(f"Ошибка построения графика: {exc}")

    def _render_anomalies_table(anomalies_df: Optional[pd.DataFrame]) -> None:
        with anomalies_table_placeholder.container():
            st.markdown("2. Табличка с найденными аномалиями")
            if anomalies_df is None or anomalies_df.empty:
                st.info("Аномалии не найдены за выбранное окно.")
                return
            anomaly_columns = [c for c in ["timestamp", "value", "source"] if c in anomalies_df.columns]
            st.dataframe(anomalies_df[anomaly_columns], use_container_width=True)

    def _render_analysis_cards() -> None:
        _render_anomaly_cards(
            anomaly_cards_placeholder,
            summary_state,
            summary_state.get("merged_df"),
            summary_state.get("predictions_df"),
            summary_state.get("actual_end_ts"),
            LOOPBACK_MINUTES,
        )

    if not run_clicked:
        return

    def _on_stage(stage: str, progress: int, payload: Dict[str, Any]) -> None:
        nonlocal had_stage_error
        log_event(
            logger,
            "streamlit.on_stage",
            stage=stage,
            progress=progress,
            **_payload_for_log(payload),
        )
        if stage == "stage_error":
            had_stage_error = True
            stage_name = payload.get("stage_name", "unknown")
            error_text = payload.get("error", "Unknown error")
            with runtime_error_placeholder.container():
                st.error(f"Ошибка на этапе `{stage_name}`: {error_text}")
            return

        if stage == "process_selected":
            selected = payload.get("recent_anomalies", [])
            summary_state["selected_anomalies"] = {
                str(idx): anomaly for idx, anomaly in enumerate(selected)
            }
            summary_state["empty_message"] = None

        if stage == "process_live":
            event = payload.get("event")
            idx = payload.get("index")
            timestamp = payload.get("timestamp")
            if idx is not None:
                key = str(idx)
                known_anomaly = summary_state.get("selected_anomalies", {}).get(key, {})
                row = summary_state["by_idx"].setdefault(
                    key,
                    {
                        "anomaly_idx": int(idx) + 1,
                        "anomaly": known_anomaly,
                        "timestamp": timestamp,
                        "status": "queued",
                        "elapsed_sec": None,
                        "summary_len": None,
                        "error": None,
                        "map_batches": [],
                        "final_summary": None,
                        "notification_text": None,
                    },
                )
                if known_anomaly and not row.get("anomaly"):
                    row["anomaly"] = known_anomaly
                if timestamp:
                    row["timestamp"] = timestamp
                    row.setdefault("anomaly", {})
                    if not row["anomaly"].get("timestamp"):
                        row["anomaly"]["timestamp"] = timestamp
                if event == "anomaly_start":
                    row["status"] = "queued"
                if event == "summary_start":
                    row["status"] = "summarizing"
                elif event == "map_start":
                    row["status"] = "map"
                    row["map_batches"] = []
                elif event == "map_batch":
                    row["status"] = "map"
                    row.setdefault("map_batches", []).append(
                        {
                            "batch_index": payload.get("batch_index"),
                            "batch_total": payload.get("batch_total"),
                            "batch_summary": payload.get("batch_summary"),
                        }
                    )
                elif event == "map_done":
                    row["status"] = "reduce"
                elif event == "reduce_start":
                    row["status"] = "reduce"
                elif event == "summary_done":
                    row["status"] = "summary_ready"
                    row["elapsed_sec"] = payload.get("elapsed_sec")
                    row["summary_len"] = payload.get("summary_len")
                elif event == "reduce_done":
                    row["status"] = "summary_ready"
                    if payload.get("summary"):
                        row["final_summary"] = str(payload.get("summary"))
                elif event == "notification_ready":
                    row["notification_text"] = payload.get("notification_text")
                elif event == "alert_start":
                    row["status"] = "sending_alert"
                elif event == "alert_done":
                    row["status"] = "done"
                elif event == "anomaly_done":
                    row["status"] = "done"
                elif event == "anomaly_error":
                    row["status"] = "error"
                    row["error"] = payload.get("error")

            _render_analysis_cards()

        if stage == "fetch_done":
            stage_actual_df = payload.get("actual_df")
            summary_state["predictions_df"] = payload.get("predictions_df")
            ui_data["actual_df"] = stage_actual_df
            if isinstance(stage_actual_df, pd.DataFrame) and not stage_actual_df.empty:
                summary_state["actual_end_ts"] = pd.to_datetime(
                    stage_actual_df["timestamp"], utc=True
                ).max()
            summary_state["predictions_df"] = _only_future_predictions(
                summary_state.get("predictions_df"),
                summary_state.get("actual_end_ts"),
            )

        if stage == "detect_done":
            summary_state["merged_df"] = payload.get("merged_df")
            ui_data["anomalies_df"] = payload.get("anomalies_df")
            anomalies_df = ui_data.get("anomalies_df")
            if isinstance(anomalies_df, pd.DataFrame) and anomalies_df.empty:
                summary_state["empty_message"] = "Аномалии не найдены. Пошаговый разбор не требуется."
            else:
                summary_state["empty_message"] = None
            _render_graph(
                ui_data.get("actual_df"),
                summary_state.get("predictions_df"),
                ui_data.get("anomalies_df"),
            )
            _render_anomalies_table(ui_data.get("anomalies_df"))
            _render_analysis_cards()

        if stage == "process_skip":
            reason = str(payload.get("reason", ""))
            if reason == "no_anomalies":
                summary_state["empty_message"] = "Аномалии не найдены. Пошаговый разбор не требуется."
            elif reason == "alerts_disabled":
                summary_state["empty_message"] = (
                    "Обработка аномалий отключена настройкой `CONTROL_PLANE_PROCESS_ALERTS=false`."
                )
            _render_analysis_cards()

    try:
        result = run_single_iteration(
            test_mode=TEST_MODE,
            query=PROM_QUERY,
            detector_name=ANOMALY_DETECTOR,
            data_lookback_minutes=DATA_LOOKBACK_MINUTES,
            prediction_lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES,
            analyze_top_n=ANALYZE_TOP_N_ANOMALIES,
            process_lookback_minutes=LOOPBACK_MINUTES,
            process_alerts=PROCESS_ALERTS,
            on_stage=_on_stage,
        )
    except Exception as exc:
        logger.exception("Streamlit iteration failed")
        if not had_stage_error:
            st.error(f"Ошибка выполнения: {exc}")
        st.stop()

    actual_end_ts = (
        pd.to_datetime(result["actual_df"]["timestamp"], utc=True).max()
        if not result["actual_df"].empty
        else None
    )
    future_predictions_df = _only_future_predictions(result["predictions_df"], actual_end_ts)
    summary_state["actual_end_ts"] = actual_end_ts
    summary_state["predictions_df"] = future_predictions_df
    summary_state["merged_df"] = result["merged_df"]
    ui_data["actual_df"] = result["actual_df"]
    ui_data["anomalies_df"] = result["anomalies_df"]

    _render_graph(
        ui_data.get("actual_df"),
        summary_state.get("predictions_df"),
        ui_data.get("anomalies_df"),
    )
    _render_anomalies_table(ui_data.get("anomalies_df"))
    _render_analysis_cards()


if __name__ == "__main__":
    main()
