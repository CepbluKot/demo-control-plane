from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
import logging
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

MSK = timezone(timedelta(hours=3))


def _to_msk_ts(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize(MSK)
    return ts.tz_convert(MSK)


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
        def _fmt(value: Any) -> str:
            ts = _to_msk_ts(value)
            if pd.isna(ts):
                return str(value)
            return ts.strftime("%Y-%m-%d %H:%M:%S.%f MSK")

        out[col] = out[col].apply(_fmt).astype(str)
    return out


@dataclass(frozen=True)
class ControlPlanePageDeps:
    logger: logging.Logger
    log_event: Callable[..., None]
    payload_for_log: Callable[[Dict[str, Any]], Dict[str, Any]]
    run_single_iteration: Callable[..., Dict[str, Any]]
    visualize_combined: Callable[..., str]
    build_predictions_focus_figure: Callable[[Optional[pd.DataFrame], Optional[pd.DataFrame]], Optional[plt.Figure]]
    only_future_predictions: Callable[[Optional[pd.DataFrame], Optional[pd.Timestamp]], pd.DataFrame]
    render_anomaly_cards: Callable[..., None]
    plots_dir: Path
    anomalies_table_height: int
    loopback_minutes: int
    test_mode: bool
    prom_query: str
    anomaly_detector: str
    data_lookback_minutes: int
    prediction_lookahead_minutes: int
    analyze_top_n: int
    process_alerts: bool
    logs_fetch_mode: str
    logs_tail_limit: int


def render_control_plane_page(deps: ControlPlanePageDeps) -> None:
    with st.sidebar:
        run_clicked = st.button("Запустить 1 итерацию", type="primary", use_container_width=True)

    mode_notice_placeholder = st.empty()
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
        "predictions_all_df": None,
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
        predictions_all_df: Optional[pd.DataFrame],
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
                fig_path = deps.visualize_combined(
                    actual_df,
                    predictions_df,
                    anomalies_df,
                    output_dir=deps.plots_dir,
                )
                if Path(fig_path).exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.caption("Файл графика не найден")

                focus_fig = deps.build_predictions_focus_figure(
                    actual_df=actual_df,
                    predictions_df=predictions_all_df if predictions_all_df is not None else predictions_df,
                )
                if focus_fig is not None:
                    st.markdown(
                        "Дополнительный единый график: история и все предикты "
                        "(две линии с разным стилем)"
                    )
                    st.pyplot(focus_fig, use_container_width=True)
                    plt.close(focus_fig)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ошибка построения графика: {exc}")

    def _render_anomalies_table(anomalies_df: Optional[pd.DataFrame]) -> None:
        with anomalies_table_placeholder.container():
            st.markdown("2. Табличка с найденными аномалиями")
            if anomalies_df is None or anomalies_df.empty:
                st.info("Аномалии не найдены за выбранное окно.")
                return
            anomaly_columns = [c for c in ["timestamp", "value", "source"] if c in anomalies_df.columns]
            st.dataframe(
                _format_table_timestamps(anomalies_df[anomaly_columns]),
                use_container_width=True,
                hide_index=True,
                height=deps.anomalies_table_height,
            )

    def _render_analysis_cards() -> None:
        deps.render_anomaly_cards(
            anomaly_cards_placeholder,
            summary_state,
            summary_state.get("merged_df"),
            summary_state.get("predictions_df"),
            summary_state.get("actual_end_ts"),
            deps.loopback_minutes,
        )

    if not run_clicked:
        return

    logs_fetch_mode = str(deps.logs_fetch_mode or "time_window").strip().lower()
    if logs_fetch_mode in ("tail_n_logs", "tail_n", "last_n_logs"):
        with mode_notice_placeholder.container():
            st.info(
                "Режим суммаризации логов: по количеству. "
                f"Для каждой аномалии берём последние `{int(deps.logs_tail_limit)}` логов "
                "(tail_n_logs)."
            )
    else:
        with mode_notice_placeholder.container():
            st.info(
                "Режим суммаризации логов: по датам. "
                f"Для каждой аномалии берём окно `±{int(deps.loopback_minutes)}` минут "
                "(time_window)."
            )

    def _on_stage(stage: str, progress: int, payload: Dict[str, Any]) -> None:
        nonlocal had_stage_error
        deps.log_event(
            deps.logger,
            "streamlit.on_stage",
            stage=stage,
            progress=progress,
            **deps.payload_for_log(payload),
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
                        "logs_processed": 0,
                        "logs_total": None,
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
                    row["logs_processed"] = payload.get("rows_processed", 0)
                    if payload.get("rows_total") is not None:
                        row["logs_total"] = payload.get("rows_total")
                elif event == "map_batch":
                    row["status"] = "map"
                    row.setdefault("map_batches", []).append(
                        {
                            "batch_index": payload.get("batch_index"),
                            "batch_total": payload.get("batch_total"),
                            "batch_summary": payload.get("batch_summary"),
                            "batch_logs_count": payload.get("batch_logs_count"),
                            "batch_logs": payload.get("batch_logs", []),
                            "batch_period_start": payload.get("batch_period_start"),
                            "batch_period_end": payload.get("batch_period_end"),
                        }
                    )
                    if payload.get("rows_processed") is not None:
                        row["logs_processed"] = payload.get("rows_processed")
                    else:
                        current_processed = int(row.get("logs_processed") or 0)
                        current_processed += int(payload.get("batch_logs_count") or 0)
                        row["logs_processed"] = current_processed
                    if payload.get("rows_total") is not None:
                        row["logs_total"] = payload.get("rows_total")
                elif event == "map_done":
                    row["status"] = "reduce"
                    if payload.get("rows_processed") is not None:
                        row["logs_processed"] = payload.get("rows_processed")
                    if payload.get("rows_total") is not None:
                        row["logs_total"] = payload.get("rows_total")
                elif event == "reduce_start":
                    row["status"] = "reduce"
                elif event == "summary_progress":
                    if payload.get("rows_processed") is not None:
                        row["logs_processed"] = payload.get("rows_processed")
                    if payload.get("rows_total") is not None:
                        row["logs_total"] = payload.get("rows_total")
                elif event == "summary_done":
                    row["status"] = "summary_ready"
                    row["elapsed_sec"] = payload.get("elapsed_sec")
                    row["summary_len"] = payload.get("summary_len")
                    if payload.get("rows_processed") is not None:
                        row["logs_processed"] = payload.get("rows_processed")
                    if payload.get("rows_total") is not None:
                        row["logs_total"] = payload.get("rows_total")
                elif event == "reduce_done":
                    row["status"] = "summary_ready"
                    if payload.get("summary"):
                        row["final_summary"] = str(payload.get("summary"))
                    if payload.get("rows_processed") is not None:
                        row["logs_processed"] = payload.get("rows_processed")
                    if payload.get("rows_total") is not None:
                        row["logs_total"] = payload.get("rows_total")
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
            stage_predictions_df = payload.get("predictions_df")
            summary_state["predictions_all_df"] = stage_predictions_df
            summary_state["predictions_df"] = stage_predictions_df
            ui_data["actual_df"] = stage_actual_df
            if isinstance(stage_actual_df, pd.DataFrame) and not stage_actual_df.empty:
                summary_state["actual_end_ts"] = stage_actual_df["timestamp"].apply(_to_msk_ts).max()
            summary_state["predictions_df"] = deps.only_future_predictions(
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
                summary_state.get("predictions_all_df"),
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
        result = deps.run_single_iteration(
            test_mode=deps.test_mode,
            query=deps.prom_query,
            detector_name=deps.anomaly_detector,
            data_lookback_minutes=deps.data_lookback_minutes,
            prediction_lookahead_minutes=deps.prediction_lookahead_minutes,
            analyze_top_n=deps.analyze_top_n,
            process_lookback_minutes=deps.loopback_minutes,
            process_alerts=deps.process_alerts,
            on_stage=_on_stage,
        )
    except Exception as exc:  # noqa: BLE001
        deps.logger.exception("Streamlit iteration failed")
        if not had_stage_error:
            st.error(f"Ошибка выполнения: {exc}")
        st.stop()

    actual_end_ts = (
        result["actual_df"]["timestamp"].apply(_to_msk_ts).max()
        if not result["actual_df"].empty
        else None
    )
    future_predictions_df = deps.only_future_predictions(result["predictions_df"], actual_end_ts)
    summary_state["actual_end_ts"] = actual_end_ts
    summary_state["predictions_all_df"] = result["predictions_df"]
    summary_state["predictions_df"] = future_predictions_df
    summary_state["merged_df"] = result["merged_df"]
    ui_data["actual_df"] = result["actual_df"]
    ui_data["anomalies_df"] = result["anomalies_df"]

    _render_graph(
        ui_data.get("actual_df"),
        summary_state.get("predictions_df"),
        summary_state.get("predictions_all_df"),
        ui_data.get("anomalies_df"),
    )
    _render_anomalies_table(ui_data.get("anomalies_df"))
    _render_analysis_cards()
