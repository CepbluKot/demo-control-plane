import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .api_client import build_api_response
from .anomaly_detectors import get_anomaly_detector
from .config import (
    ANALYZE_TOP_N_ANOMALIES,
    ANOMALY_IQR_MIN_PERIODS,
    ANOMALY_IQR_SCALE,
    ANOMALY_IQR_WINDOW,
    ANOMALY_ZSCORE,
    ANOMALY_DETECTOR,
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
from .logging_config import configure_logging
from .predictions_db import fetch_predictions_from_db
from .processing import process_anomalies
from .prometheus_io import fetch_prometheus_df
from .test_mode import generate_mock_data
from .trace import StageTimer, log_dataframe, log_event
from .utils import to_iso_z
from .visualization import visualize

logger = logging.getLogger(__name__)


def _log_stage(title: str) -> None:
    logger.info("==== %s ====", title)


def run_loop(sleep_seconds: int = 60) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()
    logger.info("Test mode: %s", TEST_MODE)
    logger.debug(
        "Runtime config | sleep_seconds=%s, lookback=%s, step=%s, detector=%s",
        sleep_seconds,
        DATA_LOOKBACK_MINUTES,
        PROM_STEP,
        ANOMALY_DETECTOR,
    )
    logger.info(
        "Artifacts dirs: base=%s, plots=%s, logs=%s",
        ARTIFACTS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
    )
    iteration = 0
    while True:
        iteration += 1
        logger.info("======== LOOP ITERATION %s START ========", iteration)
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=DATA_LOOKBACK_MINUTES)
            logger.info(
                "Loop start: window=%s..%s, step=%s",
                start_time.isoformat(),
                end_time.isoformat(),
                PROM_STEP,
            )

            # 1) Данные из Prometheus + 2) Предсказания из БД
            _log_stage("Data Fetch")
            with StageTimer(logger, "data_fetch"):
                if TEST_MODE:
                    actual_df, predictions_df = generate_mock_data(
                        start_time=start_time,
                        end_time=end_time,
                        step=PROM_STEP,
                        lookahead_minutes=PREDICTION_LOOKAHEAD_MINUTES,
                    )
                    logger.info(
                        "Mock data generated: actual=%s, predicted=%s",
                        len(actual_df),
                        len(predictions_df),
                    )
                else:
                    actual_df = fetch_prometheus_df(
                        query=PROM_QUERY,
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
                        end_time=end_time + timedelta(minutes=PREDICTION_LOOKAHEAD_MINUTES),
                        forecast_type=FORECAST_TYPE,
                        prediction_kind=PREDICTION_KIND,
                    )
                log_dataframe(logger, "actual_df", actual_df)
                log_dataframe(logger, "predictions_df", predictions_df)

            # 3) Детекция аномалий (алгоритм подключается через интерфейс)
            _log_stage("Anomaly Detection")
            with StageTimer(logger, "anomaly_detection"):
                detector = get_anomaly_detector(
                    ANOMALY_DETECTOR,
                    iqr_window=ANOMALY_IQR_WINDOW,
                    iqr_scale=ANOMALY_IQR_SCALE,
                    min_periods=ANOMALY_IQR_MIN_PERIODS,
                    zscore_threshold=ANOMALY_ZSCORE,
                )
                logger.info(
                    "Anomaly detector: %s",
                    getattr(detector, "name", detector.__class__.__name__),
                )
                result = detector.detect(actual_df, predictions_df, step=PROM_STEP)
                merged_df = result.merged_df
                anomalies_df = result.anomalies_df
                if not anomalies_df.empty and "source" not in anomalies_df.columns:
                    anomalies_df["source"] = "actual"
                log_dataframe(logger, "merged_df", merged_df)
                log_dataframe(logger, "anomalies_df", anomalies_df)

            # Список аномалий для дальнейшей обработки
            anomalies = []
            if not anomalies_df.empty:
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
            log_event(
                logger,
                "anomalies_prepared",
                total=len(anomalies),
                first_anomaly=anomalies[0] if anomalies else None,
            )

            _log_stage("Response + Visualization")
            with StageTimer(logger, "response_visualization"):
                api_response = build_api_response(merged_df, predictions_df, anomalies_df)
                log_filename = LOGS_DIR / f"api_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(api_response, f, ensure_ascii=False, indent=2, default=str)
                logger.info("API response logged to %s", log_filename)
                log_event(
                    logger,
                    "api_response_counts",
                    merged=len(api_response.get("data", {}).get("merged_data", [])),
                    predictions=len(api_response.get("data", {}).get("predictions", [])),
                    anomalies=len(api_response.get("data", {}).get("anomalies", [])),
                )
                logger.info("Creating visualization of predictions and anomalies")
                plot_info = visualize(PROM_QUERY, api_response)
                for plot in plot_info:
                    if plot["type"] == "main":
                        logger.info(f"Main plot saved at: {plot['path']}")
                    else:
                        logger.info(
                            f"Anomaly plot saved at: {plot['path']} "
                            f"for anomaly at {plot['anomaly_timestamp']}"
                        )
            _log_stage("Anomaly Processing")
            with StageTimer(logger, "anomaly_processing"):
                if not anomalies:
                    logger.warning("No anomalies found for the given query")
                else:
                    # Берем только аномалии из фактических данных
                    anomalies_for_processing = [a for a in anomalies if a.get("source") == "actual"]
                    sorted_anomalies = sorted(
                        anomalies_for_processing,
                        key=lambda x: datetime.fromisoformat(
                            x["timestamp"].replace("Z", "+00:00")
                        ),
                        reverse=True,
                    )
                    recent_anomalies = sorted_anomalies[:ANALYZE_TOP_N_ANOMALIES]
                    logger.info(
                        "Processing %s most recent anomalies (limit=%s) out of %s total",
                        len(recent_anomalies),
                        ANALYZE_TOP_N_ANOMALIES,
                        len(anomalies),
                    )
                    results = process_anomalies(
                        anomalies=recent_anomalies,
                        lookback_minutes=LOOPBACK_MINUTES,
                        continue_on_error=True,
                        test_mode=TEST_MODE,
                    )
                    for i, result in enumerate(results):
                        logger.info("--- Результат обработки выброса %s ---", i + 1)
                        logger.info("Время выброса: %s", result["anomaly"]["timestamp"])
                        logger.info("Успешно: %s", result["success"])
                        if result["error"]:
                            logger.error("Ошибка: %s", result["error"])
                        if result["summary"]:
                            logger.info("Summary: %s...", str(result["summary"])[:200])
                        if result["alert_result"]:
                            logger.info("Результат алерта: %s", result["alert_result"])
            logger.info("Loop done: anomalies=%s, sleep=%ss", len(anomalies), sleep_seconds)
            logger.info("======== LOOP ITERATION %s END ========", iteration)
        except Exception:
            logger.exception("Loop iteration %s failed with error", iteration)
            logger.info("======== LOOP ITERATION %s END WITH ERROR ========", iteration)
        finally:
            logger.debug("Sleeping for %s seconds", sleep_seconds)
            time.sleep(sleep_seconds)


def main() -> None:
    run_loop()
