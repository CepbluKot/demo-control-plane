import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .trace import log_dataframe, log_event

logger = logging.getLogger(__name__)


def step_to_pandas_freq(step: str) -> str:
    log_event(logger, "step_to_pandas_freq.start", step=step)
    step = str(step).strip()
    if step.endswith("s"):
        result = f"{int(step[:-1])}S"
        log_event(logger, "step_to_pandas_freq.done", result=result)
        return result
    if step.endswith("m"):
        result = f"{int(step[:-1])}min"
        log_event(logger, "step_to_pandas_freq.done", result=result)
        return result
    if step.endswith("h"):
        result = f"{int(step[:-1])}H"
        log_event(logger, "step_to_pandas_freq.done", result=result)
        return result
    if step.endswith("d"):
        result = f"{int(step[:-1])}D"
        log_event(logger, "step_to_pandas_freq.done", result=result)
        return result
    result = f"{int(step)}S"
    log_event(logger, "step_to_pandas_freq.done", result=result)
    return result


def merge_actual_and_predictions(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    step: str = "1m",
) -> pd.DataFrame:
    log_event(
        logger,
        "merge_actual_and_predictions.start",
        actual_rows=len(actual_df),
        predicted_rows=len(pred_df),
        step=step,
    )
    if actual_df.empty or pred_df.empty:
        log_event(logger, "merge_actual_and_predictions.empty_input")
        return pd.DataFrame(columns=["timestamp", "value", "predicted"])
    freq = step_to_pandas_freq(step)
    act = actual_df.copy()
    pred = pred_df.copy()
    act["timestamp"] = pd.to_datetime(act["timestamp"], utc=True).dt.floor(freq)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True).dt.floor(freq)
    pred = pred.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    merged = act.merge(pred, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "Merge done: actual=%s, predicted=%s, merged=%s",
        len(actual_df),
        len(pred_df),
        len(merged),
    )
    log_dataframe(logger, "merged_actual_predictions", merged)
    return merged


def build_combined_series(
    actual_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Собирает единый ряд: фактические данные + прогнозы только после них.
    """
    if actual_df.empty and predictions_df.empty:
        log_event(logger, "build_combined_series.empty_input")
        return pd.DataFrame(columns=["timestamp", "value", "source"])

    actual = actual_df[["timestamp", "value"]].copy()
    actual["source"] = "actual"
    last_actual_ts = actual["timestamp"].max() if not actual.empty else None

    future_preds = predictions_df.copy()
    if last_actual_ts is not None:
        future_preds = future_preds[future_preds["timestamp"] > last_actual_ts]
    future_preds = future_preds.dropna(subset=["timestamp", "predicted"])
    future = future_preds.rename(columns={"predicted": "value"})[
        ["timestamp", "value"]
    ].copy()
    future["source"] = "forecast"

    combined = pd.concat([actual, future], ignore_index=True)
    if combined.empty:
        return combined
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    combined = combined.dropna(subset=["timestamp", "value"])
    out = combined.sort_values("timestamp").reset_index(drop=True)
    log_dataframe(logger, "combined_series", out)
    return out


def detect_anomalies_from_merged(
    merged_df: pd.DataFrame,
    zscore_threshold: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log_event(
        logger,
        "detect_anomalies_from_merged.start",
        rows=len(merged_df),
        zscore_threshold=zscore_threshold,
    )
    if merged_df.empty:
        empty = pd.DataFrame(
            columns=["timestamp", "value", "predicted", "residual", "is_anomaly"]
        )
        return merged_df, empty
    residual = merged_df["value"] - merged_df["predicted"]
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad == 0:
        std = float(np.std(residual)) or 1.0
        score = (residual - median) / std
    else:
        score = 0.6745 * (residual - median) / mad
    out = merged_df.copy()
    out["residual"] = residual
    out["is_anomaly"] = np.abs(score) > zscore_threshold
    anomalies = out[out["is_anomaly"]].copy()
    logger.info(
        "Anomaly detection done: points=%s, anomalies=%s, z=%.2f",
        len(out),
        len(anomalies),
        zscore_threshold,
    )
    log_dataframe(logger, "anomalies_from_merged", anomalies)
    return out, anomalies


def detect_anomalies_on_series(
    series_df: pd.DataFrame,
    zscore_threshold: float = 3.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log_event(
        logger,
        "detect_anomalies_on_series.start",
        rows=len(series_df),
        zscore_threshold=zscore_threshold,
    )
    if series_df.empty:
        empty = pd.DataFrame(columns=["timestamp", "value", "residual", "is_anomaly", "source"])
        return series_df, empty
    values = pd.to_numeric(series_df["value"], errors="coerce")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        std = float(np.std(values)) or 1.0
        score = (values - median) / std
    else:
        score = 0.6745 * (values - median) / mad
    out = series_df.copy()
    out["residual"] = values - median
    out["is_anomaly"] = np.abs(score) > zscore_threshold
    anomalies = out[out["is_anomaly"]].copy()
    logger.info(
        "Series anomaly detection done: points=%s, anomalies=%s, z=%.2f",
        len(out),
        len(anomalies),
        zscore_threshold,
    )
    log_dataframe(logger, "anomalies_on_series", anomalies)
    return out, anomalies


def detect_anomalies_rolling_iqr(
    merged_df: pd.DataFrame,
    window: int = 60,
    iqr_scale: float = 1.5,
    min_periods: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Детекция аномалий по остаткам (actual - predicted) с адаптивным порогом
    на основе скользящих квантилей (IQR).
    Это более современный и устойчивый подход, чем глобальный z-score.
    """
    log_event(
        logger,
        "detect_anomalies_rolling_iqr.start",
        rows=len(merged_df),
        window=window,
        iqr_scale=iqr_scale,
        min_periods=min_periods,
    )
    if merged_df.empty:
        empty = pd.DataFrame(
            columns=["timestamp", "value", "predicted", "residual", "is_anomaly"]
        )
        return merged_df, empty
    if "predicted" not in merged_df.columns:
        raise ValueError("merged_df должен содержать столбец 'predicted' для расчета остатка")
    if min_periods is None:
        min_periods = max(5, window // 2)

    residual = merged_df["value"] - merged_df["predicted"]
    rolling = residual.rolling(window=window, min_periods=min_periods)
    q1 = rolling.quantile(0.25)
    q3 = rolling.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_scale * iqr
    upper = q3 + iqr_scale * iqr
    is_anomaly = ((residual < lower) | (residual > upper)).fillna(False)

    out = merged_df.copy()
    out["residual"] = residual
    out["is_anomaly"] = is_anomaly
    anomalies = out[out["is_anomaly"]].copy()
    logger.info(
        "Rolling IQR detection done: points=%s, anomalies=%s, window=%s, scale=%.2f",
        len(out),
        len(anomalies),
        window,
        iqr_scale,
    )
    log_dataframe(logger, "rolling_iqr_out", out)
    log_dataframe(logger, "rolling_iqr_anomalies", anomalies)
    return out, anomalies
