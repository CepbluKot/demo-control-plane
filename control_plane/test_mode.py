from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from .config import (
    TEST_MODE_NOISE_SCALE,
    TEST_MODE_SEED,
    TEST_MODE_SPIKE_RATE,
    TEST_MODE_SPIKE_SCALE,
)
from .trace import log_dataframe, log_event
from .timeseries import step_to_pandas_freq

logger = logging.getLogger(__name__)


def generate_mock_data(
    start_time: datetime,
    end_time: datetime,
    step: str,
    lookahead_minutes: int,
    seed: int = TEST_MODE_SEED,
    spike_rate: float = TEST_MODE_SPIKE_RATE,
    spike_scale: float = TEST_MODE_SPIKE_SCALE,
    noise_scale: float = TEST_MODE_NOISE_SCALE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Генерирует синтетические ряды для тестового режима:
    - actual: базовый тренд + синус + шум + редкие всплески
    - predicted: сглаженный базовый тренд без всплесков
    """
    log_event(
        logger=logger,
        event="generate_mock_data.start",
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        step=step,
        lookahead_minutes=lookahead_minutes,
        seed=seed,
        spike_rate=spike_rate,
        spike_scale=spike_scale,
        noise_scale=noise_scale,
    )
    freq = step_to_pandas_freq(step)
    prediction_end = end_time + timedelta(minutes=lookahead_minutes)
    timestamps = pd.date_range(start=start_time, end=prediction_end, freq=freq, tz="UTC")

    if len(timestamps) == 0:
        actual_df = pd.DataFrame(columns=["timestamp", "value"])
        pred_df = pd.DataFrame(columns=["timestamp", "predicted"])
        log_event(logger=logger, event="generate_mock_data.empty")
        return actual_df, pred_df

    t = np.arange(len(timestamps))
    base = 100 + 8 * np.sin(2 * np.pi * t / 144) + 0.02 * t
    rng = np.random.default_rng(seed)

    predicted = base + rng.normal(0, noise_scale * 0.4, size=len(timestamps))

    actual_mask = timestamps <= end_time
    actual_times = timestamps[actual_mask]
    actual = base[actual_mask] + rng.normal(0, noise_scale, size=actual_mask.sum())

    spike_mask = rng.random(actual_mask.sum()) < spike_rate
    if spike_mask.any():
        actual[spike_mask] += rng.normal(spike_scale, spike_scale * 0.3, size=spike_mask.sum())
    log_event(
        logger=logger,
        event="generate_mock_data.spikes",
        spikes=int(spike_mask.sum()),
        total=int(actual_mask.sum()),
    )

    actual_df = pd.DataFrame({"timestamp": actual_times, "value": actual})
    pred_df = pd.DataFrame({"timestamp": timestamps, "predicted": predicted})
    log_dataframe(logger, "mock_actual_df", actual_df)
    log_dataframe(logger, "mock_pred_df", pred_df)
    log_event(
        logger=logger,
        event="generate_mock_data.done",
        actual_rows=len(actual_df),
        predicted_rows=len(pred_df),
    )
    return actual_df, pred_df
