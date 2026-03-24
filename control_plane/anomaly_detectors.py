from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Type

import pandas as pd

from .timeseries import (
    detect_anomalies_from_merged,
    detect_anomalies_rolling_iqr,
    merge_actual_and_predictions,
)
from .trace import log_event

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    merged_df: pd.DataFrame
    anomalies_df: pd.DataFrame


class AnomalyDetector(Protocol):
    name: str

    def detect(
        self,
        actual_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        step: str,
    ) -> DetectionResult:
        ...


_REGISTRY: Dict[str, Type[Any]] = {}


def register_detector(cls: Type[Any]) -> Type[Any]:
    _REGISTRY[getattr(cls, "name", cls.__name__)] = cls
    log_event(
        logger,
        "register_detector",
        name=getattr(cls, "name", cls.__name__),
        cls=str(cls),
    )
    return cls


@register_detector
class RollingIQRDetector:
    name = "rolling_iqr"

    def __init__(
        self,
        iqr_window: int = 60,
        iqr_scale: float = 1.5,
        min_periods: int = 30,
        **_: Any,
    ) -> None:
        self.iqr_window = iqr_window
        self.iqr_scale = iqr_scale
        self.min_periods = min_periods
        log_event(
            logger,
            "RollingIQRDetector.init",
            iqr_window=iqr_window,
            iqr_scale=iqr_scale,
            min_periods=min_periods,
        )

    def detect(
        self,
        actual_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        step: str,
    ) -> DetectionResult:
        log_event(
            logger,
            "RollingIQRDetector.detect.start",
            actual_rows=len(actual_df),
            predictions_rows=len(predictions_df),
            step=step,
        )
        merged_df = merge_actual_and_predictions(actual_df, predictions_df, step=step)
        merged_df, anomalies_df = detect_anomalies_rolling_iqr(
            merged_df,
            window=self.iqr_window,
            iqr_scale=self.iqr_scale,
            min_periods=self.min_periods,
        )
        log_event(
            logger,
            "RollingIQRDetector.detect.done",
            merged_rows=len(merged_df),
            anomalies_rows=len(anomalies_df),
        )
        return DetectionResult(merged_df=merged_df, anomalies_df=anomalies_df)


@register_detector
class ResidualZScoreDetector:
    name = "residual_zscore"

    def __init__(self, zscore_threshold: float = 3.5, **_: Any) -> None:
        self.zscore_threshold = zscore_threshold
        log_event(logger, "ResidualZScoreDetector.init", zscore_threshold=zscore_threshold)

    def detect(
        self,
        actual_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        step: str,
    ) -> DetectionResult:
        log_event(
            logger,
            "ResidualZScoreDetector.detect.start",
            actual_rows=len(actual_df),
            predictions_rows=len(predictions_df),
            step=step,
        )
        merged_df = merge_actual_and_predictions(actual_df, predictions_df, step=step)
        merged_df, anomalies_df = detect_anomalies_from_merged(
            merged_df,
            zscore_threshold=self.zscore_threshold,
        )
        log_event(
            logger,
            "ResidualZScoreDetector.detect.done",
            merged_rows=len(merged_df),
            anomalies_rows=len(anomalies_df),
        )
        return DetectionResult(merged_df=merged_df, anomalies_df=anomalies_df)


def _load_detector_from_path(path: str) -> Type[Any]:
    log_event(logger, "_load_detector_from_path.start", path=path)
    if ":" in path:
        module_path, cls_name = path.split(":", 1)
    elif "." in path:
        module_path, cls_name = path.rsplit(".", 1)
    else:
        raise ValueError(
            "Unknown detector '%s'. Available: %s" % (path, ", ".join(sorted(_REGISTRY)))
        )
    module = importlib.import_module(module_path)
    detector_cls = getattr(module, cls_name, None)
    if detector_cls is None:
        raise ValueError(f"Detector class '{cls_name}' not found in {module_path}")
    log_event(logger, "_load_detector_from_path.done", module=module_path, cls=cls_name)
    return detector_cls


def get_anomaly_detector(name: str, **kwargs: Any) -> AnomalyDetector:
    log_event(logger, "get_anomaly_detector.start", name=name, kwargs=kwargs)
    if name in _REGISTRY:
        cls = _REGISTRY[name]
        detector = cls(**kwargs)
        log_event(logger, "get_anomaly_detector.from_registry", name=name)
        return detector
    detector_cls = _load_detector_from_path(name)
    logger.info("Loaded detector from path: %s", name)
    detector = detector_cls()
    log_event(logger, "get_anomaly_detector.from_path", name=name)
    return detector
