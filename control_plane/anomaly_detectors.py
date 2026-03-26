from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Type

import numpy as np
import pandas as pd

from .timeseries import (
    detect_anomalies_from_merged,
    detect_anomalies_on_series,
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


def _ensure_at_least_one_anomaly(
    out_df: pd.DataFrame,
    *,
    score_column: str = "residual",
    reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if out_df.empty:
        return out_df, out_df.copy()
    anomalies = out_df[out_df["is_anomaly"]].copy()
    if not anomalies.empty:
        return out_df, anomalies
    score = pd.to_numeric(out_df.get(score_column), errors="coerce").abs().fillna(0.0)
    picked_idx = score.idxmax()
    out_df.loc[picked_idx, "is_anomaly"] = True
    anomalies = out_df[out_df["is_anomaly"]].copy()
    log_event(
        logger,
        "_ensure_at_least_one_anomaly.applied",
        reason=reason,
        score_column=score_column,
        picked_index=str(picked_idx),
    )
    return out_df, anomalies


@register_detector
class RollingIQRDetector:
    name = "rolling_iqr"

    def __init__(
        self,
        iqr_window: int = 60,
        iqr_scale: float = 1.5,
        min_periods: int = 30,
        zscore_threshold: float = 3.0,
        **_: Any,
    ) -> None:
        self.iqr_window = iqr_window
        self.iqr_scale = iqr_scale
        self.min_periods = min_periods
        self.fallback_zscore = zscore_threshold
        log_event(
            logger,
            "RollingIQRDetector.init",
            iqr_window=iqr_window,
            iqr_scale=iqr_scale,
            min_periods=min_periods,
            fallback_zscore=zscore_threshold,
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
        if merged_df.empty:
            log_event(
                logger,
                "RollingIQRDetector.detect.fallback_actual_only",
                reason="no_overlap_with_predictions",
                actual_rows=len(actual_df),
                zscore_threshold=self.fallback_zscore,
            )
            series_out, anomalies_df = detect_anomalies_on_series(
                actual_df,
                zscore_threshold=self.fallback_zscore,
            )
            return DetectionResult(merged_df=series_out, anomalies_df=anomalies_df)
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


@register_detector
class PyODECODDetector:
    name = "pyod_ecod"

    def __init__(
        self,
        pyod_contamination: float = 0.05,
        zscore_threshold: float = 3.0,
        **_: Any,
    ) -> None:
        self.contamination = float(pyod_contamination)
        self.fallback_zscore = float(zscore_threshold)
        log_event(
            logger,
            "PyODECODDetector.init",
            contamination=self.contamination,
            fallback_zscore=self.fallback_zscore,
        )

    def detect(
        self,
        actual_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        step: str,
    ) -> DetectionResult:
        log_event(
            logger,
            "PyODECODDetector.detect.start",
            actual_rows=len(actual_df),
            predictions_rows=len(predictions_df),
            step=step,
            contamination=self.contamination,
        )
        merged_df = merge_actual_and_predictions(actual_df, predictions_df, step=step)
        if merged_df.empty:
            log_event(
                logger,
                "PyODECODDetector.detect.fallback_actual_only",
                reason="no_overlap_with_predictions",
                zscore_threshold=self.fallback_zscore,
            )
            series_out, anomalies_df = detect_anomalies_on_series(
                actual_df,
                zscore_threshold=self.fallback_zscore,
            )
            return DetectionResult(merged_df=series_out, anomalies_df=anomalies_df)

        try:
            from pyod.models.ecod import ECOD
        except Exception as exc:
            raise ImportError(
                "Detector 'pyod_ecod' requires package 'pyod'. "
                "Install with: pip install pyod"
            ) from exc

        residual = pd.to_numeric(
            merged_df["value"] - merged_df["predicted"],
            errors="coerce",
        ).fillna(0.0)
        x = residual.to_numpy(dtype=float).reshape(-1, 1)
        model = ECOD(contamination=self.contamination)
        model.fit(x)
        labels = model.predict(x).astype(bool)

        out = merged_df.copy()
        out["residual"] = residual
        out["is_anomaly"] = labels
        out["anomaly_score"] = getattr(model, "decision_scores_", np.zeros(len(out)))
        out, anomalies_df = _ensure_at_least_one_anomaly(
            out,
            score_column="anomaly_score",
            reason="pyod_ecod_no_hits",
        )
        log_event(
            logger,
            "PyODECODDetector.detect.done",
            merged_rows=len(out),
            anomalies_rows=len(anomalies_df),
        )
        return DetectionResult(merged_df=out, anomalies_df=anomalies_df)


@register_detector
class PyODIForestDetector:
    name = "pyod_iforest"

    def __init__(
        self,
        pyod_contamination: float = 0.05,
        pyod_random_state: int = 42,
        zscore_threshold: float = 3.0,
        **_: Any,
    ) -> None:
        self.contamination = float(pyod_contamination)
        self.random_state = int(pyod_random_state)
        self.fallback_zscore = float(zscore_threshold)
        log_event(
            logger,
            "PyODIForestDetector.init",
            contamination=self.contamination,
            random_state=self.random_state,
            fallback_zscore=self.fallback_zscore,
        )

    def detect(
        self,
        actual_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        step: str,
    ) -> DetectionResult:
        log_event(
            logger,
            "PyODIForestDetector.detect.start",
            actual_rows=len(actual_df),
            predictions_rows=len(predictions_df),
            step=step,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        merged_df = merge_actual_and_predictions(actual_df, predictions_df, step=step)
        if merged_df.empty:
            log_event(
                logger,
                "PyODIForestDetector.detect.fallback_actual_only",
                reason="no_overlap_with_predictions",
                zscore_threshold=self.fallback_zscore,
            )
            series_out, anomalies_df = detect_anomalies_on_series(
                actual_df,
                zscore_threshold=self.fallback_zscore,
            )
            return DetectionResult(merged_df=series_out, anomalies_df=anomalies_df)

        try:
            from pyod.models.iforest import IForest
        except Exception as exc:
            raise ImportError(
                "Detector 'pyod_iforest' requires package 'pyod'. "
                "Install with: pip install pyod"
            ) from exc

        residual = pd.to_numeric(
            merged_df["value"] - merged_df["predicted"],
            errors="coerce",
        ).fillna(0.0)
        x = residual.to_numpy(dtype=float).reshape(-1, 1)
        model = IForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(x)
        labels = model.predict(x).astype(bool)

        out = merged_df.copy()
        out["residual"] = residual
        out["is_anomaly"] = labels
        out["anomaly_score"] = getattr(model, "decision_scores_", np.zeros(len(out)))
        out, anomalies_df = _ensure_at_least_one_anomaly(
            out,
            score_column="anomaly_score",
            reason="pyod_iforest_no_hits",
        )
        log_event(
            logger,
            "PyODIForestDetector.detect.done",
            merged_rows=len(out),
            anomalies_rows=len(anomalies_df),
        )
        return DetectionResult(merged_df=out, anomalies_df=anomalies_df)


@register_detector
class RupturesPeltDetector:
    name = "ruptures_pelt"

    def __init__(
        self,
        ruptures_penalty: float = 8.0,
        ruptures_model: str = "rbf",
        zscore_threshold: float = 3.0,
        **_: Any,
    ) -> None:
        self.penalty = float(ruptures_penalty)
        self.model = str(ruptures_model)
        self.fallback_zscore = float(zscore_threshold)
        log_event(
            logger,
            "RupturesPeltDetector.init",
            penalty=self.penalty,
            model=self.model,
            fallback_zscore=self.fallback_zscore,
        )

    def detect(
        self,
        actual_df: pd.DataFrame,
        predictions_df: pd.DataFrame,
        step: str,
    ) -> DetectionResult:
        log_event(
            logger,
            "RupturesPeltDetector.detect.start",
            actual_rows=len(actual_df),
            predictions_rows=len(predictions_df),
            step=step,
            penalty=self.penalty,
            model=self.model,
        )
        merged_df = merge_actual_and_predictions(actual_df, predictions_df, step=step)
        if merged_df.empty:
            log_event(
                logger,
                "RupturesPeltDetector.detect.fallback_actual_only",
                reason="no_overlap_with_predictions",
                zscore_threshold=self.fallback_zscore,
            )
            series_out, anomalies_df = detect_anomalies_on_series(
                actual_df,
                zscore_threshold=self.fallback_zscore,
            )
            return DetectionResult(merged_df=series_out, anomalies_df=anomalies_df)

        try:
            import ruptures as rpt
        except Exception as exc:
            raise ImportError(
                "Detector 'ruptures_pelt' requires package 'ruptures'. "
                "Install with: pip install ruptures"
            ) from exc

        residual = pd.to_numeric(
            merged_df["value"] - merged_df["predicted"],
            errors="coerce",
        ).fillna(0.0)
        signal = residual.to_numpy(dtype=float)
        if len(signal) < 3:
            out, anomalies_df = detect_anomalies_from_merged(
                merged_df,
                zscore_threshold=self.fallback_zscore,
            )
            return DetectionResult(merged_df=out, anomalies_df=anomalies_df)

        algo = rpt.Pelt(model=self.model).fit(signal)
        bkps = algo.predict(pen=self.penalty)
        labels = np.zeros(len(signal), dtype=bool)
        for bkp in bkps[:-1]:
            idx = int(min(max(bkp - 1, 0), len(signal) - 1))
            labels[idx] = True

        out = merged_df.copy()
        out["residual"] = residual
        out["is_anomaly"] = labels
        # simple local change magnitude score to rank forced fallback if needed
        residual_diff = residual.diff().abs().fillna(0.0)
        out["anomaly_score"] = residual_diff
        out, anomalies_df = _ensure_at_least_one_anomaly(
            out,
            score_column="anomaly_score",
            reason="ruptures_pelt_no_hits",
        )
        log_event(
            logger,
            "RupturesPeltDetector.detect.done",
            merged_rows=len(out),
            anomalies_rows=len(anomalies_df),
            breakpoints=len(bkps),
        )
        return DetectionResult(merged_df=out, anomalies_df=anomalies_df)


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
