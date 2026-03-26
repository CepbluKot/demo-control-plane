import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import streamlit_app


class _FakeDetector:
    name = "fake-detector"

    def detect(self, actual_df: pd.DataFrame, predictions_df: pd.DataFrame, step: str):
        _ = (predictions_df, step)
        merged_df = actual_df.copy()
        anomalies_df = pd.DataFrame(
            columns=["timestamp", "value", "predicted", "residual", "is_anomaly", "source"]
        )
        return SimpleNamespace(merged_df=merged_df, anomalies_df=anomalies_df)


class TestStreamlitIteration(unittest.TestCase):
    def test_test_mode_skips_external_fetchers(self) -> None:
        actual_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-03-25T10:00:00Z", "2026-03-25T10:01:00Z"],
                    utc=True,
                ),
                "value": [10.0, 12.0],
            }
        )
        predictions_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-03-25T10:02:00Z", "2026-03-25T10:03:00Z"],
                    utc=True,
                ),
                "predicted": [11.5, 12.2],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            logs_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(streamlit_app, "_ensure_runtime", return_value=None), patch.object(
                streamlit_app, "LOGS_DIR", logs_dir
            ), patch.object(
                streamlit_app, "generate_mock_data", return_value=(actual_df, predictions_df)
            ), patch.object(
                streamlit_app,
                "fetch_actual_metrics_df",
                side_effect=AssertionError("external actuals fetch should be skipped in test_mode"),
            ) as actual_fetch_mock, patch.object(
                streamlit_app,
                "fetch_predictions_from_db",
                side_effect=AssertionError("external predictions fetch should be skipped in test_mode"),
            ) as pred_fetch_mock, patch.object(
                streamlit_app, "get_anomaly_detector", return_value=_FakeDetector()
            ), patch.object(
                streamlit_app, "visualize", return_value=[]
            ):
                result = streamlit_app.run_single_iteration(
                    test_mode=True,
                    query="up",
                    detector_name="rolling_iqr",
                    data_lookback_minutes=60,
                    prediction_lookahead_minutes=30,
                    analyze_top_n=1,
                    process_lookback_minutes=30,
                    process_alerts=True,
                )

        actual_fetch_mock.assert_not_called()
        pred_fetch_mock.assert_not_called()
        self.assertEqual(len(result["actual_df"]), 2)
        self.assertEqual(len(result["predictions_df"]), 2)
        self.assertEqual(result["detector"], "fake-detector")

    def test_non_test_mode_fetch_error_emits_stage_error(self) -> None:
        events = []
        root_logger = logging.getLogger()
        saved_handlers = list(root_logger.handlers)
        try:
            root_logger.handlers = [logging.NullHandler()]
            with patch.object(streamlit_app, "_ensure_runtime", return_value=None), patch.object(
                streamlit_app,
                "fetch_actual_metrics_df",
                side_effect=RuntimeError("fetch failed"),
            ):
                with self.assertRaises(RuntimeError):
                    streamlit_app.run_single_iteration(
                        test_mode=False,
                        query="up",
                        detector_name="rolling_iqr",
                        data_lookback_minutes=60,
                        prediction_lookahead_minutes=30,
                        analyze_top_n=1,
                        process_lookback_minutes=30,
                        process_alerts=True,
                        on_stage=lambda stage, progress, payload: events.append(
                            (stage, progress, payload)
                        ),
                    )
        finally:
            root_logger.handlers = saved_handlers

        stage_errors = [payload for stage, _, payload in events if stage == "stage_error"]
        self.assertTrue(stage_errors, "Expected stage_error event when fetch fails")
        self.assertEqual(stage_errors[0]["stage_name"], "fetch")
        self.assertIn("RuntimeError: fetch failed", stage_errors[0]["error"])

    def test_prometheus_real_mode_uses_last_month_window(self) -> None:
        actual_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-03-25T10:00:00Z", "2026-03-25T10:01:00Z"],
                    utc=True,
                ),
                "value": [10.0, 12.0],
            }
        )
        predictions_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-03-25T10:02:00Z", "2026-03-25T10:03:00Z"],
                    utc=True,
                ),
                "predicted": [11.5, 12.2],
            }
        )
        captured_window = {}

        def _fake_fetch_actual_metrics_df(**kwargs):
            captured_window["start"] = kwargs["start_time"]
            captured_window["end"] = kwargs["end_time"]
            return actual_df

        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir)
            logs_dir.mkdir(parents=True, exist_ok=True)
            with patch.object(streamlit_app, "_ensure_runtime", return_value=None), patch.object(
                streamlit_app, "LOGS_DIR", logs_dir
            ), patch.object(
                streamlit_app, "METRICS_SOURCE", "prometheus"
            ), patch.object(
                streamlit_app,
                "fetch_actual_metrics_df",
                side_effect=_fake_fetch_actual_metrics_df,
            ), patch.object(
                streamlit_app,
                "fetch_predictions_from_db",
                return_value=predictions_df,
            ), patch.object(
                streamlit_app, "get_anomaly_detector", return_value=_FakeDetector()
            ), patch.object(
                streamlit_app, "visualize", return_value=[]
            ):
                result = streamlit_app.run_single_iteration(
                    test_mode=False,
                    query="up",
                    detector_name="rolling_iqr",
                    data_lookback_minutes=60,
                    prediction_lookahead_minutes=30,
                    analyze_top_n=1,
                    process_lookback_minutes=30,
                    process_alerts=False,
                )

        self.assertEqual(result["detector"], "fake-detector")
        self.assertIn("start", captured_window)
        self.assertIn("end", captured_window)
        delta = captured_window["end"] - captured_window["start"]
        self.assertGreaterEqual(delta.total_seconds(), 30 * 24 * 3600 - 5)
        self.assertLessEqual(delta.total_seconds(), 30 * 24 * 3600 + 5)


if __name__ == "__main__":
    unittest.main()
