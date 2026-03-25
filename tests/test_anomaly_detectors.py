import unittest

import pandas as pd

from control_plane.anomaly_detectors import RollingIQRDetector


class TestAnomalyDetectors(unittest.TestCase):
    def test_rolling_iqr_fallback_on_no_overlap_with_predictions(self) -> None:
        actual_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-25T10:00:00Z",
                        "2026-03-25T10:01:00Z",
                        "2026-03-25T10:02:00Z",
                        "2026-03-25T10:03:00Z",
                    ],
                    utc=True,
                ),
                "value": [10.0, 10.0, 10.0, 50.0],
            }
        )
        predictions_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-25T11:00:00Z",
                        "2026-03-25T11:01:00Z",
                        "2026-03-25T11:02:00Z",
                    ],
                    utc=True,
                ),
                "predicted": [11.0, 11.0, 11.0],
            }
        )

        detector = RollingIQRDetector(iqr_window=3, iqr_scale=1.5, min_periods=2, zscore_threshold=3.0)
        result = detector.detect(actual_df, predictions_df, step="1m")

        self.assertFalse(result.merged_df.empty)
        self.assertFalse(result.anomalies_df.empty)
        self.assertIn("is_anomaly", result.anomalies_df.columns)

    def test_rolling_iqr_force_at_least_one_anomaly(self) -> None:
        actual_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-25T10:00:00Z",
                        "2026-03-25T10:01:00Z",
                        "2026-03-25T10:02:00Z",
                    ],
                    utc=True,
                ),
                "value": [10.0, 10.0, 10.0],
            }
        )
        predictions_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-25T10:00:00Z",
                        "2026-03-25T10:01:00Z",
                        "2026-03-25T10:02:00Z",
                    ],
                    utc=True,
                ),
                "predicted": [10.0, 10.0, 10.0],
            }
        )

        detector = RollingIQRDetector(iqr_window=3, iqr_scale=1.5, min_periods=2, zscore_threshold=3.0)
        result = detector.detect(actual_df, predictions_df, step="1m")

        self.assertFalse(result.anomalies_df.empty)
        self.assertGreaterEqual(len(result.anomalies_df), 1)


if __name__ == "__main__":
    unittest.main()
