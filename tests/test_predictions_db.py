import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from control_plane.predictions_db import fetch_predictions_from_db


class TestPredictionsDb(unittest.TestCase):
    def test_fetch_predictions_reads_latest_generated_batch(self) -> None:
        latest_df = pd.DataFrame([{"generated_at": "2026-03-25 10:00:00"}])
        rows_df = pd.DataFrame(
            [
                {"timestamp": "2026-03-25T10:00:00Z", "predicted": 11.2},
                {"timestamp": "2026-03-25T10:01:00Z", "predicted": 11.4},
            ]
        )
        available_columns = {
            "timestamp",
            "service",
            "metric_name",
            "generated_at",
            "value",
            "prediction_kind",
            "forecast_type",
        }
        with patch(
            "control_plane.predictions_db._load_metrics_forecast_columns",
            return_value=available_columns,
        ), patch(
            "control_plane.predictions_db._query_metrics_df",
            side_effect=[latest_df, rows_df],
        ) as query_mock:
            out = fetch_predictions_from_db(
                service="svc-a",
                metric_name="memory",
                start_time=datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 3, 25, 10, 30, 0, tzinfo=timezone.utc),
                forecast_type="short",
                prediction_kind="forecast",
            )

        self.assertEqual(len(out), 2)
        self.assertEqual(list(out.columns), ["timestamp", "predicted"])
        self.assertEqual(query_mock.call_count, 2)
        latest_query = query_mock.call_args_list[0].args[0]
        self.assertIn("SELECT max(generated_at)", latest_query)
        rows_query = query_mock.call_args_list[1].args[0]
        self.assertIn("SELECT timestamp, value AS predicted", rows_query)

    def test_fetch_predictions_raises_when_no_latest_generated_at(self) -> None:
        latest_df = pd.DataFrame([{"generated_at": None}])
        with patch(
            "control_plane.predictions_db._load_metrics_forecast_columns",
            return_value={"timestamp", "service", "metric_name", "generated_at", "value"},
        ), patch(
            "control_plane.predictions_db._query_metrics_df",
            return_value=latest_df,
        ):
            with self.assertRaises(ValueError):
                fetch_predictions_from_db(
                    service="svc-a",
                    metric_name="memory",
                    start_time=datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
                    end_time=datetime(2026, 3, 25, 10, 30, 0, tzinfo=timezone.utc),
                )


if __name__ == "__main__":
    unittest.main()
