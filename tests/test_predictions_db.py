import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd

from control_plane.predictions_db import _query_metrics_df, fetch_predictions_from_db
from settings import settings


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

    def test_fetch_predictions_uses_fixed_metrics_forecast_table(self) -> None:
        latest_df = pd.DataFrame([{"generated_at": "2026-03-25 10:00:00"}])
        rows_df = pd.DataFrame(
            [{"timestamp": "2026-03-25T10:00:00Z", "predicted": 11.2}]
        )
        with patch(
            "control_plane.predictions_db._load_metrics_forecast_columns",
            return_value={"timestamp", "service", "metric_name", "generated_at", "value"},
        ), patch(
            "control_plane.predictions_db._query_metrics_df",
            side_effect=[latest_df, rows_df],
        ) as query_mock:
            fetch_predictions_from_db(
                service="svc-a",
                metric_name="memory",
                start_time=datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
                end_time=datetime(2026, 3, 25, 10, 30, 0, tzinfo=timezone.utc),
            )

        for call in query_mock.call_args_list:
            self.assertIn("metrics_forecast", call.args[0])

    def test_query_metrics_df_uses_predictions_connection_settings(self) -> None:
        captured = {}

        class _FakeClient:
            def query_df(self, query):
                captured["query"] = query
                return pd.DataFrame([{"ok": 1}])

            @staticmethod
            def close():
                return None

        def _fake_get_client(**kwargs):
            captured["client_kwargs"] = kwargs
            return _FakeClient()

        fake_module = SimpleNamespace(get_client=_fake_get_client)
        with patch.multiple(
            settings,
            CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_HOST="pred-host",
            CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PORT=9440,
            CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_USERNAME="pred-user",
            CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_PASSWORD="pred-pass",
            CONTROL_PLANE_CLICKHOUSE_PREDICTIONS_SECURE=True,
        ), patch.dict("sys.modules", {"clickhouse_connect": fake_module}):
            _query_metrics_df("SELECT 1")

        self.assertEqual(captured["query"], "SELECT 1")
        self.assertEqual(captured["client_kwargs"]["host"], "pred-host")
        self.assertEqual(captured["client_kwargs"]["port"], 9440)
        self.assertEqual(captured["client_kwargs"]["username"], "pred-user")
        self.assertEqual(captured["client_kwargs"]["password"], "pred-pass")
        self.assertEqual(captured["client_kwargs"]["secure"], True)


if __name__ == "__main__":
    unittest.main()
