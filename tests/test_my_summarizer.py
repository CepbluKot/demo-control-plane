import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from my_summarizer import (
    _make_llm_call,
    _build_db_fetch_page,
    _estimate_total_logs,
    _render_logs_query,
    summarize_logs,
)
from settings import settings


class TestMySummarizer(unittest.TestCase):
    def test_make_llm_call_uses_new_llm_functions(self) -> None:
        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=lambda *, message, system_prompt="": f"ok::{message}",
        ):
            llm_call = _make_llm_call()
            out = llm_call("test prompt")
        self.assertEqual(out, "ok::test prompt")

    def test_render_logs_query_substitutes_placeholders(self) -> None:
        query = _render_logs_query(
            query_template=(
                "SELECT timestamp, value FROM logs_table "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit} OFFSET {offset}"
            ),
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            limit=500,
            offset=1000,
            service="svc-a",
        )
        self.assertIn("SELECT timestamp, value FROM logs_table", query)
        self.assertIn("parseDateTimeBestEffort('2026-03-25T10:00:00+00:00')", query)
        self.assertIn("parseDateTimeBestEffort('2026-03-25T11:00:00+00:00')", query)
        self.assertIn("LIMIT 500 OFFSET 1000", query)

    def test_render_logs_query_keeps_unknown_placeholder(self) -> None:
        query = _render_logs_query(
            query_template="SELECT * FROM logs WHERE x = '{unknown}'",
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            limit=100,
            offset=0,
            service="svc-a",
        )
        self.assertIn("{unknown}", query)

    def test_summarize_logs_reads_batches(self) -> None:
        def fake_fetcher(_anomaly, *, fetch_mode, tail_limit):
            _ = (fetch_mode, tail_limit)
            calls = {"count": 0}

            def _fetch_page(*, columns, period_start, period_end, limit, offset):
                _ = (columns, period_start, period_end, limit)
                calls["count"] += 1
                if calls["count"] == 1:
                    return [
                        {"timestamp": "2026-03-25T10:00:00Z", "value": "ok"},
                        {"timestamp": "2026-03-25T10:00:10Z", "value": "failed timeout"},
                    ]
                return []

            return _fetch_page

        config_overrides = {
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY": (
                "SELECT timestamp, value FROM logs_svc_a "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit} OFFSET {offset}"
            ),
            "CONTROL_PLANE_LOGS_PAGE_LIMIT": 1000,
        }

        with patch.multiple(settings, **config_overrides):
            with patch("my_summarizer._build_db_fetch_page", side_effect=fake_fetcher), patch(
                "my_summarizer._estimate_total_logs",
                return_value=2,
            ):
                result = summarize_logs(
                    start_dt=datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
                    end_dt=datetime(2026, 3, 25, 11, 0, 0, tzinfo=timezone.utc),
                    anomaly={"service": "svc-a"},
                )

        self.assertEqual(result["rows_processed"], 2)
        self.assertEqual(result["pages_fetched"], 1)
        self.assertTrue(result["chunk_summaries"])
        self.assertTrue(result["map_batches"])
        self.assertIn("rows", result["map_batches"][0])
        self.assertIn("batch_period_start", result["map_batches"][0])
        self.assertIn("batch_period_end", result["map_batches"][0])
        self.assertIn("Сервис: svc-a", result["summary"])

    def test_summarize_logs_emits_live_progress_events(self) -> None:
        events = []

        def fake_fetcher(_anomaly, *, fetch_mode, tail_limit):
            _ = (fetch_mode, tail_limit)
            calls = {"count": 0}

            def _fetch_page(*, columns, period_start, period_end, limit, offset):
                _ = (columns, period_start, period_end, limit, offset)
                calls["count"] += 1
                if calls["count"] == 1:
                    return [
                        {"timestamp": "2026-03-25T10:00:00Z", "value": "ok"},
                        {"timestamp": "2026-03-25T10:00:10Z", "value": "failed timeout"},
                    ]
                return []

            return _fetch_page

        config_overrides = {
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY": (
                "SELECT timestamp, value FROM logs_svc_a "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit} OFFSET {offset}"
            ),
            "CONTROL_PLANE_LOGS_PAGE_LIMIT": 1000,
        }

        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer._build_db_fetch_page",
            side_effect=fake_fetcher,
        ), patch(
            "my_summarizer._estimate_total_logs",
            return_value=2,
        ):
            summarize_logs(
                start_dt=datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
                end_dt=datetime(2026, 3, 25, 11, 0, 0, tzinfo=timezone.utc),
                anomaly={"service": "svc-a"},
                on_progress=lambda event, payload: events.append((event, payload)),
            )

        event_names = [name for name, _ in events]
        self.assertIn("map_start", event_names)
        self.assertIn("map_batch", event_names)
        self.assertIn("map_done", event_names)
        self.assertIn("reduce_start", event_names)
        self.assertIn("reduce_done", event_names)
        map_batch_payload = next(payload for name, payload in events if name == "map_batch")
        self.assertIn("batch_logs", map_batch_payload)
        self.assertIn("rows_processed", map_batch_payload)
        self.assertIn("batch_period_start", map_batch_payload)
        self.assertIn("batch_period_end", map_batch_payload)

    def test_build_db_fetch_page_without_offset_placeholder_single_shot(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY": (
                "SELECT timestamp, value FROM logs_svc_a "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit}"
            ),
            "CONTROL_PLANE_LOGS_PAGE_LIMIT": 1,
        }
        calls = {"count": 0}

        def _fake_query_df(_query: str):
            calls["count"] += 1
            return pd.DataFrame(
                [{"timestamp": "2026-03-25T10:00:00Z", "value": "one"}]
            )

        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer._query_logs_df",
            side_effect=_fake_query_df,
        ):
            fetch_page = _build_db_fetch_page(
                {"service": "svc-a"},
                fetch_mode="time_window",
                tail_limit=1000,
            )
            first_page = fetch_page(
                columns=["timestamp", "value"],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                limit=1,
                offset=0,
            )
            second_page = fetch_page(
                columns=["timestamp", "value"],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                limit=1,
                offset=1,
            )

        self.assertEqual(len(first_page), 1)
        self.assertEqual(second_page, [])
        self.assertEqual(calls["count"], 1)

    def test_build_db_fetch_page_tail_mode_uses_latest_n_before_anomaly(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY": (
                "SELECT timestamp, value FROM logs_svc_a "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit} OFFSET {offset}"
            ),
            "CONTROL_PLANE_LOGS_FETCH_MODE": "tail_n_logs",
            "CONTROL_PLANE_LOGS_TAIL_LIMIT": 1000,
        }
        captured = {"query": ""}

        def _fake_query_df(query: str):
            captured["query"] = query
            return pd.DataFrame(
                [{"timestamp": "2026-03-25T10:00:00Z", "value": "one"}]
            )

        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer._query_logs_df",
            side_effect=_fake_query_df,
        ):
            fetch_page = _build_db_fetch_page(
                {"service": "svc-a"},
                fetch_mode="tail_n_logs",
                tail_limit=1000,
            )
            page = fetch_page(
                columns=["timestamp", "value"],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                limit=100,
                offset=0,
            )

        self.assertEqual(len(page), 1)
        self.assertIn("ORDER BY timestamp DESC LIMIT 1000", captured["query"])
        self.assertIn("ORDER BY timestamp ASC LIMIT 100 OFFSET 0", captured["query"])
        self.assertIn(
            "parseDateTimeBestEffort('1970-01-01T00:00:00+00:00')",
            captured["query"],
        )

    def test_estimate_total_logs_tail_mode_caps_by_tail_limit(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY": (
                "SELECT timestamp, value FROM logs_svc_a "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit} OFFSET {offset}"
            ),
        }

        def _fake_query_df(_query: str):
            return pd.DataFrame([{"total_rows": 5000}])

        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer._query_logs_df",
            side_effect=_fake_query_df,
        ):
            total = _estimate_total_logs(
                anomaly={"service": "svc-a"},
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                page_limit=1000,
                fetch_mode="tail_n_logs",
                tail_limit=1000,
            )

        self.assertEqual(total, 1000)

    def test_summarize_logs_requires_service_in_anomaly(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY": (
                "SELECT timestamp, value FROM logs_svc_a "
                "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
                "AND timestamp < parseDateTimeBestEffort('{period_end}') "
                "ORDER BY timestamp LIMIT {limit} OFFSET {offset}"
            ),
            "CONTROL_PLANE_LOGS_PAGE_LIMIT": 1000,
        }
        with patch.multiple(settings, **config_overrides):
            with self.assertRaises(ValueError):
                summarize_logs(
                    start_dt=datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
                    end_dt=datetime(2026, 3, 25, 11, 0, 0, tzinfo=timezone.utc),
                    anomaly={},
                )


if __name__ == "__main__":
    unittest.main()
