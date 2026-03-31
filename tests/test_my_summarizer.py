import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from my_summarizer import (
    PeriodLogSummarizer,
    _make_llm_call,
    _build_db_fetch_page,
    _estimate_total_logs,
    _render_logs_query,
    build_cross_source_reduce_prompt,
    communicate_with_llm,
    regenerate_reduce_summary_from_map_summaries,
    summarize_logs,
)
from settings import settings


class TestMySummarizer(unittest.TestCase):
    def test_communicate_with_llm_continues_when_finish_reason_length(self) -> None:
        class _FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self._payload

        calls = []

        def _fake_post(_url, *, json=None, headers=None, timeout=None):
            calls.append({"json": json, "headers": headers, "timeout": timeout})
            if len(calls) == 1:
                return _FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {"content": "part-1 "},
                                "finish_reason": "length",
                            }
                        ]
                    }
                )
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "part-2"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        config_overrides = {
            "OPENAI_API_BASE_DB": "https://example.test/v1",
            "OPENAI_API_KEY_DB": "secret",
            "LLM_MODEL_ID": "demo-model",
            "CONTROL_PLANE_LLM_MAX_TOKENS": 0,
            "CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH": True,
            "CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS": 4,
        }
        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer.requests.post",
            side_effect=_fake_post,
        ):
            out = communicate_with_llm(
                message="test prompt",
                system_prompt="system",
                timeout=15.0,
            )

        self.assertEqual(out, "part-1 part-2")
        self.assertEqual(len(calls), 2)
        second_messages = calls[1]["json"]["messages"]
        self.assertIn(
            "Продолжи строго с того места",
            str(second_messages[-1]["content"]),
        )

    def test_communicate_with_llm_passes_max_tokens_when_configured(self) -> None:
        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                }

        captured = {}

        def _fake_post(_url, *, json=None, headers=None, timeout=None):
            captured["json"] = json
            captured["timeout"] = timeout
            captured["headers"] = headers
            return _FakeResponse()

        config_overrides = {
            "OPENAI_API_BASE_DB": "https://example.test/v1",
            "OPENAI_API_KEY_DB": "secret",
            "LLM_MODEL_ID": "demo-model",
            "CONTROL_PLANE_LLM_MAX_TOKENS": 2048,
            "CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH": True,
            "CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS": 4,
        }
        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer.requests.post",
            side_effect=_fake_post,
        ):
            out = communicate_with_llm(
                message="test prompt",
                system_prompt="",
                timeout=12.0,
            )

        self.assertEqual(out, "ok")
        self.assertEqual(int(captured["json"]["max_tokens"]), 2048)
        self.assertEqual(float(captured["timeout"]), 12.0)

    def test_make_llm_call_uses_new_llm_functions(self) -> None:
        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=lambda *, message, system_prompt="", timeout=60.0: f"ok::{message}",
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
        def fake_fetcher(_anomaly, *, fetch_mode, tail_limit, on_error=None):
            _ = (fetch_mode, tail_limit, on_error)
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
        self.assertIn("chunk_summaries", result)
        self.assertIsInstance(result["chunk_summaries"], list)
        self.assertIn("map_batches", result)
        self.assertIsInstance(result["map_batches"], list)
        self.assertIn("Сервис: svc-a", result["summary"])

    def test_summarize_logs_emits_live_progress_events(self) -> None:
        events = []

        def fake_fetcher(_anomaly, *, fetch_mode, tail_limit, on_error=None):
            _ = (fetch_mode, tail_limit, on_error)
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

    def test_build_db_fetch_page_without_offset_placeholder_uses_auto_paging(self) -> None:
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

        def _fake_query_df(query: str):
            calls["count"] += 1
            if "OFFSET 0" in query:
                return pd.DataFrame(
                    [{"timestamp": "2026-03-25T10:00:00Z", "value": "one"}]
                )
            return pd.DataFrame(columns=["timestamp", "value"])

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
        self.assertEqual(calls["count"], 2)

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

    def test_cross_source_reduce_prompt_uses_custom_template(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_LLM_CROSS_SOURCE_REDUCE_PROMPT_TEMPLATE": (
                "CUSTOM CROSS | period={period_start}->{period_end} | "
                "sources={source_names} | body={source_summaries_text}"
            ),
        }
        with patch.multiple(settings, **config_overrides):
            prompt = build_cross_source_reduce_prompt(
                summaries_by_source={"query_1": "summary one"},
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
            )
        self.assertIn("CUSTOM CROSS", prompt)
        self.assertIn("query_1", prompt)
        self.assertIn("summary one", prompt)

    def test_map_prompt_uses_custom_template(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_LLM_MAP_PROMPT_TEMPLATE": (
                "CUSTOM MAP | period={period_start}->{period_end} | "
                "type={data_type} | time_col={time_column} | rows={rows_count}"
            ),
        }
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
        )
        with patch.multiple(settings, **config_overrides):
            prompt = summarizer._build_chunk_prompt(
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
                columns=["start_time", "end_time", "cnt", "message"],
                rows=[
                    {
                        "start_time": "2026-03-18T00:01:00Z",
                        "end_time": "2026-03-18T00:02:00Z",
                        "cnt": 4,
                    }
                ],
            )
        self.assertIn("CUSTOM MAP", prompt)
        self.assertIn("type=aggregated", prompt)
        self.assertIn("ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЛОКАЛЬНАЯ ЦЕПОЧКА СОБЫТИЙ БАТЧА", prompt)
        self.assertIn(
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ",
            prompt,
        )

    def test_truncate_with_zero_limit_keeps_text(self) -> None:
        text = "a" * 5000
        out = PeriodLogSummarizer._truncate(text, 0)
        self.assertEqual(out, text)

    def test_build_freeform_prompt_default_includes_map_summaries(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
        )
        prompt = summarizer._build_freeform_prompt(
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            structured_summary="structured summary",
            map_summaries=["map one", "map two"],
        )
        self.assertIn("MAP SUMMARY ПО БАТЧАМ ЛОГОВ", prompt)
        self.assertIn("[MAP SUMMARY #1]", prompt)
        self.assertIn("map one", prompt)
        self.assertIn("map two", prompt)

    def test_build_freeform_prompt_custom_template_uses_map_summaries_vars(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
        )
        config_overrides = {
            "CONTROL_PLANE_LLM_FREEFORM_PROMPT_TEMPLATE": (
                "CUSTOM FREEFORM | maps={map_summaries_text} | final={structured_summary}"
            ),
        }
        with patch.multiple(settings, **config_overrides):
            prompt = summarizer._build_freeform_prompt(
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
                structured_summary="structured summary",
                map_summaries=["map one"],
            )
        self.assertIn("CUSTOM FREEFORM", prompt)
        self.assertIn("maps=[MAP SUMMARY #1]", prompt)
        self.assertIn("final=structured summary", prompt)

    def test_regenerate_reduce_summary_from_map_summaries(self) -> None:
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=["batch-1 summary", "batch-2 summary"],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda _prompt: "MERGED SUMMARY",
        )
        self.assertIn("MERGED SUMMARY", out)

    def test_regenerate_reduce_summary_does_not_truncate_when_limits_zero(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS": 0,
            "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS": 0,
            "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS": 0,
        }
        long_result = "MERGED " + ("X" * 10000)
        with patch.multiple(settings, **config_overrides):
            out = regenerate_reduce_summary_from_map_summaries(
                map_summaries=["batch-1 summary", "batch-2 summary"],
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
                llm_call=lambda _prompt: long_result,
            )
        self.assertEqual(out, long_result)

    def test_regenerate_reduce_summary_from_map_summaries_empty(self) -> None:
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=[],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda _prompt: "SHOULD NOT BE USED",
        )
        self.assertIn("Нет map-summary", out)

    def test_regenerate_reduce_summary_from_map_summaries_emits_progress(self) -> None:
        events = []
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=["batch-1 summary", "batch-2 summary"],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda _prompt: "MERGED SUMMARY",
            on_progress=lambda event, payload: events.append((event, payload)),
        )
        self.assertIn("MERGED SUMMARY", out)
        event_names = [name for name, _ in events]
        self.assertIn("reduce_group_start", event_names)


if __name__ == "__main__":
    unittest.main()
