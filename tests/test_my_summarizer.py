import unittest
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pandas as pd
import requests

from my_summarizer import (
    MapBatchSummaryModel,
    PeriodLogSummarizer,
    SummarizerConfig,
    _is_instructor_empty_choices_exception,
    _is_read_timeout_exception,
    _is_transient_gateway_exception,
    _make_llm_call,
    _build_db_fetch_page,
    _estimate_total_logs,
    _render_logs_query,
    build_cross_source_reduce_prompt,
    communicate_with_llm,
    generate_final_reports_with_instructor,
    regenerate_reduce_summary_from_map_summaries,
    summarize_logs,
)
from report_schema import (
    AlertExplanation,
    AlertsSection,
    AnalysisLimitations,
    CausalChain,
    CausalChainsSection,
    ChronologyEvent,
    ChronologySection,
    CoverageSection,
    FreeformReport,
    ImpactAssessment,
    ImpactSection,
    IncidentReport,
    LimitationsSection,
    MetricsSection,
    ReportPartAnalytical,
    ReportPartDescriptive,
    SummarySection,
    ReportSummary,
    DataCoverage,
)
from schemas import IncidentSummary
from settings import settings


class TestMySummarizer(unittest.TestCase):
    @staticmethod
    def _structured_summary_dict(
        batch_id: str,
        *,
        timestamp: str = "2026-03-18T00:00:00+00:00",
        total_log_entries: int = 1,
        include_timeline: bool = True,
        is_empty: bool = False,
    ) -> dict[str, Any]:
        timeline = []
        if include_timeline and not is_empty:
            event_id = f"evt-{batch_id.replace(':', '-').replace('_', '-')}"
            timeline = [
                {
                    "id": event_id,
                    "timestamp": timestamp,
                    "source": "svc-a",
                    "description": f"Событие для {batch_id}",
                    "severity": "high",
                    "importance": 0.9,
                    "evidence_type": "FACT",
                    "evidence_quote": "timeout detected",
                    "tags": ["timeout"],
                }
            ]
        return {
            "context": {
                "batch_id": batch_id,
                "time_range_start": timestamp,
                "time_range_end": timestamp,
                "total_log_entries": int(total_log_entries),
                "source_query": [],
                "source_services": ["svc-a"],
            },
            "timeline": timeline,
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {
                "is_empty": bool(is_empty),
                "noise_ratio": 0.1 if timeline else 1.0,
                "notes": "",
            },
            "preliminary_recommendations": [],
        }

    @classmethod
    def _structured_summary_json(cls, batch_id: str, **kwargs: Any) -> str:
        return json.dumps(cls._structured_summary_dict(batch_id, **kwargs), ensure_ascii=False)

    @staticmethod
    def _extract_total_rows_from_map_prompt(prompt: str) -> int:
        marker = "Всего строк логов:"
        if marker not in prompt:
            return 0
        tail = prompt.split(marker, 1)[1].strip().splitlines()[0]
        try:
            return int(tail)
        except Exception:
            return 0

    def test_make_llm_call_raises_when_env_missing(self) -> None:
        with patch("my_summarizer.has_required_env", return_value=False):
            with self.assertRaises(RuntimeError):
                _make_llm_call()

    def test_make_llm_call_uses_20min_default_timeout(self) -> None:
        captured: list[float] = []

        def _fake_communicate(*, message, system_prompt="", timeout=0.0):
            _ = (message, system_prompt)
            captured.append(float(timeout))
            return "ok"

        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=_fake_communicate,
        ):
            llm_call = _make_llm_call(max_retries=0, retry_delay=0.0)
            out = llm_call("test prompt")

        self.assertEqual(out, "ok")
        self.assertEqual(captured, [1200.0])

    def test_make_llm_call_raises_after_retry_exhaustion(self) -> None:
        retries = []

        def _raise_conn_error(*, message, system_prompt="", timeout=60.0):
            _ = (message, system_prompt, timeout)
            raise requests.exceptions.ConnectionError("network down")

        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=_raise_conn_error,
        ):
            llm_call = _make_llm_call(
                max_retries=1,
                retry_delay=0.0,
                on_retry=lambda *_: retries.append("retry"),
            )
            with self.assertRaises(requests.exceptions.ConnectionError):
                llm_call("test prompt")

        self.assertEqual(retries, ["retry"])

    def test_make_llm_call_fail_open_returns_empty_after_retry_exhaustion(self) -> None:
        retries = []

        def _raise_conn_error(*, message, system_prompt="", timeout=60.0):
            _ = (message, system_prompt, timeout)
            raise requests.exceptions.ConnectionError("network down")

        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=_raise_conn_error,
        ):
            llm_call = _make_llm_call(
                max_retries=1,
                retry_delay=0.0,
                on_retry=lambda *_: retries.append("retry"),
                fail_open_return_empty=True,
            )
            out = llm_call("test prompt")

        self.assertEqual(out, "")
        self.assertEqual(retries, ["retry"])

    def test_communicate_with_llm_retries_without_max_tokens_on_400(self) -> None:
        class _FakeResponse:
            def __init__(self, status_code: int, payload: dict):
                self.status_code = status_code
                self._payload = payload
                self.text = str(payload)

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.exceptions.HTTPError(
                        f"{self.status_code} Client Error",
                        response=self,
                    )

            def json(self):
                return self._payload

        calls = []

        def _fake_post(_url, *, json=None, headers=None, timeout=None):
            calls.append({"json": json, "headers": headers, "timeout": timeout})
            if len(calls) == 1:
                return _FakeResponse(400, {"error": {"message": "unsupported max_tokens"}})
            return _FakeResponse(
                200,
                {
                    "choices": [
                        {"message": {"content": "ok after fallback"}, "finish_reason": "stop"}
                    ]
                },
            )

        config_overrides = {
            "OPENAI_API_BASE_DB": "https://example.test/v1",
            "OPENAI_API_KEY_DB": "secret",
            "LLM_MODEL_ID": "demo-model",
            "CONTROL_PLANE_LLM_MAX_TOKENS": 16384,
            "CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH": True,
            "CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS": 2,
        }
        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer.requests.post",
            side_effect=_fake_post,
        ):
            out = communicate_with_llm("prompt", timeout=10.0)

        self.assertEqual(out, "ok after fallback")
        self.assertEqual(len(calls), 2)
        self.assertIn("max_tokens", calls[0]["json"])
        self.assertNotIn("max_tokens", calls[1]["json"])

    def test_communicate_with_llm_includes_exact_response_body_in_http_error(self) -> None:
        class _FakeResponse:
            def __init__(self):
                self.status_code = 400
                self.url = "https://example.test/v1/chat/completions"
                self.text = '{"error":{"message":"context too large","code":"bad_request"}}'

            def raise_for_status(self):
                raise requests.exceptions.HTTPError(
                    "400 Client Error: Bad Request for url: https://example.test/v1/chat/completions",
                    response=self,
                )

            def json(self):
                return {"error": {"message": "context too large", "code": "bad_request"}}

        config_overrides = {
            "OPENAI_API_BASE_DB": "https://example.test/v1",
            "OPENAI_API_KEY_DB": "secret",
            "LLM_MODEL_ID": "demo-model",
            "CONTROL_PLANE_LLM_MAX_TOKENS": 0,
            "CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH": True,
            "CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS": 2,
        }
        with patch.multiple(settings, **config_overrides), patch(
            "my_summarizer.requests.post",
            return_value=_FakeResponse(),
        ):
            with self.assertRaises(requests.exceptions.HTTPError) as cm:
                communicate_with_llm("prompt", timeout=10.0)

        msg = str(cm.exception)
        self.assertIn("400 Client Error: Bad Request", msg)
        self.assertIn("RESPONSE_BODY:", msg)
        self.assertIn('"context too large"', msg)

    def test_make_llm_call_does_not_retry_non_retryable_400(self) -> None:
        class _Resp:
            status_code = 400

        retries = []

        def _raise_bad_request(*, message, system_prompt="", timeout=60.0):
            _ = (message, system_prompt, timeout)
            raise requests.exceptions.HTTPError(
                "400 Client Error: Bad Request",
                response=_Resp(),
            )

        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=_raise_bad_request,
        ):
            llm_call = _make_llm_call(
                max_retries=-1,
                retry_delay=0.0,
                on_retry=lambda *_: retries.append("retry"),
            )
            with self.assertRaises(requests.exceptions.HTTPError):
                llm_call("test prompt")

        self.assertEqual(retries, [])

    def test_make_llm_call_fail_open_returns_empty_on_non_retryable_400(self) -> None:
        class _Resp:
            status_code = 400

        retries = []

        def _raise_bad_request(*, message, system_prompt="", timeout=60.0):
            _ = (message, system_prompt, timeout)
            raise requests.exceptions.HTTPError(
                "400 Client Error: Bad Request",
                response=_Resp(),
            )

        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=_raise_bad_request,
        ):
            llm_call = _make_llm_call(
                max_retries=-1,
                retry_delay=0.0,
                on_retry=lambda *_: retries.append("retry"),
                fail_open_return_empty=True,
            )
            out = llm_call("test prompt")

        self.assertEqual(out, "")
        self.assertEqual(retries, [])

    def test_make_llm_call_doubles_timeout_on_read_timeout(self) -> None:
        observed_timeouts: list[float] = []
        calls = {"count": 0}

        def _fake_communicate(*, message, system_prompt="", timeout=60.0):
            _ = (message, system_prompt)
            observed_timeouts.append(float(timeout))
            calls["count"] += 1
            if calls["count"] < 3:
                raise requests.exceptions.ReadTimeout("read timeout")
            return "ok"

        with patch("my_summarizer.has_required_env", return_value=True), patch(
            "my_summarizer.communicate_with_llm",
            side_effect=_fake_communicate,
        ), patch(
            "my_summarizer.time.sleep",
            return_value=None,
        ):
            llm_call = _make_llm_call(max_retries=0, retry_delay=0.0, llm_timeout=5.0)
            out = llm_call("test prompt")

        self.assertEqual(out, "ok")
        self.assertEqual(observed_timeouts, [5.0, 10.0, 20.0])

    def test_summarizer_auto_shrinks_batch_on_400_bad_request(self) -> None:
        calls = {"map_large": 0, "map_small": 0}

        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [
                {"timestamp": f"2026-03-25T10:00:0{i}Z", "message": f"log {i}"}
                for i in range(8)
            ]

        def _fake_llm(prompt: str) -> str:
            if "Summary для объединения" in prompt:
                return self._structured_summary_json("reduce-final", total_log_entries=8)
            if "Логи для батча" in prompt:
                rows_count = self._extract_total_rows_from_map_prompt(prompt)
                if rows_count > 2:
                    calls["map_large"] += 1

                    class _Resp:
                        status_code = 400

                    raise requests.exceptions.HTTPError(
                        "400 Client Error: Bad Request for url",
                        response=_Resp(),
                    )
                calls["map_small"] += 1
                return self._structured_summary_json(
                    f"map-small-{calls['map_small']}",
                    total_log_entries=max(rows_count, 1),
                )
            return "GENERIC_OK"

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=_fake_llm,
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=8,
                min_llm_chunk_rows=2,
                auto_shrink_on_400=True,
                max_shrink_rounds=6,
                keep_map_summaries_in_result=True,
            ),
        )
        result = summarizer.summarize_period(
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            columns=["timestamp", "message"],
        )
        self.assertIn('"batch_id": "reduce-final"', str(result.summary))
        self.assertGreater(calls["map_large"], 0)
        self.assertGreater(calls["map_small"], 0)
        parsed_map = [json.loads(item) for item in result.map_summaries if str(item).strip().startswith("{")]
        self.assertTrue(parsed_map)
        self.assertTrue(
            any(int(entry.get("context", {}).get("total_log_entries", 0)) == 2 for entry in parsed_map)
        )

    def test_summarizer_raises_on_non_400_map_error_without_fallback(self) -> None:
        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [{"timestamp": "2026-03-25T10:00:00Z", "message": "log"}]

        def _fake_llm(_prompt: str) -> str:
            raise requests.exceptions.ConnectionError("llm transport error")

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=_fake_llm,
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=8,
                min_llm_chunk_rows=2,
                auto_shrink_on_400=True,
                max_shrink_rounds=6,
                keep_map_summaries_in_result=True,
            ),
        )
        with self.assertRaises(requests.exceptions.ConnectionError):
            summarizer.summarize_period(
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                columns=["timestamp", "message"],
            )

    def test_communicate_with_llm_continues_when_finish_reason_length(self) -> None:
        class _FakeResponse:
            def __init__(self, payload):
                self.status_code = 200
                self._payload = payload
                self.text = str(payload)

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
            status_code = 200
            text = "ok"

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
                        {
                            "timestamp": "2026-03-25T10:00:00Z",
                            "value": "ok",
                            "__cp_db_query": "SELECT * FROM logs LIMIT 100 OFFSET 0",
                        },
                        {
                            "timestamp": "2026-03-25T10:00:10Z",
                            "value": "failed timeout",
                            "__cp_db_query": "SELECT * FROM logs LIMIT 100 OFFSET 0",
                        },
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
            ), patch(
                "my_summarizer._make_llm_call",
                return_value=lambda _prompt: "LLM_OK",
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
                        {
                            "timestamp": "2026-03-25T10:00:00Z",
                            "value": "ok",
                            "__cp_db_query": "SELECT * FROM logs LIMIT 100 OFFSET 0",
                        },
                        {
                            "timestamp": "2026-03-25T10:00:10Z",
                            "value": "failed timeout",
                            "__cp_db_query": "SELECT * FROM logs LIMIT 100 OFFSET 0",
                        },
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
        ), patch(
            "my_summarizer._make_llm_call",
            return_value=lambda _prompt: "LLM_OK",
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
        self.assertEqual(
            map_batch_payload.get("batch_queries"),
            ["SELECT * FROM logs LIMIT 100 OFFSET 0"],
        )

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

    def test_map_prompt_uses_builtin_template_even_if_legacy_template_set(self) -> None:
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
        self.assertNotIn("CUSTOM MAP", prompt)
        self.assertIn("Формат ответа (строго)", prompt)
        self.assertIn("Верни ТОЛЬКО один валидный JSON-объект.", prompt)
        self.assertIn("Всего строк логов: 1", prompt)

    def test_map_prompt_requires_strict_json_output(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_new_algorithm=True),
        )
        prompt = summarizer._build_chunk_prompt(
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            columns=["timestamp", "message"],
            rows=[
                {
                    "timestamp": "2026-03-18T00:00:01Z",
                    "message": "error timeout to redis",
                }
            ],
            batch_number=1,
            total_batches=3,
        )
        self.assertIn("Формат ответа (строго)", prompt)
        self.assertIn("Верни ТОЛЬКО один валидный JSON-объект.", prompt)
        self.assertIn("Top-level keys", prompt)

    def test_truncate_with_zero_limit_keeps_text(self) -> None:
        text = "a" * 5000
        out = PeriodLogSummarizer._truncate(text, 0)
        self.assertEqual(out, text)

    def test_extract_batch_queries_from_rows_deduplicates_queries(self) -> None:
        rows = [
            {"timestamp": "2026-03-18T00:00:00Z", "__cp_db_query": "SELECT 1"},
            {"timestamp": "2026-03-18T00:01:00Z", "__cp_db_query": "SELECT 1"},
            {"timestamp": "2026-03-18T00:02:00Z", "__cp_db_query": "SELECT 2"},
            {"timestamp": "2026-03-18T00:03:00Z"},
        ]
        queries = PeriodLogSummarizer._extract_batch_queries_from_rows(rows)
        self.assertEqual(queries, ["SELECT 1", "SELECT 2"])

    def test_build_freeform_prompt_default_includes_map_summaries(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
        )
        summarizer.prompt_context = {
            "incident_description": "INCIDENT TEXT FROM UI",
            "alerts_list": "ALERT LIST FROM UI",
        }
        prompt = summarizer._build_freeform_prompt(
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            structured_summary="structured summary",
            map_summaries=["map one", "map two"],
        )
        self.assertIn("MAP САММАРИ ПО БАТЧАМ ЛОГОВ", prompt)
        self.assertIn("[MAP САММАРИ #1]", prompt)
        self.assertIn("map one", prompt)
        self.assertIn("map two", prompt)
        self.assertIn("ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)", prompt)

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
        self.assertIn("maps=[MAP САММАРИ #1]", prompt)
        self.assertIn("final=structured summary", prompt)

    def test_regenerate_reduce_summary_from_map_summaries(self) -> None:
        merged_json = self._structured_summary_json("reduce-merged", total_log_entries=2)
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=[
                self._structured_summary_json("map-1", total_log_entries=1),
                self._structured_summary_json("map-2", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda _prompt: merged_json,
        )
        self.assertIn('"batch_id": "reduce-merged"', out)

    def test_regenerate_reduce_summary_does_not_truncate_when_limits_zero(self) -> None:
        config_overrides = {
            "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS": 0,
            "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS": 0,
            "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS": 0,
        }
        long_text = "X" * 10000
        long_payload = self._structured_summary_dict("map-long", total_log_entries=1)
        long_payload["timeline"][0]["description"] = long_text
        long_json = json.dumps(long_payload, ensure_ascii=False)
        with patch.multiple(settings, **config_overrides):
            out = regenerate_reduce_summary_from_map_summaries(
                map_summaries=[long_json],
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
                llm_call=lambda _prompt: self._structured_summary_json(
                    "unused",
                    total_log_entries=1,
                ),
            )
        self.assertIn(long_text, out)

    def test_regenerate_reduce_summary_from_map_summaries_empty(self) -> None:
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=[],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda _prompt: "SHOULD NOT BE USED",
        )
        self.assertIn("Нет map-summary", out)

    def test_regenerate_reduce_summary_filters_llm_error_stub_items(self) -> None:
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=[
                "[LLM ERROR]\n\nОШИБКА: 400 Client Error",
                "валидный map summary",
            ],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda prompt: prompt,
        )
        self.assertEqual(out, "валидный map summary")

    def test_regenerate_reduce_summary_from_map_summaries_emits_progress(self) -> None:
        events = []
        merged_json = self._structured_summary_json("reduce-merged", total_log_entries=2)
        out = regenerate_reduce_summary_from_map_summaries(
            map_summaries=[
                self._structured_summary_json("map-1", total_log_entries=1),
                self._structured_summary_json("map-2", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            llm_call=lambda _prompt: merged_json,
            on_progress=lambda event, payload: events.append((event, payload)),
        )
        self.assertIn('"batch_id": "reduce-merged"', out)
        event_names = [name for name, _ in events]
        self.assertIn("reduce_group_start", event_names)

    def test_reduce_summaries_handles_400_with_structured_split_merge(self) -> None:
        class _Resp:
            status_code = 400

        calls = {"n": 0}
        merged_json = self._structured_summary_json("reduce-400-ok", total_log_entries=2)

        def _fake_llm(_prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.HTTPError(
                    "400 Client Error: Bad Request for url",
                    response=_Resp(),
                )
            return merged_json

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(use_instructor=False),
        )
        with patch.object(
            summarizer,
            "_compress_summary_on_overflow",
            side_effect=lambda summary, **_: (summary, 0),
        ):
            final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
                chunk_summaries=[
                    self._structured_summary_json("map-a", total_log_entries=1),
                    self._structured_summary_json("map-b", total_log_entries=1),
                ],
                period_start="2026-03-18T00:00:00+00:00",
                period_end="2026-03-18T01:00:00+00:00",
            )
        self.assertIn('"batch_id": "reduce-400-ok"', final_summary)
        self.assertGreaterEqual(llm_calls, 1)
        self.assertGreaterEqual(reduce_rounds, 1)

    def test_reduce_summaries_shrinks_group_size_on_400_inside_round(self) -> None:
        class _Resp:
            status_code = 400

        merged_json = self._structured_summary_json("reduce-shrink-ok", total_log_entries=3)

        def _fake_llm(prompt: str) -> str:
            # Fail when reducing 3 summaries in a single call, succeed for smaller groups.
            if prompt.count('"batch_id"') >= 3 and "Summary для объединения" in prompt:
                raise requests.exceptions.HTTPError(
                    "400 Client Error: Bad Request for url",
                    response=_Resp(),
                )
            return merged_json

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(use_instructor=False, reduce_group_size=3),
        )
        with patch.object(
            summarizer,
            "_compress_summary_on_overflow",
            side_effect=lambda summary, **_: (summary, 0),
        ):
            final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
                chunk_summaries=[
                    self._structured_summary_json("map-a", total_log_entries=1),
                    self._structured_summary_json("map-b", total_log_entries=1),
                    self._structured_summary_json("map-c", total_log_entries=1),
                ],
                period_start="2026-03-18T00:00:00+00:00",
                period_end="2026-03-18T01:00:00+00:00",
            )
        self.assertTrue(final_summary)
        self.assertGreaterEqual(llm_calls, 1)
        self.assertGreaterEqual(reduce_rounds, 1)

    def test_summarize_period_does_not_fail_on_reduce_400(self) -> None:
        class _Resp:
            status_code = 400

        calls = {"reduce_errors": 0}
        progress_events = []

        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [
                {"timestamp": "2026-03-25T10:00:00Z", "message": "m1"},
                {"timestamp": "2026-03-25T10:00:01Z", "message": "m2"},
                {"timestamp": "2026-03-25T10:00:02Z", "message": "m3"},
                {"timestamp": "2026-03-25T10:00:03Z", "message": "m4"},
            ]

        def _fake_llm(prompt: str) -> str:
            if "Summary для объединения" in prompt:
                if calls["reduce_errors"] == 0:
                    calls["reduce_errors"] += 1
                    raise requests.exceptions.HTTPError(
                        "400 Client Error: Bad Request for url",
                        response=_Resp(),
                    )
                return self._structured_summary_json("reduce-ok-after-400", total_log_entries=4)
            if "Логи для батча" in prompt:
                rows_count = self._extract_total_rows_from_map_prompt(prompt)
                return self._structured_summary_json(
                    f"map-batch-{rows_count}",
                    total_log_entries=max(rows_count, 1),
                )
            return "FREEFORM_OK"

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=_fake_llm,
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=2,
                use_instructor=False,
                keep_map_summaries_in_result=True,
            ),
            on_progress=lambda event, payload: progress_events.append((event, payload)),
        )
        with patch.object(
            summarizer,
            "_compress_summary_on_overflow",
            side_effect=lambda summary, **_: (summary, 0),
        ):
            result = summarizer.summarize_period(
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                columns=["timestamp", "message"],
            )

        self.assertIn('"batch_id": "reduce-ok-after-400"', result.summary)
        self.assertGreaterEqual(result.reduce_rounds, 1)
        self.assertEqual(calls["reduce_errors"], 1)
        self.assertIn("reduce_group_start", [name for name, _ in progress_events])

    def test_reduce_summaries_single_summary_shortcut(self) -> None:
        calls = {"n": 0}

        def _fake_llm(_prompt: str) -> str:
            calls["n"] += 1
            return "SHOULD_NOT_CALL"

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=["only_one"],
            period_start="2026-03-18T00:00:00+00:00",
            period_end="2026-03-18T01:00:00+00:00",
        )
        self.assertEqual(final_summary, "only_one")
        self.assertEqual(llm_calls, 0)
        self.assertEqual(reduce_rounds, 0)
        self.assertEqual(calls["n"], 0)

    def test_reduce_summaries_keeps_all_batches_including_empty(self) -> None:
        captured_prompt: Dict[str, str] = {}

        def _fake_llm(prompt: str) -> str:
            captured_prompt["text"] = prompt
            return self._structured_summary_json("reduce-all-batches", total_log_entries=2)

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(use_instructor=False),
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=[
                self._structured_summary_json("map-empty", total_log_entries=1, is_empty=True),
                self._structured_summary_json("map-nonempty", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00+00:00",
            period_end="2026-03-18T01:00:00+00:00",
        )
        self.assertIn('"batch_id": "reduce-all-batches"', final_summary)
        self.assertGreaterEqual(llm_calls, 1)
        self.assertGreaterEqual(reduce_rounds, 1)
        prompt_text = captured_prompt.get("text", "")
        self.assertIn("map-empty", prompt_text)
        self.assertIn("map-nonempty", prompt_text)

    def test_reduce_summaries_raises_on_non_overflow_error(self) -> None:
        def _fake_llm(_prompt: str) -> str:
            raise requests.exceptions.ConnectionError("network error")

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(use_instructor=False),
        )
        with self.assertRaises(requests.exceptions.ConnectionError):
            summarizer._reduce_summaries(
                chunk_summaries=[
                    self._structured_summary_json("map-a", total_log_entries=1),
                    self._structured_summary_json("map-b", total_log_entries=1),
                ],
                period_start="2026-03-18T00:00:00+00:00",
                period_end="2026-03-18T01:00:00+00:00",
            )

    def test_reduce_summaries_raises_on_empty_reduce_payload(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "",
            config=SummarizerConfig(use_instructor=False),
        )
        with self.assertRaises(RuntimeError):
            summarizer._reduce_summaries(
                chunk_summaries=[
                    self._structured_summary_json("map-a", total_log_entries=1),
                    self._structured_summary_json("map-b", total_log_entries=1),
                ],
                period_start="2026-03-18T00:00:00+00:00",
                period_end="2026-03-18T01:00:00+00:00",
            )

    def test_reduce_summaries_respects_fixed_group_size(self) -> None:
        merged_json = self._structured_summary_json("reduce-fixed-groups", total_log_entries=3)
        calls = {"n": 0}

        def _fake_llm(_prompt: str) -> str:
            calls["n"] += 1
            return merged_json

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(
                adaptive_reduce_on_overflow=False,
                reduce_group_size=2,
                use_instructor=False,
            ),
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=[
                self._structured_summary_json("map-a", total_log_entries=1),
                self._structured_summary_json("map-b", total_log_entries=1),
                self._structured_summary_json("map-c", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00+00:00",
            period_end="2026-03-18T01:00:00+00:00",
        )
        self.assertIn('"batch_id": "reduce-fixed-groups"', final_summary)
        self.assertEqual(llm_calls, 2)
        self.assertEqual(reduce_rounds, 2)
        self.assertEqual(calls["n"], 2)

    def test_reduce_summaries_raises_on_empty_payload_with_group_size(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "",
            config=SummarizerConfig(
                adaptive_reduce_on_overflow=False,
                reduce_group_size=2,
                use_instructor=False,
            ),
        )
        with self.assertRaises(RuntimeError):
            summarizer._reduce_summaries(
                chunk_summaries=[
                    self._structured_summary_json("map-a", total_log_entries=1),
                    self._structured_summary_json("map-b", total_log_entries=1),
                ],
                period_start="2026-03-18T00:00:00+00:00",
                period_end="2026-03-18T01:00:00+00:00",
            )

    def test_reduce_summaries_clamps_group_size_to_two(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: self._structured_summary_json("reduce-clamped", total_log_entries=2),
            config=SummarizerConfig(
                adaptive_reduce_on_overflow=False,
                reduce_group_size=1,
                use_instructor=False,
            ),
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=[
                self._structured_summary_json("map-a", total_log_entries=1),
                self._structured_summary_json("map-b", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00+00:00",
            period_end="2026-03-18T01:00:00+00:00",
        )
        self.assertIn('"batch_id": "reduce-clamped"', final_summary)
        self.assertEqual(llm_calls, 1)
        self.assertEqual(reduce_rounds, 1)

    def test_reduce_summaries_budget_fallback_emits_event_and_completes(self) -> None:
        events = []
        merged_json = self._structured_summary_json("reduce-budget-ok", total_log_entries=3)
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: merged_json,
            config=SummarizerConfig(
                adaptive_reduce_on_overflow=True,
                reduce_prompt_max_chars=1,
                use_instructor=False,
            ),
            on_progress=lambda event, payload: events.append((event, payload)),
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=[
                self._structured_summary_json("map-a", total_log_entries=1),
                self._structured_summary_json("map-b", total_log_entries=1),
                self._structured_summary_json("map-c", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00+00:00",
            period_end="2026-03-18T01:00:00+00:00",
        )
        self.assertIn('"batch_id": "reduce-budget-ok"', final_summary)
        self.assertGreaterEqual(llm_calls, 1)
        self.assertGreaterEqual(reduce_rounds, 1)
        self.assertIn("reduce_group_start", [name for name, _ in events])

    def test_reduce_group_start_progress_contains_input_chars(self) -> None:
        events = []
        merged_json = self._structured_summary_json("reduce-input-chars", total_log_entries=3)
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: merged_json,
            config=SummarizerConfig(
                adaptive_reduce_on_overflow=False,
                reduce_group_size=2,
                use_instructor=False,
            ),
            on_progress=lambda event, payload: events.append((event, payload)),
        )
        summarizer._reduce_summaries(
            chunk_summaries=[
                self._structured_summary_json("map-a", total_log_entries=1),
                self._structured_summary_json("map-b", total_log_entries=1),
                self._structured_summary_json("map-c", total_log_entries=1),
            ],
            period_start="2026-03-18T00:00:00+00:00",
            period_end="2026-03-18T01:00:00+00:00",
        )
        starts = [payload for event, payload in events if event == "reduce_group_start"]
        self.assertTrue(starts)
        first = starts[0]
        self.assertIn("input_chars", first)
        self.assertGreater(int(first.get("input_chars", 0)), 0)

    def test_map_summary_returns_structured_json_with_required_sections(self) -> None:
        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [
                {
                    "timestamp": "2026-03-25T10:00:00+00:00",
                    "message": "error timeout",
                    "_source": "svc-a",
                }
            ]

        def _fake_llm(_prompt: str) -> str:
            return json.dumps(
                {
                    "context": {},
                    "timeline": [
                        {
                            "id": "evt-1",
                            "timestamp": "2026-03-25T10:00:00+00:00",
                            "source": "svc-a",
                            "description": "Timeout detected",
                            "severity": "high",
                            "importance": 0.9,
                            "evidence_type": "FACT",
                            "evidence_quote": "error timeout",
                            "tags": ["timeout"],
                        }
                    ],
                    "causal_links": [],
                    "alert_refs": [],
                    "hypotheses": [],
                    "pinned_facts": [],
                    "gaps": [],
                    "impact": {},
                    "conflicts": [],
                    "data_quality": {
                        "is_empty": False,
                        "noise_ratio": 0.2,
                        "has_gaps": False,
                        "gap_periods": [],
                        "notes": "",
                    },
                    "preliminary_recommendations": [],
                },
                ensure_ascii=False,
            )

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=_fake_llm,
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=200,
                keep_map_summaries_in_result=True,
            ),
            prompt_context={"sql_query": "SELECT * FROM logs"},
        )
        result = summarizer.summarize_period(
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            columns=["timestamp", "message", "_source"],
        )
        payload = json.loads(result.summary)
        for key in (
            "context",
            "timeline",
            "causal_links",
            "alert_refs",
            "hypotheses",
            "pinned_facts",
            "gaps",
            "impact",
            "conflicts",
            "data_quality",
            "preliminary_recommendations",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["context"]["batch_id"], "batch-000001")
        self.assertEqual(payload["context"]["source_query"], ["SELECT * FROM logs"])

    def test_map_summary_uses_instructor_structured_path_when_enabled(self) -> None:
        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [{"timestamp": "2026-03-25T10:00:00+00:00", "message": "line"}]

        map_model = MapBatchSummaryModel.model_validate(
            {
                "context": {
                    "batch_id": "tmp",
                    "time_range_start": "2026-03-25T10:00:00+00:00",
                    "time_range_end": "2026-03-25T10:00:01+00:00",
                    "total_log_entries": 1,
                    "source_query": [],
                    "source_services": [],
                },
                "timeline": [],
                "causal_links": [],
                "alert_refs": [],
                "hypotheses": [],
                "pinned_facts": [],
                "gaps": [],
                "impact": {},
                "conflicts": [],
                "data_quality": {
                    "is_empty": True,
                    "noise_ratio": 1.0,
                    "has_gaps": False,
                    "gap_periods": [],
                    "notes": "",
                },
                "preliminary_recommendations": [],
            }
        )
        llm_prompts: list[str] = []
        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=lambda prompt: llm_prompts.append(prompt) or "FREEFORM",
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=200,
                keep_map_summaries_in_result=True,
                use_instructor=True,
            ),
            prompt_context={"sql_query": "SELECT * FROM logs"},
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer,
            "_call_structured_with_instructor",
            return_value=(map_model, 2),
        ) as mocked_instructor:
            result = summarizer.summarize_period(
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                columns=["timestamp", "message"],
            )

        self.assertTrue(mocked_instructor.called)
        payload = json.loads(result.summary)
        self.assertEqual(payload["context"]["batch_id"], "batch-000001")
        self.assertEqual(payload["context"]["source_query"], ["SELECT * FROM logs"])
        # One map structured call (2 attempts) + freeform call.
        self.assertGreaterEqual(int(result.llm_calls), 3)
        self.assertEqual(len(llm_prompts), 1)

    def test_map_summary_invalid_grammar_falls_back_to_legacy_and_disables_map_instructor(self) -> None:
        valid_map_json = json.dumps(
            {
                "context": {
                    "batch_id": "tmp",
                    "time_range_start": "2026-03-25T10:00:00+00:00",
                    "time_range_end": "2026-03-25T10:00:01+00:00",
                    "total_log_entries": 1,
                    "source_query": [],
                    "source_services": [],
                },
                "timeline": [],
                "causal_links": [],
                "alert_refs": [],
                "hypotheses": [],
                "pinned_facts": [],
                "gaps": [],
                "impact": {},
                "conflicts": [],
                "data_quality": {
                    "is_empty": True,
                    "noise_ratio": 1.0,
                    "has_gaps": False,
                    "gap_periods": [],
                    "notes": "",
                },
                "preliminary_recommendations": [],
            },
            ensure_ascii=False,
        )
        progress_events = []
        llm_calls = []

        def _fake_llm(_prompt: str) -> str:
            llm_calls.append("call")
            return valid_map_json

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(use_instructor=True),
            on_progress=lambda event, payload: progress_events.append((event, payload)),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer,
            "_call_structured_with_instructor",
            side_effect=RuntimeError("Hosted_vllmException - Invalid grammar request with cache hit"),
        ) as mocked_instructor:
            items_first, calls_first = summarizer._summarize_rows_with_auto_shrink(
                rows_chunk=[{"timestamp": "2026-03-25T10:00:00+00:00", "message": "line-1"}],
                columns=["timestamp", "message"],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                batch_number=1,
                total_batches=1,
            )
            items_second, calls_second = summarizer._summarize_rows_with_auto_shrink(
                rows_chunk=[{"timestamp": "2026-03-25T10:00:01+00:00", "message": "line-2"}],
                columns=["timestamp", "message"],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                batch_number=2,
                total_batches=2,
            )

        self.assertEqual(mocked_instructor.call_count, 1)
        self.assertTrue(summarizer._instructor_map_disabled_due_grammar)
        self.assertEqual(len(items_first), 1)
        self.assertEqual(len(items_second), 1)
        self.assertGreaterEqual(calls_first, 2)
        self.assertEqual(calls_second, 1)
        self.assertEqual(len(llm_calls), 2)
        event_names = [event for event, _ in progress_events]
        self.assertIn("map_instructor_fallback", event_names)

    def test_map_summary_instructor_empty_choices_falls_back_to_legacy(self) -> None:
        valid_map_json = json.dumps(
            {
                "context": {
                    "batch_id": "tmp",
                    "time_range_start": "2026-03-25T10:00:00+00:00",
                    "time_range_end": "2026-03-25T10:00:01+00:00",
                    "total_log_entries": 1,
                    "source_query": [],
                    "source_services": [],
                },
                "timeline": [],
                "causal_links": [],
                "alert_refs": [],
                "hypotheses": [],
                "pinned_facts": [],
                "gaps": [],
                "impact": {},
                "conflicts": [],
                "data_quality": {
                    "is_empty": True,
                    "noise_ratio": 1.0,
                    "has_gaps": False,
                    "gap_periods": [],
                    "notes": "",
                },
                "preliminary_recommendations": [],
            },
            ensure_ascii=False,
        )
        progress_events = []
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: valid_map_json,
            config=SummarizerConfig(use_instructor=True),
            on_progress=lambda event, payload: progress_events.append((event, payload)),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer,
            "_call_structured_with_instructor",
            side_effect=RuntimeError("'NoneType' object has no attribute 'choices'"),
        ):
            items, calls = summarizer._summarize_rows_with_auto_shrink(
                rows_chunk=[{"timestamp": "2026-03-25T10:00:00+00:00", "message": "line-1"}],
                columns=["timestamp", "message"],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                batch_number=1,
                total_batches=1,
            )

        self.assertEqual(len(items), 1)
        self.assertGreaterEqual(calls, 2)
        fallback_payloads = [
            payload for event, payload in progress_events if event == "map_instructor_fallback"
        ]
        self.assertTrue(fallback_payloads)
        self.assertEqual(
            str(fallback_payloads[0].get("reason")),
            "instructor_empty_choices_payload",
        )

    def test_is_instructor_empty_choices_exception_detects_nested_last_exception(self) -> None:
        outer = RuntimeError("instructor retry wrapper")
        setattr(outer, "last_exception", AttributeError("'NoneType' object has no attribute 'choices'"))
        self.assertTrue(_is_instructor_empty_choices_exception(outer))

    def test_is_read_timeout_exception_detects_nested_instructor_timeout(self) -> None:
        outer = RuntimeError("instructor retry wrapper")
        setattr(outer, "last_exception", RuntimeError("Request timed out."))
        self.assertTrue(_is_read_timeout_exception(outer))

    def test_call_structured_with_instructor_retries_on_empty_choices_then_succeeds(self) -> None:
        payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        expected = IncidentSummary.model_validate(payload)

        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                _ = kwargs
                self.calls += 1
                if self.calls < 3:
                    raise RuntimeError(
                        "<failed_attempts><last_exception>'NoneType' object has no attribute 'choices'</last_exception>"
                    )
                return expected

        fake_completions = _FakeCompletions()
        fake_chat = type("FakeChat", (), {"completions": fake_completions})()
        fake_client = type("FakeClient", (), {"chat": fake_chat})()

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_instructor=True),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer, "_get_instructor_client", return_value=fake_client
        ), patch.object(
            summarizer, "_structured_call_retry_limit", return_value=3
        ), patch.object(
            summarizer, "_structured_call_timeout_base", return_value=1.0
        ), patch(
            "my_summarizer.time.sleep", return_value=None
        ):
            parsed, attempts = summarizer._call_structured_with_instructor(
                prompt="test",
                response_model=IncidentSummary,
                stage="reduce",
            )

        self.assertEqual(parsed, expected)
        self.assertEqual(attempts, 3)
        self.assertEqual(fake_completions.calls, 3)

    def test_is_transient_gateway_exception_detects_nginx_html(self) -> None:
        err = RuntimeError(
            "<html><body><center><h1>503 Service Temporarily Unavailable</h1></center><hr><center>nginx</center></body></html>"
        )
        self.assertTrue(_is_transient_gateway_exception(err))

    def test_is_transient_gateway_exception_detects_504_time_out_html(self) -> None:
        err = RuntimeError(
            "<html><body><center><h1>504 Gateway Time-out</h1></center><hr><center>nginx</center></body></html>"
        )
        self.assertTrue(_is_transient_gateway_exception(err))

    def test_is_transient_gateway_exception_detects_500_internal_server_error(self) -> None:
        err = RuntimeError(
            "Error code: 500 - {'error': {'message': 'litellm.InternalServerError: "
            "InternalServerError: Hosted_vllmException - Connection error..'}}"
        )
        self.assertTrue(_is_transient_gateway_exception(err))

    def test_call_structured_with_instructor_retries_on_gateway_error_then_succeeds(self) -> None:
        payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        expected = IncidentSummary.model_validate(payload)

        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                _ = kwargs
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(
                        "<html><head><title>502 Bad Gateway</title></head><body><center><h1>502 Bad Gateway</h1></center><hr><center>nginx</center></body></html>"
                    )
                return expected

        fake_completions = _FakeCompletions()
        fake_chat = type("FakeChat", (), {"completions": fake_completions})()
        fake_client = type("FakeClient", (), {"chat": fake_chat})()

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_instructor=True),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer, "_get_instructor_client", return_value=fake_client
        ), patch.object(
            summarizer, "_structured_call_retry_limit", return_value=2
        ), patch.object(
            summarizer, "_structured_call_timeout_base", return_value=1.0
        ), patch(
            "my_summarizer.time.sleep", return_value=None
        ) as mocked_sleep:
            parsed, attempts = summarizer._call_structured_with_instructor(
                prompt="test",
                response_model=IncidentSummary,
                stage="reduce",
            )

        self.assertEqual(parsed, expected)
        self.assertEqual(attempts, 2)
        self.assertEqual(fake_completions.calls, 2)
        mocked_sleep.assert_called()

    def test_call_structured_with_instructor_stops_on_gateway_retry_cap(self) -> None:
        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                _ = kwargs
                self.calls += 1
                raise RuntimeError(
                    "<html><head><title>502 Bad Gateway</title></head>"
                    "<body><center><h1>502 Bad Gateway</h1></center><hr><center>nginx</center></body></html>"
                )

        fake_completions = _FakeCompletions()
        fake_chat = type("FakeChat", (), {"completions": fake_completions})()
        fake_client = type("FakeClient", (), {"chat": fake_chat})()

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_instructor=True),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer, "_get_instructor_client", return_value=fake_client
        ), patch.object(
            summarizer, "_structured_call_retry_limit", return_value=-1
        ), patch.object(
            summarizer, "_structured_call_timeout_base", return_value=1.0
        ), patch(
            "my_summarizer.time.sleep", return_value=None
        ):
            with self.assertRaises(RuntimeError):
                summarizer._call_structured_with_instructor(
                    prompt="test",
                    response_model=IncidentSummary,
                    stage="reduce",
                    gateway_retry_cap=2,
                )

        # 2 retries allowed after initial attempt => total 3 attempts.
        self.assertEqual(fake_completions.calls, 3)

    def test_call_structured_with_instructor_stops_on_malformed_payload_retry_cap(self) -> None:
        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                _ = kwargs
                self.calls += 1
                raise RuntimeError(
                    "<failed_attempts><last_exception>'NoneType' object has no attribute 'choices'</last_exception>"
                )

        fake_completions = _FakeCompletions()
        fake_chat = type("FakeChat", (), {"completions": fake_completions})()
        fake_client = type("FakeClient", (), {"chat": fake_chat})()

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_instructor=True),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer, "_get_instructor_client", return_value=fake_client
        ), patch.object(
            summarizer, "_structured_call_retry_limit", return_value=-1
        ), patch.object(
            summarizer, "_structured_call_timeout_base", return_value=1.0
        ), patch(
            "my_summarizer.time.sleep", return_value=None
        ):
            with self.assertRaises(RuntimeError):
                summarizer._call_structured_with_instructor(
                    prompt="test",
                    response_model=IncidentSummary,
                    stage="reduce",
                    gateway_retry_cap=2,
                )

        # 2 retries allowed after initial attempt => total 3 attempts.
        self.assertEqual(fake_completions.calls, 3)

    def test_call_structured_with_instructor_doubles_timeout_on_read_timeout(self) -> None:
        payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        expected = IncidentSummary.model_validate(payload)

        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0
                self.timeouts: list[float] = []

            def create(self, **kwargs):
                self.calls += 1
                self.timeouts.append(float(kwargs.get("timeout", 0.0)))
                if self.calls == 1:
                    raise requests.exceptions.ReadTimeout("read timeout")
                return expected

        fake_completions = _FakeCompletions()
        fake_chat = type("FakeChat", (), {"completions": fake_completions})()
        fake_client = type("FakeClient", (), {"chat": fake_chat})()

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_instructor=True),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer, "_get_instructor_client", return_value=fake_client
        ), patch.object(
            summarizer, "_structured_call_retry_limit", return_value=1
        ), patch.object(
            summarizer, "_structured_call_timeout_base", return_value=3.0
        ), patch(
            "my_summarizer.time.sleep", return_value=None
        ):
            parsed, attempts = summarizer._call_structured_with_instructor(
                prompt="test",
                response_model=IncidentSummary,
                stage="reduce",
            )

        self.assertEqual(parsed, expected)
        self.assertEqual(attempts, 2)
        self.assertEqual(fake_completions.calls, 2)
        self.assertEqual(fake_completions.timeouts, [3.0, 6.0])

    def test_call_structured_with_instructor_reduce_timeout_cap_breaks_after_one_escalation(self) -> None:
        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0
                self.timeouts: list[float] = []

            def create(self, **kwargs):
                self.calls += 1
                self.timeouts.append(float(kwargs.get("timeout", 0.0)))
                raise RuntimeError("Request timed out.")

        fake_completions = _FakeCompletions()
        fake_chat = type("FakeChat", (), {"completions": fake_completions})()
        fake_client = type("FakeClient", (), {"chat": fake_chat})()

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda prompt: prompt,
            config=SummarizerConfig(use_instructor=True),
        )

        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer, "_get_instructor_client", return_value=fake_client
        ), patch.object(
            summarizer, "_structured_call_retry_limit", return_value=-1
        ), patch.object(
            summarizer, "_structured_call_timeout_base", return_value=120.0
        ), patch(
            "my_summarizer.time.sleep", return_value=None
        ):
            with self.assertRaises(RuntimeError):
                summarizer._call_structured_with_instructor(
                    prompt="test",
                    response_model=IncidentSummary,
                    stage="reduce_l2_g1of1_d0",
                )

        # One escalation only: 120 -> 240, then propagate for split.
        self.assertEqual(fake_completions.calls, 2)
        self.assertEqual(fake_completions.timeouts, [120.0, 240.0])

    def test_reduce_structured_group_uses_instructor_when_enabled(self) -> None:
        summary_payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        inp = IncidentSummary.model_validate(summary_payload)
        out = IncidentSummary.model_validate(
            {
                **summary_payload,
                "context": {
                    **summary_payload["context"],
                    "batch_id": "reduce-r1-g1",
                },
            }
        )
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "legacy-path-should-not-be-used",
            config=SummarizerConfig(use_instructor=True, use_new_algorithm=True),
        )
        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer,
            "_call_structured_with_instructor",
            return_value=(out, 1),
        ) as mocked_instructor:
            reduced, calls = summarizer._reduce_structured_group(
                summaries=[inp, inp],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                reduce_round=1,
                group_index=1,
                group_total=1,
                sources=None,
            )

        self.assertTrue(mocked_instructor.called)
        self.assertEqual(calls, 1)
        self.assertEqual(reduced.context.batch_id, "reduce-r1-g1")
        call_kwargs = mocked_instructor.call_args.kwargs
        self.assertEqual(call_kwargs.get("gateway_retry_cap"), summarizer.config.reduce_gateway_retry_cap)
        self.assertEqual(call_kwargs.get("stage"), "reduce_l1_g1of1_d0")

    def test_reduce_structured_group_l2_uses_gateway_cap_one(self) -> None:
        summary_payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        inp = IncidentSummary.model_validate(summary_payload)
        out = IncidentSummary.model_validate(
            {
                **summary_payload,
                "context": {
                    **summary_payload["context"],
                    "batch_id": "reduce-r2-g1",
                },
            }
        )
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "legacy-path-should-not-be-used",
            config=SummarizerConfig(use_instructor=True, use_new_algorithm=True, reduce_gateway_retry_cap=5),
        )
        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer,
            "_call_structured_with_instructor",
            return_value=(out, 1),
        ) as mocked_instructor:
            reduced, calls = summarizer._reduce_structured_group(
                summaries=[inp, inp],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                reduce_round=2,
                group_index=1,
                group_total=1,
                sources=None,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(reduced.context.batch_id, "reduce-r2-g1")
        call_kwargs = mocked_instructor.call_args.kwargs
        self.assertEqual(call_kwargs.get("gateway_retry_cap"), 1)
        self.assertEqual(call_kwargs.get("stage"), "reduce_l2_g1of1_d0")

    def test_reduce_structured_group_splits_on_500_and_recovers(self) -> None:
        summary_payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        inp = IncidentSummary.model_validate(summary_payload)
        out = IncidentSummary.model_validate(
            {
                **summary_payload,
                "context": {
                    **summary_payload["context"],
                    "batch_id": "reduce-r1-g1",
                },
            }
        )
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "legacy-path-should-not-be-used",
            config=SummarizerConfig(use_instructor=True, use_new_algorithm=True),
        )
        call_sequence = [
            RuntimeError(
                "Error code: 500 - {'error': {'message': 'litellm.InternalServerError: "
                "InternalServerError: Hosted_vllmException - Connection error..'}}"
            ),
            (out, 1),
        ]
        with patch.object(summarizer, "_instructor_enabled", return_value=True), patch.object(
            summarizer,
            "_call_structured_with_instructor",
            side_effect=call_sequence,
        ) as mocked_instructor, patch.object(
            summarizer,
            "_compress_summary_on_overflow",
            side_effect=lambda summary, **_: (summary, 0),
        ):
            reduced, calls = summarizer._reduce_structured_group(
                summaries=[inp, inp],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                reduce_round=1,
                group_index=1,
                group_total=1,
                sources=None,
            )

        self.assertEqual(reduced.context.batch_id, "reduce-r1-g1")
        self.assertEqual(calls, 1)
        self.assertEqual(mocked_instructor.call_count, 2)

    def test_reduce_structured_group_aborts_on_max_depth(self) -> None:
        summary_payload = {
            "context": {
                "batch_id": "map-000001",
                "time_range_start": "2026-03-25T10:00:00+00:00",
                "time_range_end": "2026-03-25T10:05:00+00:00",
                "total_log_entries": 1,
                "source_query": [],
                "source_services": [],
            },
            "timeline": [],
            "causal_links": [],
            "alert_refs": [],
            "hypotheses": [],
            "pinned_facts": [],
            "gaps": [],
            "impact": {},
            "conflicts": [],
            "data_quality": {"is_empty": True, "noise_ratio": 1.0, "notes": ""},
            "preliminary_recommendations": [],
        }
        inp = IncidentSummary.model_validate(summary_payload)
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "legacy-path-should-not-be-used",
            config=SummarizerConfig(use_instructor=False, use_new_algorithm=True),
        )
        with self.assertRaises(RuntimeError):
            summarizer._reduce_structured_group(
                summaries=[inp, inp],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
                reduce_round=1,
                group_index=1,
                group_total=1,
                sources=None,
                depth=13,
            )

    def test_prepare_summary_for_reduce_level_prunes_l2_fields(self) -> None:
        summary_payload = self._structured_summary_dict("map-prune")
        summary_payload["pinned_facts"] = [
            {
                "id": "pf-1",
                "fact": "Very long pinned fact text " + ("x" * 200),
                "evidence_quote": "quote",
                "relevance": "test",
                "importance": 0.95,
            }
        ]
        summary_payload["gaps"] = [
            {
                "id": "gap-1",
                "description": "Gap description " + ("y" * 180),
                "between_events": [],
                "missing_data": "Need more logs",
            }
        ]
        summary_payload["data_quality"]["notes"] = "n" * 250
        summary = IncidentSummary.model_validate(summary_payload)
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "unused",
            config=SummarizerConfig(reduce_input_max_chars=100_000),
        )

        prepared, calls = summarizer._prepare_summary_for_reduce_level(
            summary=summary,
            reduce_round=2,
            summary_index=1,
            summary_total=1,
        )
        self.assertEqual(calls, 0)
        self.assertLessEqual(len(prepared.pinned_facts), 1)
        if prepared.pinned_facts:
            self.assertEqual(prepared.pinned_facts[0].id, "pf-annotation")
        self.assertLessEqual(len(prepared.gaps), 1)
        self.assertIn("reduce_l2_pruned", prepared.data_quality.notes)

    def test_prepare_summary_for_reduce_level_raises_when_still_too_large(self) -> None:
        summary_payload = self._structured_summary_dict("map-too-large")
        summary_payload["timeline"] = [
            {
                "id": f"evt-{i}",
                "timestamp": "2026-03-25T10:00:00+00:00",
                "source": "svc-a",
                "description": "event " + ("z" * 300),
                "severity": "low",
                "importance": 0.8,
                "evidence_type": "HYPOTHESIS",
                "tags": ["test"],
            }
            for i in range(40)
        ]
        summary = IncidentSummary.model_validate(summary_payload)
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "unused",
            config=SummarizerConfig(reduce_input_max_chars=1000),
        )

        with patch.object(summarizer, "_prune_l2_plus_fields", return_value=summary), patch.object(
            summarizer, "_compress_summary_on_overflow", return_value=(summary, 0)
        ), patch.object(
            summarizer, "_aggressive_reduce_prune", return_value=summary
        ):
            with self.assertRaises(RuntimeError):
                summarizer._prepare_summary_for_reduce_level(
                    summary=summary,
                    reduce_round=2,
                    summary_index=1,
                    summary_total=1,
                )

    def test_map_summary_invalid_json_is_converted_to_degraded_structured_payload(self) -> None:
        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [{"timestamp": "2026-03-25T10:00:00+00:00", "message": "line"}]

        progress_events = []
        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=lambda _prompt: "not a json response",
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=200,
                keep_map_summaries_in_result=True,
            ),
            on_progress=lambda event, payload: progress_events.append((event, payload)),
        )
        result = summarizer.summarize_period(
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            columns=["timestamp", "message"],
        )
        payload = json.loads(result.summary)
        self.assertIn("data_quality", payload)
        notes_text = str(payload["data_quality"].get("notes", ""))
        self.assertTrue(notes_text)
        self.assertIn("Raw LLM map summary", notes_text)
        event_names = [event for event, _ in progress_events]
        self.assertIn("map_schema_retry", event_names)
        self.assertIn("map_schema_degraded", event_names)
        degraded_payload = next(
            payload for event, payload in progress_events if event == "map_schema_degraded"
        )
        self.assertIn("raw_preview", degraded_payload)
        self.assertIn("raw_len", degraded_payload)

    def test_map_summary_schema_can_recover_after_retry(self) -> None:
        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [{"timestamp": "2026-03-25T10:00:00+00:00", "message": "line"}]

        valid_payload = json.dumps(
            {
                "context": {
                    "batch_id": "batch-000001",
                    "time_range_start": "2026-03-25T10:00:00+00:00",
                    "time_range_end": "2026-03-25T10:00:01+00:00",
                    "total_log_entries": 1,
                    "source_query": [],
                    "source_services": [],
                },
                "timeline": [],
                "causal_links": [],
                "alert_refs": [],
                "hypotheses": [],
                "pinned_facts": [],
                "gaps": [],
                "impact": {},
                "conflicts": [],
                "data_quality": {
                    "is_empty": True,
                    "noise_ratio": 1.0,
                    "has_gaps": False,
                    "gap_periods": [],
                    "notes": "",
                },
                "preliminary_recommendations": [],
            },
            ensure_ascii=False,
        )
        llm_responses = iter(["not a json response", valid_payload])
        progress_events = []

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=lambda _prompt: next(llm_responses),
            config=SummarizerConfig(
                page_limit=1000,
                llm_chunk_rows=200,
                keep_map_summaries_in_result=True,
            ),
            on_progress=lambda event, payload: progress_events.append((event, payload)),
        )
        result = summarizer.summarize_period(
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            columns=["timestamp", "message"],
        )
        payload = json.loads(result.summary)
        self.assertIn("data_quality", payload)
        event_names = [event for event, _ in progress_events]
        self.assertIn("map_schema_retry", event_names)
        self.assertIn("map_schema_recovered", event_names)
        self.assertNotIn("map_schema_degraded", event_names)

    def test_reduce_uses_premerged_alert_refs(self) -> None:
        def _mk_summary(
            batch_id: str,
            alert_status: str,
            explanation: str,
            *,
            event_id: str = "evt-1",
        ) -> str:
            return json.dumps(
                {
                    "context": {
                        "batch_id": batch_id,
                        "time_range_start": "2026-03-25T10:00:00+00:00",
                        "time_range_end": "2026-03-25T10:05:00+00:00",
                        "total_log_entries": 1,
                        "source_query": ["SELECT * FROM logs"],
                        "source_services": ["svc-a"],
                    },
                    "timeline": [
                        {
                            "id": event_id,
                            "timestamp": "2026-03-25T10:00:00+00:00",
                            "source": "svc-a",
                            "description": "event",
                            "severity": "low",
                            "importance": 0.9,
                            "evidence_type": "HYPOTHESIS",
                            "tags": ["test"],
                        }
                    ],
                    "causal_links": [],
                    "alert_refs": [
                        {
                            "alert_id": "A1",
                            "status": alert_status,
                            "related_events": [event_id],
                            "explanation": explanation,
                        }
                    ],
                    "hypotheses": [],
                    "pinned_facts": [],
                    "gaps": [],
                    "impact": {
                        "affected_services": ["svc-a"],
                        "affected_operations": [],
                        "error_counts": [],
                    },
                    "conflicts": [],
                    "data_quality": {"is_empty": False, "noise_ratio": 0.1, "notes": ""},
                    "preliminary_recommendations": [],
                },
                ensure_ascii=False,
            )

        prompts: list[str] = []

        def _fake_llm(prompt: str) -> str:
            prompts.append(prompt)
            return _mk_summary(
                "reduce-r1-g1",
                "EXPLAINED",
                "merged explanation",
                event_id="evt-out",
            )

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(
                use_new_algorithm=True,
                adaptive_reduce_on_overflow=True,
            ),
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=[
                _mk_summary("batch-1", "PARTIALLY", "from batch 1", event_id="evt-1"),
                _mk_summary("batch-2", "EXPLAINED", "from batch 2", event_id="evt-2"),
            ],
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
        )

        self.assertEqual(llm_calls, 1)
        self.assertEqual(reduce_rounds, 1)
        self.assertIn("PROGRAMMATICALLY MERGED ALERT_REFS", prompts[0])
        self.assertIn('"status": "EXPLAINED"', prompts[0])
        self.assertIn('"related_events": [', prompts[0])
        self.assertIn('"evt-1"', prompts[0])
        self.assertIn('"evt-2"', prompts[0])
        parsed = json.loads(final_summary)
        self.assertEqual(parsed["alert_refs"][0]["status"], "EXPLAINED")

    def test_reduce_prompt_uses_json_objects_not_escaped_strings(self) -> None:
        captured_prompt = {"text": ""}

        def _mk_summary(batch_id: str) -> str:
            return json.dumps(
                {
                    "context": {
                        "batch_id": batch_id,
                        "time_range_start": "2026-03-25T10:00:00+00:00",
                        "time_range_end": "2026-03-25T10:05:00+00:00",
                        "total_log_entries": 1,
                        "source_query": [],
                        "source_services": ["svc-a"],
                    },
                    "timeline": [
                        {
                            "id": "evt-1",
                            "timestamp": "2026-03-25T10:00:00+00:00",
                            "source": "svc-a",
                            "description": "event",
                            "severity": "low",
                            "importance": 0.9,
                            "evidence_type": "HYPOTHESIS",
                            "tags": ["test"],
                        }
                    ],
                    "causal_links": [],
                    "alert_refs": [],
                    "hypotheses": [],
                    "pinned_facts": [],
                    "gaps": [],
                    "impact": {},
                    "conflicts": [],
                    "data_quality": {"is_empty": False, "noise_ratio": 0.1, "notes": ""},
                    "preliminary_recommendations": [],
                },
                ensure_ascii=False,
            )

        def _fake_llm(prompt: str) -> str:
            captured_prompt["text"] = prompt
            return _mk_summary("reduce-out")

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(use_new_algorithm=True),
        )
        summarizer._reduce_summaries(
            chunk_summaries=[_mk_summary("batch-1"), _mk_summary("batch-2")],
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
        )

        prompt_text = captured_prompt["text"]
        self.assertIn('"context": {', prompt_text)
        # Old behavior encoded summaries as JSON strings with escaped quotes.
        self.assertNotIn('\\"context\\"', prompt_text)

    def test_reduce_splits_group_on_400(self) -> None:
        def _mk_summary(batch_id: str) -> str:
            return json.dumps(
                {
                    "context": {
                        "batch_id": batch_id,
                        "time_range_start": "2026-03-25T10:00:00+00:00",
                        "time_range_end": "2026-03-25T10:05:00+00:00",
                        "total_log_entries": 1,
                        "source_query": [],
                        "source_services": ["svc-a"],
                    },
                    "timeline": [
                        {
                            "id": "evt-1",
                            "timestamp": "2026-03-25T10:00:00+00:00",
                            "source": "svc-a",
                            "description": "event",
                            "severity": "low",
                            "importance": 0.9,
                            "evidence_type": "HYPOTHESIS",
                            "tags": ["test"],
                        }
                    ],
                    "causal_links": [],
                    "alert_refs": [],
                    "hypotheses": [],
                    "pinned_facts": [],
                    "gaps": [],
                    "impact": {},
                    "conflicts": [],
                    "data_quality": {"is_empty": False, "noise_ratio": 0.1, "notes": ""},
                    "preliminary_recommendations": [],
                },
                ensure_ascii=False,
            )

        class _Resp:
            status_code = 400

        calls = {"count": 0}

        def _fake_llm(_prompt: str) -> str:
            calls["count"] += 1
            if calls["count"] == 1:
                raise requests.exceptions.HTTPError(
                    "400 Client Error: Bad Request for url",
                    response=_Resp(),
                )
            return _mk_summary(f"reduce-{calls['count']}")

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=_fake_llm,
            config=SummarizerConfig(
                use_new_algorithm=True,
                adaptive_reduce_on_overflow=True,
            ),
        )
        final_summary, llm_calls, reduce_rounds = summarizer._reduce_summaries(
            chunk_summaries=[
                _mk_summary("batch-1"),
                _mk_summary("batch-2"),
                _mk_summary("batch-3"),
            ],
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
        )

        self.assertGreaterEqual(calls["count"], 3)
        self.assertGreaterEqual(llm_calls, 2)
        self.assertGreaterEqual(reduce_rounds, 1)
        parsed = json.loads(final_summary)
        self.assertIn("timeline", parsed)
        self.assertFalse(bool(parsed.get("data_quality", {}).get("is_empty", True)))

    def test_map_respects_llm_chunk_rows_limit(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "ok",
            config=SummarizerConfig(use_new_algorithm=True, llm_chunk_rows=2),
        )
        rows = [{"timestamp": f"2026-03-25T10:00:0{i}+00:00", "message": f"log-{i}"} for i in range(5)]
        chunks = summarizer._split_rows_for_map(rows=rows, columns=["timestamp", "message"])
        self.assertEqual(len(chunks), 3)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        self.assertEqual(chunks[0], rows[:2])
        self.assertEqual(chunks[1], rows[2:4])
        self.assertEqual(chunks[2], rows[4:])

    def test_map_split_for_grouped_rows_is_by_row_count(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "ok",
            config=SummarizerConfig(use_new_algorithm=True, llm_chunk_rows=8),
        )
        rows = [
            {"timestamp": "2026-03-25T10:00:00+00:00", "message": "g1", "cnt": 4},
            {"timestamp": "2026-03-25T10:00:10+00:00", "message": "g2", "cnt": 4},
            {"timestamp": "2026-03-25T10:00:20+00:00", "message": "g3", "cnt": 4},
        ]
        chunks = summarizer._split_rows_for_map(rows=rows, columns=["timestamp", "message", "cnt"])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], rows)

    def test_map_split_honors_row_limit_even_when_cnt_is_large(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "ok",
            config=SummarizerConfig(use_new_algorithm=True, llm_chunk_rows=2),
        )
        rows = [
            {"timestamp": "2026-03-25T10:00:00+00:00", "message": "g1", "cnt": 1000},
            {"timestamp": "2026-03-25T10:00:10+00:00", "message": "g2", "cnt": 1000},
            {"timestamp": "2026-03-25T10:00:20+00:00", "message": "g3", "cnt": 1000},
        ]
        chunks = summarizer._split_rows_for_map(rows=rows, columns=["timestamp", "message", "cnt"])
        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])
        self.assertEqual(chunks[0], rows[:2])
        self.assertEqual(chunks[1], rows[2:])

    def test_summarize_period_emits_aggregated_query_warning_event(self) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        calls = {"n": 0}

        def _fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit, offset)
            calls["n"] += 1
            if calls["n"] == 1:
                return [
                    {
                        "timestamp": "2026-03-25T10:00:00+00:00",
                        "message": "grouped row",
                        "cnt": 900,
                    }
                ]
            return []

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fetch_page,
            llm_call=lambda _prompt: self._structured_summary_json("map-agg"),
            config=SummarizerConfig(use_new_algorithm=True, llm_chunk_rows=200),
            on_progress=lambda event, payload: events.append((event, dict(payload))),
        )
        result = summarizer.summarize_period(
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T11:00:00+00:00",
            columns=["timestamp", "message", "cnt"],
            total_rows_estimate=900,
        )
        self.assertTrue(result.summary)
        warning_payloads = [payload for name, payload in events if name == "map_query_aggregated"]
        self.assertTrue(warning_payloads)
        warning_payload = warning_payloads[0]
        self.assertEqual(int(warning_payload.get("page_rows", 0)), 1)
        self.assertEqual(int(warning_payload.get("effective_rows", 0)), 900)

    def test_reduce_structured_keeps_non_json_batch_as_degraded_input(self) -> None:
        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "ok",
            config=SummarizerConfig(use_new_algorithm=True, llm_chunk_rows=8),
        )
        parsed_inputs: list[Any] = []

        def _fake_reduce(*, summaries, **_kwargs):
            parsed_inputs.extend(summaries)
            return summaries[0], 0

        with patch.object(summarizer, "_reduce_structured_group", side_effect=_fake_reduce):
            summary_text, llm_calls, reduce_rounds = summarizer._reduce_summaries_structured(
                chunk_summaries=[
                    "plain non-json text",
                    json.dumps(
                        {
                            "context": {
                                "batch_id": "map-000002",
                                "time_range_start": "2026-03-25T10:00:00+00:00",
                                "time_range_end": "2026-03-25T10:05:00+00:00",
                                "total_log_entries": 1,
                                "source_query": [],
                                "source_services": [],
                            },
                            "timeline": [],
                            "causal_links": [],
                            "alert_refs": [],
                            "hypotheses": [],
                            "pinned_facts": [],
                            "gaps": [],
                            "impact": {},
                            "conflicts": [],
                            "data_quality": {"is_empty": True, "noise_ratio": 1.0},
                            "preliminary_recommendations": [],
                        }
                    ),
                ],
                period_start="2026-03-25T10:00:00+00:00",
                period_end="2026-03-25T11:00:00+00:00",
            )
        self.assertEqual(llm_calls, 0)
        self.assertEqual(reduce_rounds, 1)
        self.assertTrue(summary_text)
        self.assertEqual(len(parsed_inputs), 2)
        self.assertIn("raw_summary", parsed_inputs[0].timeline[0].tags)

    def test_reduce_groups_use_fixed_group_size(self) -> None:
        def _mk_summary_obj(batch_id: str):
            raw = json.dumps(
                {
                    "context": {
                        "batch_id": batch_id,
                        "time_range_start": "2026-03-25T10:00:00+00:00",
                        "time_range_end": "2026-03-25T10:05:00+00:00",
                        "total_log_entries": 1,
                        "source_query": [],
                        "source_services": ["svc-a"],
                    },
                    "timeline": [
                        {
                            "id": "evt-1",
                            "timestamp": "2026-03-25T10:00:00+00:00",
                            "source": "svc-a",
                            "description": "event",
                            "severity": "low",
                            "importance": 0.9,
                            "evidence_type": "HYPOTHESIS",
                            "tags": ["test"],
                        }
                    ],
                    "causal_links": [],
                    "alert_refs": [],
                    "hypotheses": [],
                    "pinned_facts": [],
                    "gaps": [],
                    "impact": {},
                    "conflicts": [],
                    "data_quality": {"is_empty": False, "noise_ratio": 0.1, "notes": ""},
                    "preliminary_recommendations": [],
                },
                ensure_ascii=False,
            )
            obj = PeriodLogSummarizer(
                db_fetch_page=lambda **_: [],
                llm_call=lambda _prompt: "ok",
                config=SummarizerConfig(use_new_algorithm=True, reduce_group_size=3),
            )._parse_incident_summary_text(
                raw,
                fallback_batch_id=batch_id,
                fallback_start="2026-03-25T10:00:00+00:00",
                fallback_end="2026-03-25T10:05:00+00:00",
            )
            self.assertIsNotNone(obj)
            return obj

        summarizer = PeriodLogSummarizer(
            db_fetch_page=lambda **_: [],
            llm_call=lambda _prompt: "ok",
            config=SummarizerConfig(use_new_algorithm=True, reduce_group_size=3),
        )
        summaries = [_mk_summary_obj(f"batch-{i}") for i in range(7)]
        groups = summarizer._group_structured_summaries_by_budget(summaries)
        self.assertEqual([len(g) for g in groups], [3, 3, 1])

    def test_map_start_progress_has_expected_fields(self) -> None:
        progress_events = []

        def _fake_fetch_page(*, columns, period_start, period_end, limit, offset):
            _ = (columns, period_start, period_end, limit)
            if offset > 0:
                return []
            return [{"timestamp": "2026-03-25T10:00:00+00:00", "message": "line"}]

        summarizer = PeriodLogSummarizer(
            db_fetch_page=_fake_fetch_page,
            llm_call=lambda _prompt: "MAP_OK",
            config=SummarizerConfig(use_new_algorithm=True, llm_chunk_rows=1),
            on_progress=lambda event, payload: progress_events.append((event, payload)),
        )
        summarizer.summarize_period(
            period_start="2026-03-25T10:00:00+00:00",
            period_end="2026-03-25T10:05:00+00:00",
            columns=["timestamp", "message"],
        )
        map_start_payload = next(payload for event, payload in progress_events if event == "map_start")
        self.assertNotIn("token_budget", map_start_payload)

    def test_generate_final_reports_with_instructor_full_attempt_success(self) -> None:
        report = IncidentReport(
            summary=ReportSummary(text="full summary"),
            data_coverage=DataCoverage(period_start="2026-03-25T10:00:00+03:00", period_end="2026-03-25T11:00:00+03:00"),
            chronology=[],
            causal_chains=[],
            alert_explanations=[],
            hypotheses=[],
            conflicts=[],
            gaps=[],
            recommendations=[],
        )

        def _fake_call(*, prompt, response_model, stage):  # noqa: ANN001
            _ = (prompt, stage)
            if response_model is IncidentReport:
                return report, 2
            if response_model is FreeformReport:
                return FreeformReport(text="freeform from instructor"), 1
            raise AssertionError("unexpected response_model")

        with patch.object(
            PeriodLogSummarizer,
            "_call_structured_with_instructor",
            side_effect=_fake_call,
        ):
            out = generate_final_reports_with_instructor(
                base_structured_report='{"context":{"batch_id":"b1","time_range_start":"2026-03-25T10:00:00+03:00","time_range_end":"2026-03-25T11:00:00+03:00","total_log_entries":1,"source_query":[],"source_services":[]}}',
                base_freeform_report="",
                user_goal="incident context",
                period_start="2026-03-25T10:00:00+03:00",
                period_end="2026-03-25T11:00:00+03:00",
                alerts=[{"title": "alert-1", "details": "cpu"}],
            )

        self.assertEqual(out["algorithm_stage"], "attempt_1_full")
        self.assertIn("## 2. Резюме Инцидента", out["structured_report"])
        self.assertIn("freeform from instructor", out["freeform_report"])
        self.assertGreaterEqual(int(out["attempts"]), 3)
        self.assertIn("incident_report", out)

    def test_generate_final_reports_with_instructor_falls_back_to_split_attempt(self) -> None:
        class _Resp:
            status_code = 400

        split_part_1 = ReportPartAnalytical(
            causal_chains=[],
            alert_explanations=[],
            hypotheses=[],
            recommendations=[],
        )
        split_part_2 = ReportPartDescriptive(
            summary=ReportSummary(text="split summary"),
            data_coverage=DataCoverage(period_start="2026-03-25T10:00:00+03:00", period_end="2026-03-25T11:00:00+03:00"),
            chronology=[],
            conflicts=[],
            gaps=[],
        )
        call_order = []

        def _fake_call(*, prompt, response_model, stage):  # noqa: ANN001
            _ = prompt
            call_order.append(stage)
            if stage == "final_report_full":
                raise requests.exceptions.HTTPError(
                    "400 Client Error: Bad Request",
                    response=_Resp(),
                )
            if response_model is ReportPartAnalytical:
                return split_part_1, 1
            if response_model is ReportPartDescriptive:
                return split_part_2, 1
            if response_model is FreeformReport:
                return FreeformReport(text="freeform split"), 1
            raise AssertionError("unexpected response_model")

        with patch.object(
            PeriodLogSummarizer,
            "_call_structured_with_instructor",
            side_effect=_fake_call,
        ):
            out = generate_final_reports_with_instructor(
                base_structured_report='{"context":{"batch_id":"b1","time_range_start":"2026-03-25T10:00:00+03:00","time_range_end":"2026-03-25T11:00:00+03:00","total_log_entries":1,"source_query":[],"source_services":[]}}',
                base_freeform_report="",
                user_goal="incident context",
                period_start="2026-03-25T10:00:00+03:00",
                period_end="2026-03-25T11:00:00+03:00",
            )

        self.assertEqual(out["algorithm_stage"], "attempt_2_split")
        self.assertIn("split summary", out["structured_report"])
        self.assertIn("attempt_1_full=fail", out["notes"])
        self.assertIn("attempt_2_split=ok", out["notes"])
        self.assertIn("final_report_part1", "|".join(call_order))

    def test_generate_final_reports_with_instructor_falls_back_to_sectional_attempt(self) -> None:
        class _Resp:
            status_code = 400

        minimal_summary_json = json.dumps(
            {
                "context": {
                    "batch_id": "b-1",
                    "time_range_start": "2026-03-25T10:00:00+03:00",
                    "time_range_end": "2026-03-25T11:00:00+03:00",
                    "total_log_entries": 10,
                    "source_query": ["SELECT 1"],
                    "source_services": ["svc-a"],
                }
            },
            ensure_ascii=False,
        )

        def _fake_call(*, prompt, response_model, stage):  # noqa: ANN001
            _ = prompt
            if stage in {"final_report_full", "final_report_part1", "final_report_part2"}:
                raise requests.exceptions.HTTPError(
                    "400 Client Error: Bad Request",
                    response=_Resp(),
                )
            if response_model is ChronologySection:
                return (
                    ChronologySection(
                        chronology=[
                            ChronologyEvent(
                                id="evt-001",
                                timestamp="2026-03-25T10:15:00.123456+03:00",
                                source="svc-a",
                                description="error spike",
                                severity="high",
                                evidence_type="FACT",
                                evidence_quote="timeout to db",
                                tags=["timeout"],
                            )
                        ]
                    ),
                    1,
                )
            if response_model is CausalChainsSection:
                return (
                    CausalChainsSection(
                        causal_chains=[
                            CausalChain(
                                id="c-1",
                                cause_event_id="evt-001",
                                effect_event_id="evt-001",
                                mechanism="same-event simplified chain",
                                confidence=0.5,
                            )
                        ]
                    ),
                    1,
                )
            if response_model is AlertsSection:
                return (
                    AlertsSection(
                        alert_explanations=[
                            AlertExplanation(
                                alert_id="A1",
                                status="PARTIALLY_EXPLAINED",
                                related_events=["evt-001"],
                                explanation="partial signal",
                            )
                        ]
                    ),
                    1,
                )
            if response_model is CoverageSection:
                return (
                    CoverageSection(
                        data_coverage=DataCoverage(
                            period_start="2026-03-25T10:00:00+03:00",
                            period_end="2026-03-25T11:00:00+03:00",
                            sql_queries=["SELECT 1"],
                            services_covered=["svc-a"],
                            services_missing=[],
                            logs_processed=10,
                            notes="ok",
                        )
                    ),
                    1,
                )
            if response_model is ImpactSection:
                return (
                    ImpactSection(
                        impact=ImpactAssessment(
                            affected_services=["svc-a"],
                            affected_operations=[],
                            error_counts=["timeouts=3"],
                            duration="5m",
                            severity_assessment="high",
                        )
                    ),
                    1,
                )
            if response_model is LimitationsSection:
                return (
                    LimitationsSection(
                        limitations=AnalysisLimitations(
                            overall_confidence="medium",
                            rationale="limited logs",
                            limitations=["partial period"],
                            low_confidence_hypotheses=[],
                        )
                    ),
                    1,
                )
            if response_model is SummarySection:
                return (SummarySection(summary=ReportSummary(text="sectional summary")), 1)
            if response_model is FreeformReport:
                return (FreeformReport(text="sectional freeform"), 1)
            # for optional sections, force fallback defaults
            raise requests.exceptions.HTTPError(
                "400 Client Error: Bad Request",
                response=_Resp(),
            )

        with patch.object(
            PeriodLogSummarizer,
            "_call_structured_with_instructor",
            side_effect=_fake_call,
        ):
            out = generate_final_reports_with_instructor(
                base_structured_report=minimal_summary_json,
                base_freeform_report="",
                user_goal="incident context",
                period_start="2026-03-25T10:00:00+03:00",
                period_end="2026-03-25T11:00:00+03:00",
                alerts=[{"title": "A1", "details": "alert details"}],
            )

        self.assertEqual(out["algorithm_stage"], "attempt_3_sectional")
        self.assertIn("attempt_2_split=fail", out["notes"])
        self.assertIn("attempt_3_sectional=ok", out["notes"])
        self.assertIn("## 4. Полная Хронология Событий", out["structured_report"])
        self.assertIn("A1", out["structured_report"])
        self.assertIn("sectional summary", out["structured_report"])
        self.assertIn("sectional freeform", out["freeform_report"])
        self.assertEqual(
            out["incident_report"]["summary"]["text"],
            "sectional summary",
        )


if __name__ == "__main__":
    unittest.main()
