import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from settings import settings
from ui.pages.logs_summary_page import (
    _checkpoint_payload_from_state,
    _build_no_logs_hypothesis_prompt,
    _build_freeform_summary_prompt,
    _discover_resume_sessions,
    _enrich_stats_with_elapsed,
    _extract_last_batch_ts_from_run_dir,
    _form_values_from_saved_params,
    _load_map_summaries_from_jsonl,
    _normalize_summary_text,
    _write_json_file,
    _save_logs_summary_result,
)


class TestLogsSummaryPageHelpers(unittest.TestCase):
    def test_normalize_summary_text_drops_none_like_values(self) -> None:
        self.assertEqual(_normalize_summary_text(None), "")
        self.assertEqual(_normalize_summary_text("None"), "")
        self.assertEqual(_normalize_summary_text(" null "), "")
        self.assertEqual(_normalize_summary_text("NaN"), "")

    def test_normalize_summary_text_keeps_regular_text(self) -> None:
        text = "Причина инцидента подтверждена логами."
        self.assertEqual(_normalize_summary_text(text), text)

    def test_enrich_stats_with_elapsed_adds_duration(self) -> None:
        state = {
            "elapsed_seconds": 12.34,
            "stats": {"pages_fetched": 3, "rows_processed": 100},
        }
        _enrich_stats_with_elapsed(state)
        self.assertIn("logs_processing_seconds", state["stats"])
        self.assertIn("logs_processing_human", state["stats"])
        self.assertEqual(state["stats"]["logs_processing_seconds"], 12.34)

    def test_ui_freeform_prompt_uses_custom_template(self) -> None:
        with patch.object(
            settings,
            "CONTROL_PLANE_LLM_UI_FINAL_REPORT_PROMPT_TEMPLATE",
            "CUSTOM UI FREEFORM | {period_start} -> {period_end} | {final_summary}",
        ):
            prompt = _build_freeform_summary_prompt(
                final_summary="summary body",
                user_goal="incident context",
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
                stats={},
                metrics_context="",
            )
        self.assertIn("CUSTOM UI FREEFORM", prompt)
        self.assertIn("summary body", prompt)
        self.assertIn("ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЦЕПОЧКА СОБЫТИЙ", prompt)

    def test_no_logs_hypothesis_prompt_includes_period_and_goal(self) -> None:
        prompt = _build_no_logs_hypothesis_prompt(
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            user_goal="Алерт по росту 5xx на API",
            metrics_context="CPU=92%",
            logs_fetch_mode="time_window",
            logs_tail_limit=1000,
            logs_queries_count=2,
        )
        self.assertIn("Период анализа", prompt)
        self.assertIn("Алерт по росту 5xx", prompt)
        self.assertIn("SQL источников логов: 2", prompt)

    def test_save_logs_summary_result_writes_both_structured_and_freeform(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            saved = _save_logs_summary_result(
                output_dir=Path(tmp_dir),
                request_payload={"x": 1},
                result_state={
                    "status": "done",
                    "mode": "db",
                    "period_start": "2026-03-18T00:00:00Z",
                    "period_end": "2026-03-18T01:00:00Z",
                    "final_summary": "structured summary",
                    "freeform_final_summary": "freeform summary",
                    "logs_processed": 10,
                    "logs_total": 10,
                    "stats": {"llm_calls": 2},
                    "error": None,
                },
            )
            md_path = Path(saved["summary_path"])
            self.assertTrue(md_path.exists())
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Final Summary (Structured)", md_text)
            self.assertIn("structured summary", md_text)
            self.assertIn("Final Summary (Freeform)", md_text)
            self.assertIn("freeform summary", md_text)
            self.assertIn("structured_md_path", saved)
            self.assertIn("freeform_md_path", saved)
            self.assertTrue(Path(saved["structured_md_path"]).exists())
            self.assertTrue(Path(saved["freeform_md_path"]).exists())
            self.assertIn("structured_txt_path", saved)
            self.assertIn("freeform_txt_path", saved)

    def test_load_map_summaries_from_jsonl_reads_batch_summary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "map_summaries.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"batch_summary": "one"}',
                        '{"batch_summary": "two"}',
                    ]
                ),
                encoding="utf-8",
            )
            items = _load_map_summaries_from_jsonl(str(p))
            self.assertEqual(items, ["one", "two"])

    def test_checkpoint_payload_has_state(self) -> None:
        payload = _checkpoint_payload_from_state(
            {
                "status": "map",
                "logs_processed": 123,
                "resume_rows_offset": 77,
                "resume_batch_offset": 5,
                "resume_stats_offset": {"rows_processed": 77},
                "eta_seconds_left": 99,
                "log_seconds_per_second": 12.5,
                "events": ["x"],
            }
        )
        self.assertIn("saved_at", payload)
        self.assertEqual(payload["state"]["status"], "map")
        self.assertEqual(payload["state"]["logs_processed"], 123)
        self.assertEqual(payload["state"]["resume_rows_offset"], 77)
        self.assertEqual(payload["state"]["resume_batch_offset"], 5)
        self.assertEqual(payload["state"]["resume_stats_offset"]["rows_processed"], 77)
        self.assertEqual(payload["state"]["eta_seconds_left"], 99)
        self.assertEqual(payload["state"]["log_seconds_per_second"], 12.5)

    def test_discover_resume_sessions_reads_run_params(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            run_dir = base / "logs_summary_live" / "run_20260330_000000_000000"
            _write_json_file(run_dir / "run_params.json", {"logs_queries": ["SELECT 1"]})
            _write_json_file(
                run_dir / "checkpoint.json",
                {"saved_at": "2026-03-30T00:00:00+00:00", "state": {"status": "map"}},
            )
            sessions = _discover_resume_sessions(base)
            self.assertEqual(len(sessions), 1)
            self.assertIn("run_20260330_000000_000000", sessions[0]["id"])

    def test_discover_resume_sessions_ignores_runs_without_run_params(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            run_dir = base / "logs_summary_live" / "run_20260330_000000_000000"
            _write_json_file(
                run_dir / "checkpoint.json",
                {"saved_at": "2026-03-30T00:00:00+00:00", "state": {"status": "map"}},
            )
            sessions = _discover_resume_sessions(base)
            self.assertEqual(sessions, [])

    def test_discover_resume_sessions_sorted_desc(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            run_old = base / "logs_summary_live" / "run_20260330_000000_000000"
            run_new = base / "logs_summary_live" / "run_20260330_235959_999999"
            _write_json_file(run_old / "run_params.json", {"logs_queries": ["SELECT 1"]})
            _write_json_file(run_new / "run_params.json", {"logs_queries": ["SELECT 2"]})
            _write_json_file(run_old / "checkpoint.json", {"state": {"status": "error"}})
            _write_json_file(run_new / "checkpoint.json", {"state": {"status": "map"}})

            sessions = _discover_resume_sessions(base)
            self.assertEqual(len(sessions), 2)
            self.assertEqual(sessions[0]["id"], "run_20260330_235959_999999")
            self.assertEqual(sessions[1]["id"], "run_20260330_000000_000000")

    def test_load_map_summaries_from_jsonl_ignores_invalid_rows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "map_summaries.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"batch_summary": "one"}',
                        "not a json line",
                        '{"batch_summary": "None"}',
                        '{"other": "value"}',
                        '{"batch_summary": "two"}',
                    ]
                ),
                encoding="utf-8",
            )
            items = _load_map_summaries_from_jsonl(str(p))
            self.assertEqual(items, ["one", "two"])

    def test_save_logs_summary_result_includes_resume_artifact_paths_in_md(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            saved = _save_logs_summary_result(
                output_dir=Path(tmp_dir),
                request_payload={"x": 1},
                result_state={
                    "status": "done",
                    "mode": "db",
                    "period_start": "2026-03-18T00:00:00Z",
                    "period_end": "2026-03-18T01:00:00Z",
                    "final_summary": "structured summary",
                    "freeform_final_summary": "freeform summary",
                    "logs_processed": 10,
                    "logs_total": 10,
                    "stats": {"llm_calls": 2},
                    "error": None,
                    "map_summaries_jsonl_path": "/tmp/map_summaries.jsonl",
                    "reduce_summaries_jsonl_path": "/tmp/reduce_summaries.jsonl",
                    "llm_calls_jsonl_path": "/tmp/llm_calls.jsonl",
                    "run_params_path": "/tmp/run_params.json",
                    "request_path": "/tmp/request.json",
                    "checkpoint_path": "/tmp/checkpoint.json",
                },
            )
            md_text = Path(saved["summary_path"]).read_text(encoding="utf-8")
            self.assertIn("map_summaries_jsonl", md_text)
            self.assertIn("reduce_summaries_jsonl", md_text)
            self.assertIn("llm_calls_jsonl", md_text)
            self.assertIn("run_params_path", md_text)
            self.assertIn("checkpoint_path", md_text)
            self.assertIn("structured md", md_text.lower())
            self.assertIn("freeform md", md_text.lower())

    def test_extract_last_batch_ts_from_run_dir_uses_latest_timestamp(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run_20260330_000000_000000"
            path = run_dir / "summaries" / "map_summaries.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        '{"batch_period_end":"2026-03-30T10:00:00+00:00"}',
                        '{"batch_period_end":"2026-03-30T10:05:00+00:00"}',
                        '{"batch_period_end":"2026-03-30T10:03:00+00:00"}',
                    ]
                ),
                encoding="utf-8",
            )
            ts = _extract_last_batch_ts_from_run_dir(run_dir)
            self.assertTrue(ts.startswith("2026-03-30T10:05:00"))

    def test_extract_last_batch_ts_from_run_dir_returns_empty_when_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run_20260330_000000_000000"
            run_dir.mkdir(parents=True, exist_ok=True)
            ts = _extract_last_batch_ts_from_run_dir(run_dir)
            self.assertEqual(ts, "")

    def test_extract_last_batch_ts_from_run_dir_uses_batches_jsonl_fallback(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run_20260330_000000_000000"
            run_dir.mkdir(parents=True, exist_ok=True)
            batches = run_dir / "batches.jsonl"
            batches.write_text(
                "\n".join(
                    [
                        '{"event":"map_batch","batch_period_end":"2026-03-30T09:59:59+00:00"}',
                        '{"event":"map_batch","batch_period_end":"2026-03-30T10:00:01+00:00"}',
                    ]
                ),
                encoding="utf-8",
            )
            ts = _extract_last_batch_ts_from_run_dir(run_dir)
            self.assertTrue(ts.startswith("2026-03-30T10:00:01"))

    def test_form_values_from_saved_params_maps_values(self) -> None:
        mapped = _form_values_from_saved_params(
            saved_params={
                "logs_queries": ["SELECT 1"],
                "metrics_queries": ["SELECT 2"],
                "user_goal": "goal",
                "period_mode": "Окно вокруг даты (±N минут)",
                "window_minutes": 45,
                "center_dt_text": "2026-03-30T10:00:00+03:00",
                "start_dt_text": "2026-03-30T09:00:00+03:00",
                "end_dt_text": "2026-03-30T11:00:00+03:00",
                "db_batch_size": 1234,
                "llm_batch_size": 321,
                "map_workers": 4,
                "max_retries": 7,
                "llm_timeout": 111,
                "demo_mode": True,
                "demo_logs_count": 2222,
                "enable_no_logs_hypothesis": True,
            },
            default_query="SELECT default",
        )
        self.assertEqual(mapped["logs_queries"], ["SELECT 1"])
        self.assertEqual(mapped["metrics_queries"], ["SELECT 2"])
        self.assertEqual(mapped["logs_sum_user_goal"], "goal")
        self.assertEqual(mapped["logs_sum_window_minutes"], 45)
        self.assertTrue(mapped["logs_sum_parallel_map"])
        self.assertEqual(mapped["logs_sum_map_workers"], 4)
        self.assertTrue(mapped["logs_sum_demo_mode"])
        self.assertTrue(mapped["logs_sum_enable_no_logs_hypothesis"])

    def test_form_values_from_saved_params_uses_default_query_when_empty(self) -> None:
        mapped = _form_values_from_saved_params(
            saved_params={"logs_queries": []},
            default_query="SELECT fallback_query",
        )
        self.assertEqual(mapped["logs_queries"], ["SELECT fallback_query"])


if __name__ == "__main__":
    unittest.main()
