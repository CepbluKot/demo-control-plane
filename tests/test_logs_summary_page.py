import unittest
import zipfile
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from settings import settings
from ui.pages.logs_summary_page import (
    FINAL_REPORT_SECTIONS,
    PORTABLE_BUNDLE_TYPE,
    _build_portable_report_bundle,
    _build_saved_params_from_import_request,
    _build_models_url,
    _build_report_generation_progress_snapshot,
    _build_config,
    _extract_root_cause_hypotheses_block,
    _ensure_report_topics_present,
    _extract_request_result_from_bundle,
    _extract_model_ids_from_payload,
    _fetch_llm_model_candidates,
    _format_table_timestamps,
    _build_stage1_progress_snapshot,
    _checkpoint_payload_from_state,
    _build_no_logs_hypothesis_prompt,
    _build_freeform_summary_prompt,
    _build_causal_graph_dot,
    _default_alert_time_values,
    _build_sectional_freeform_prompt,
    _build_sectional_structured_prompt,
    _build_zip_artifacts_bytes,
    _collect_timeline_events,
    _discover_resume_sessions,
    _enrich_stats_with_elapsed,
    _extract_alerts_from_items,
    _extract_last_batch_ts_from_run_dir,
    _form_values_from_saved_params,
    _generate_sectional_freeform_summary,
    _generate_sectional_structured_summary,
    _load_map_summaries_from_jsonl,
    _load_map_summaries_from_jsonl_for_source,
    _normalize_alert_items,
    _normalize_summary_text,
    _render_alerts_context,
    _state_from_imported_result,
    _summary_origin_label,
    _write_json_file,
    _save_logs_summary_result,
    _split_demo_logs_by_source,
)
import pandas as pd


class TestLogsSummaryPageHelpers(unittest.TestCase):
    def test_extract_root_cause_hypotheses_block_from_section(self) -> None:
        text = (
            "1) ХРОНОЛОГИЯ\n"
            "...\n"
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ\n"
            "- Инцидент: A\n"
            "- [ГИПОТЕЗА] первопричина: X\n"
            "2) РЕКОМЕНДАЦИИ\n"
        )
        out = _extract_root_cause_hypotheses_block(text)
        self.assertIn("ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ", out)
        self.assertIn("[ГИПОТЕЗА] первопричина: X", out)
        self.assertNotIn("2) РЕКОМЕНДАЦИИ", out)

    def test_extract_root_cause_hypotheses_block_fallback(self) -> None:
        text = (
            "some text\n"
            "- [ГИПОТЕЗА] возможная первопричина: cache stampede\n"
            "- [ГИПОТЕЗА] возможная первопричина: db locks\n"
        )
        out = _extract_root_cause_hypotheses_block(text)
        self.assertIn("cache stampede", out)
        self.assertIn("db locks", out)

    def test_ensure_report_topics_present_adds_missing_from_preferred_sections(self) -> None:
        summary = "## Полная Хронология Событий\n\nok"
        preferred_sections = [
            {"title": "Рекомендации Для SRE", "text": "P0: restart component"},
        ]
        synced, missing = _ensure_report_topics_present(
            summary,
            topic_titles=["Полная Хронология Событий", "Рекомендации Для SRE"],
            preferred_sections=preferred_sections,
        )
        self.assertIn("## Полная Хронология Событий", synced)
        self.assertIn("## Рекомендации Для SRE", synced)
        self.assertIn("P0: restart component", synced)
        self.assertEqual(missing, ["Рекомендации Для SRE"])

    def test_ensure_report_topics_present_uses_placeholder_without_preferred_text(self) -> None:
        synced, missing = _ensure_report_topics_present(
            "",
            topic_titles=["Разрывы Цепочек И Недостающие Данные"],
            preferred_sections=[],
        )
        self.assertIn("## Разрывы Цепочек И Недостающие Данные", synced)
        self.assertIn("Данных недостаточно", synced)
        self.assertEqual(missing, ["Разрывы Цепочек И Недостающие Данные"])

    def test_format_table_timestamps_formats_start_end_time_with_microseconds(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "start_time": "2026-03-18T00:00:00.123456Z",
                    "end_time": "2026-03-18T00:00:01.987654Z",
                }
            ]
        )
        formatted = _format_table_timestamps(df)
        self.assertEqual(
            str(formatted.loc[0, "start_time"]),
            "2026-03-18 03:00:00.123456 MSK",
        )
        self.assertEqual(
            str(formatted.loc[0, "end_time"]),
            "2026-03-18 03:00:01.987654 MSK",
        )

    def test_split_demo_logs_by_source_partitions_rows_without_loss(self) -> None:
        demo_logs = [
            {"timestamp": "2026-03-18T00:00:00Z", "message": "m1"},
            {"timestamp": "2026-03-18T00:00:01Z", "message": "m2"},
            {"timestamp": "2026-03-18T00:00:02Z", "message": "m3"},
            {"timestamp": "2026-03-18T00:00:03Z", "message": "m4"},
            {"timestamp": "2026-03-18T00:00:04Z", "message": "m5"},
        ]
        grouped = _split_demo_logs_by_source(demo_logs, ["query_1", "query_2"])
        self.assertEqual(sorted(grouped.keys()), ["query_1", "query_2"])
        self.assertEqual(len(grouped["query_1"]), 3)
        self.assertEqual(len(grouped["query_2"]), 2)
        merged_messages = [row["message"] for row in grouped["query_1"] + grouped["query_2"]]
        self.assertEqual(sorted(merged_messages), ["m1", "m2", "m3", "m4", "m5"])

    def test_split_demo_logs_by_source_single_label_keeps_all_rows(self) -> None:
        demo_logs = [
            {"timestamp": "2026-03-18T00:00:00Z", "message": "m1"},
            {"timestamp": "2026-03-18T00:00:01Z", "message": "m2"},
        ]
        grouped = _split_demo_logs_by_source(demo_logs, ["query_1"])
        self.assertEqual(list(grouped.keys()), ["query_1"])
        self.assertEqual([row["message"] for row in grouped["query_1"]], ["m1", "m2"])

    def test_summary_origin_label_maps_known_and_unknown_values(self) -> None:
        self.assertEqual(
            _summary_origin_label("resume_rereduce"),
            "Resume: пересборка REDUCE из сохранённых MAP summary",
        )
        self.assertEqual(_summary_origin_label("custom_origin"), "custom_origin")
        self.assertEqual(_summary_origin_label(""), "")

    def test_normalize_summary_text_drops_none_like_values(self) -> None:
        self.assertEqual(_normalize_summary_text(None), "")
        self.assertEqual(_normalize_summary_text("None"), "")
        self.assertEqual(_normalize_summary_text(" null "), "")
        self.assertEqual(_normalize_summary_text("NaN"), "")

    def test_normalize_summary_text_keeps_regular_text(self) -> None:
        text = "Причина инцидента подтверждена логами."
        self.assertEqual(_normalize_summary_text(text), text)

    def test_normalize_summary_text_rewrites_legacy_heuristic_fallback(self) -> None:
        legacy = (
            "[LLM НЕДОСТУПНА — эвристический fallback]\n\n"
            "ОШИБКА: 400 Client Error: Bad Request\n\n"
            "ХРОНОЛОГИЯ: данных нет."
        )
        out = _normalize_summary_text(legacy)
        self.assertIn("[LLM ERROR]", out)
        self.assertIn("ОШИБКА: 400 Client Error: Bad Request", out)
        self.assertNotIn("эвристический fallback", out.lower())

    def test_enrich_stats_with_elapsed_adds_duration(self) -> None:
        state = {
            "elapsed_seconds": 12.34,
            "stats": {"pages_fetched": 3, "rows_processed": 100},
        }
        _enrich_stats_with_elapsed(state)
        self.assertIn("logs_processing_seconds", state["stats"])
        self.assertIn("logs_processing_human", state["stats"])
        self.assertEqual(state["stats"]["logs_processing_seconds"], 12.34)

    def test_build_stage1_progress_snapshot_uses_rows_when_total_known(self) -> None:
        snapshot = _build_stage1_progress_snapshot(
            {
                "status": "map",
                "logs_processed": 250,
                "logs_total": 1000,
                "elapsed_seconds": 10,
                "eta_seconds_left": 30,
                "eta_finish_at": "2026-03-18T10:00:00+03:00",
            }
        )
        self.assertTrue(snapshot["show"])
        self.assertAlmostEqual(float(snapshot["ratio"]), 0.25, places=6)
        self.assertIn("250/1,000", snapshot["label"])
        self.assertIn("elapsed:", snapshot["runtime_line"])
        self.assertIn("rate:", snapshot["runtime_line"])
        self.assertIn("eta:", snapshot["runtime_line"])
        self.assertIn("finish:", snapshot["runtime_line"])

    def test_build_stage1_progress_snapshot_uses_timestamp_coverage_without_rows_total(self) -> None:
        snapshot = _build_stage1_progress_snapshot(
            {
                "status": "map",
                "period_start": "2026-03-18T00:00:00+03:00",
                "period_end": "2026-03-18T01:00:00+03:00",
                "last_batch_ts": "2026-03-18T00:30:00+03:00",
            }
        )
        self.assertTrue(snapshot["show"])
        self.assertAlmostEqual(float(snapshot["ratio"]), 0.5, places=6)
        self.assertIn("покрытие периода 50.0%", snapshot["label"])

    def test_build_stage1_progress_snapshot_uses_non_truncated_done_batches_counter(self) -> None:
        snapshot = _build_stage1_progress_snapshot(
            {
                "status": "map",
                "estimated_batch_total": 120,
                "map_batches_done_total": 75,
                "map_batches": [{"batch_index": i} for i in range(10)],  # UI keeps only tail
            }
        )
        self.assertTrue(snapshot["show"])
        self.assertAlmostEqual(float(snapshot["ratio"]), 75.0 / 120.0, places=6)
        self.assertIn("батчи 75/120", snapshot["label"])

    def test_build_stage1_progress_snapshot_hidden_outside_first_stage(self) -> None:
        snapshot = _build_stage1_progress_snapshot({"status": "reduce"})
        self.assertFalse(snapshot["show"])

    def test_build_report_generation_progress_snapshot_hidden_when_not_started(self) -> None:
        snapshot = _build_report_generation_progress_snapshot({})
        self.assertFalse(snapshot["show"])

    def test_build_report_generation_progress_snapshot_builds_ratio_and_label(self) -> None:
        snapshot = _build_report_generation_progress_snapshot(
            {
                "report_progress_current": 5,
                "report_progress_total": 20,
                "report_progress_label": "Свободный отчёт: секция 5/13 готова",
                "report_progress_active": True,
            }
        )
        self.assertTrue(snapshot["show"])
        self.assertAlmostEqual(float(snapshot["ratio"]), 0.25, places=6)
        self.assertIn("(5/20)", snapshot["label"])

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
        self.assertIn(
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ",
            prompt,
        )

    def test_ui_freeform_prompt_default_includes_map_summaries(self) -> None:
        prompt = _build_freeform_summary_prompt(
            final_summary="reduce summary",
            map_summaries_text="[MAP SUMMARY #1]\nfirst batch",
            user_goal="incident context",
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={},
            metrics_context="",
        )
        self.assertIn("MAP summary по батчам логов", prompt)
        self.assertIn("[MAP SUMMARY #1]", prompt)
        self.assertIn("first batch", prompt)
        self.assertIn("ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)", prompt)
        self.assertIn("incident context", prompt)

    def test_ui_freeform_prompt_custom_template_receives_map_summaries(self) -> None:
        with patch.object(
            settings,
            "CONTROL_PLANE_LLM_UI_FINAL_REPORT_PROMPT_TEMPLATE",
            "MAPS={map_summaries_text} | FINAL={final_summary}",
        ):
            prompt = _build_freeform_summary_prompt(
                final_summary="reduce summary",
                map_summaries_text="[MAP SUMMARY #1]\nfirst batch",
                user_goal="goal",
                period_start="2026-03-18T00:00:00Z",
                period_end="2026-03-18T01:00:00Z",
                stats={},
                metrics_context="",
            )
        self.assertIn("MAPS=[MAP SUMMARY #1]", prompt)
        self.assertIn("FINAL=reduce summary", prompt)

    def test_sectional_prompt_uses_previous_sections_context(self) -> None:
        prompt = _build_sectional_freeform_prompt(
            section_index=2,
            section_total=3,
            section_title="Причины",
            section_requirement="Опиши причины",
            previous_sections_text="## Вводная\nТекст первой секции",
            final_summary="Structured reduce summary",
            user_goal="incident goal",
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={"llm_calls": 3},
            metrics_context="CPU=95%",
        )
        self.assertIn("СЕКЦИЯ 2/3: Причины", prompt)
        self.assertIn("## Вводная", prompt)
        self.assertIn("Structured reduce summary", prompt)
        self.assertNotIn("ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)", prompt)

    def test_sectional_freeform_root_cause_section_requires_verbatim_incident(self) -> None:
        incident_text = "INCIDENT_A: node restart at 18:52"
        prompt = _build_sectional_freeform_prompt(
            section_index=4,
            section_total=10,
            section_title="Предположения О Первопричине По Каждому Инциденту",
            section_requirement="Для каждого инцидента опиши первопричины",
            previous_sections_text="## Previous\nDone",
            final_summary="Structured reduce summary",
            user_goal=incident_text,
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={"llm_calls": 3},
            metrics_context="CPU=95%",
        )
        self.assertIn("ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)", prompt)
        self.assertIn(incident_text, prompt)

    def test_generate_sectional_freeform_summary_merges_sections_and_chains_context(self) -> None:
        prompts: list[str] = []

        def fake_llm(prompt: str) -> str:
            prompts.append(prompt)
            return f"section_body_{len(prompts)}"

        merged, sections = _generate_sectional_freeform_summary(
            llm_call=fake_llm,
            final_summary="Structured summary",
            user_goal="incident goal",
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={},
            metrics_context="",
        )
        self.assertEqual(len(sections), len(FINAL_REPORT_SECTIONS))
        # 2 секции заполняются программно без LLM:
        # 1) Контекст из UI (дословно), 2) Метрики (когда metrics_context пустой).
        self.assertEqual(len(prompts), len(FINAL_REPORT_SECTIONS) - 2)
        self.assertIn(f"## {FINAL_REPORT_SECTIONS[0][0]}", merged)
        self.assertIn("исходный текст инцидента из ui (дословно", merged.lower())
        self.assertIn("incident goal", merged)
        self.assertIn("Метрики не предоставлены", merged)
        self.assertIn(f"## {FINAL_REPORT_SECTIONS[0][0]}", prompts[0])
        self.assertIn("incident goal", prompts[0])
        self.assertIn("section_body_1", prompts[1])

    def test_sectional_structured_prompt_uses_previous_sections_context(self) -> None:
        prompt = _build_sectional_structured_prompt(
            section_index=2,
            section_total=3,
            section_title="Причины",
            section_requirement="Опиши причины",
            previous_sections_text="## Вводная\nТекст первой секции",
            base_summary="Base reduce summary",
            user_goal="incident goal",
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={"llm_calls": 3},
            metrics_context="CPU=95%",
        )
        self.assertIn("СЕКЦИЯ 2/3: Причины", prompt)
        self.assertIn("## Вводная", prompt)
        self.assertIn("Base reduce summary", prompt)
        self.assertNotIn("ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)", prompt)

    def test_sectional_structured_root_cause_section_requires_verbatim_incident(self) -> None:
        incident_text = "INCIDENT_B: fs > 90% at 19:10"
        prompt = _build_sectional_structured_prompt(
            section_index=4,
            section_total=10,
            section_title="Предположения О Первопричине По Каждому Инциденту",
            section_requirement="Опиши первопричины по каждому инциденту",
            previous_sections_text="## Previous\nDone",
            base_summary="Base reduce summary",
            user_goal=incident_text,
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={"llm_calls": 3},
            metrics_context="CPU=95%",
        )
        self.assertIn("ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)", prompt)
        self.assertIn(incident_text, prompt)

    def test_generate_sectional_structured_summary_merges_sections_and_chains_context(self) -> None:
        prompts: list[str] = []

        def fake_llm(prompt: str) -> str:
            prompts.append(prompt)
            return f"struct_body_{len(prompts)}"

        merged, sections = _generate_sectional_structured_summary(
            llm_call=fake_llm,
            base_summary="Structured reduce summary",
            user_goal="incident goal",
            period_start="2026-03-18T00:00:00Z",
            period_end="2026-03-18T01:00:00Z",
            stats={},
            metrics_context="",
        )
        self.assertEqual(len(sections), len(FINAL_REPORT_SECTIONS))
        self.assertEqual(len(prompts), len(FINAL_REPORT_SECTIONS) - 2)
        self.assertIn(f"## {FINAL_REPORT_SECTIONS[0][0]}", merged)
        self.assertIn("исходный текст инцидента из ui (дословно", merged.lower())
        self.assertIn("incident goal", merged)
        self.assertIn("Метрики не предоставлены", merged)
        self.assertIn(f"## {FINAL_REPORT_SECTIONS[0][0]}", prompts[0])
        self.assertIn("incident goal", prompts[0])
        self.assertIn("struct_body_1", prompts[1])

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
        self.assertIn("ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ", prompt)

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
                    "final_summary_origin": "manual_rereduce",
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
            self.assertIn("summary_origin: `manual_rereduce`", md_text)
            self.assertIn("structured_md_path", saved)
            self.assertIn("freeform_md_path", saved)
            self.assertTrue(Path(saved["structured_md_path"]).exists())
            self.assertTrue(Path(saved["freeform_md_path"]).exists())
            self.assertIn("html_path", saved)
            html_path = Path(saved["html_path"])
            self.assertTrue(html_path.exists())
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", html_text.lower())
            self.assertIn("Итоговый Отчёт По Саммаризации Логов", html_text)
            self.assertIn("tab-timeline", html_text)
            self.assertIn("timelineRows", html_text)
            self.assertIn("bundle_path", saved)
            bundle_path = Path(saved["bundle_path"])
            self.assertTrue(bundle_path.exists())
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle.get("bundle_type"), PORTABLE_BUNDLE_TYPE)
            self.assertIn("request", bundle)
            self.assertIn("result", bundle)
            self.assertIn("structured_txt_path", saved)
            self.assertIn("freeform_txt_path", saved)

    def test_save_logs_summary_result_writes_instructor_reports_when_present(self) -> None:
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
                    "instructor_structured_summary": "instructor structured summary",
                    "instructor_freeform_summary": "instructor freeform summary",
                    "logs_processed": 10,
                    "logs_total": 10,
                    "stats": {"llm_calls": 2},
                    "error": None,
                },
            )
            self.assertIn("instructor_structured_md_path", saved)
            self.assertIn("instructor_freeform_md_path", saved)
            self.assertIn("instructor_structured_txt_path", saved)
            self.assertIn("instructor_freeform_txt_path", saved)
            self.assertTrue(Path(saved["instructor_structured_md_path"]).exists())
            self.assertTrue(Path(saved["instructor_freeform_md_path"]).exists())
            md_text = Path(saved["summary_path"]).read_text(encoding="utf-8")
            self.assertIn("Final Summary (Instructor Structured)", md_text)
            self.assertIn("instructor structured summary", md_text)
            self.assertIn("Final Summary (Instructor Freeform)", md_text)
            self.assertIn("instructor freeform summary", md_text)

    def test_extract_request_result_from_bundle_supports_portable_and_legacy_json(self) -> None:
        portable = {
            "bundle_type": PORTABLE_BUNDLE_TYPE,
            "request": {"sql_query": "SELECT 1"},
            "result": {"final_summary": "ok"},
        }
        request, result = _extract_request_result_from_bundle(portable)
        self.assertEqual(request.get("sql_query"), "SELECT 1")
        self.assertEqual(result.get("final_summary"), "ok")

        legacy = {
            "saved_at": "2026-04-01T00:00:00+03:00",
            "request": {"sql_query": "SELECT 2"},
            "result": {"final_summary": "legacy"},
        }
        request_legacy, result_legacy = _extract_request_result_from_bundle(legacy)
        self.assertEqual(request_legacy.get("sql_query"), "SELECT 2")
        self.assertEqual(result_legacy.get("final_summary"), "legacy")

    def test_extract_request_result_from_bundle_invalid_returns_empty(self) -> None:
        request, result = _extract_request_result_from_bundle({"foo": "bar"})
        self.assertEqual(request, {})
        self.assertEqual(result, {})

    def test_build_portable_report_bundle_contains_schema_and_payload(self) -> None:
        bundle = _build_portable_report_bundle(
            request_payload={"sql_query": "SELECT 1"},
            result_state={"final_summary": "ok"},
        )
        self.assertEqual(bundle.get("bundle_type"), PORTABLE_BUNDLE_TYPE)
        self.assertEqual(bundle.get("bundle_version"), 1)
        self.assertEqual(bundle.get("request", {}).get("sql_query"), "SELECT 1")
        self.assertEqual(bundle.get("result", {}).get("final_summary"), "ok")

    def test_state_from_imported_result_clears_local_paths(self) -> None:
        imported = _state_from_imported_result(
            {
                "status": "done",
                "result_json_path": "/tmp/a.json",
                "result_bundle_path": "/tmp/a.bundle.json",
                "result_summary_path": "/tmp/a.md",
                "result_html_path": "/tmp/a.html",
                "final_summary": "summary",
            }
        )
        self.assertEqual(imported.get("status"), "done")
        self.assertIsNone(imported.get("result_json_path"))
        self.assertIsNone(imported.get("result_bundle_path"))
        self.assertIsNone(imported.get("result_summary_path"))
        self.assertIsNone(imported.get("result_html_path"))
        self.assertEqual(imported.get("final_summary"), "summary")
        self.assertTrue(imported.get("final_report_ready"))

    def test_build_saved_params_from_import_request_prefills_form_fields(self) -> None:
        saved_params = _build_saved_params_from_import_request(
            request_payload={
                "logs_queries": ["SELECT * FROM logs_a"],
                "metrics_queries": ["SELECT * FROM metrics_a"],
                "alerts": [{"title": "a1"}],
                "user_goal": "incident goal",
                "period_mode": "window",
                "period_start": "2026-03-18T10:00:00+03:00",
                "period_end": "2026-03-18T11:00:00+03:00",
                "window_minutes": 45,
                "db_batch_size": 1200,
                "llm_batch_size": 300,
                "llm_model_id": "model-x",
                "use_instructor": False,
                "model_supports_tool_calling": False,
                "enable_no_logs_hypothesis": True,
            },
            center_default="2026-03-01T00:00:00+03:00",
            start_default="2026-03-01T00:00:00+03:00",
            end_default="2026-03-01T01:00:00+03:00",
            default_query="SELECT default_query",
        )
        self.assertEqual(saved_params["logs_queries"], ["SELECT * FROM logs_a"])
        self.assertEqual(saved_params["metrics_queries"], ["SELECT * FROM metrics_a"])
        self.assertEqual(saved_params["period_mode"], "Окно вокруг даты (±N минут)")
        self.assertEqual(saved_params["db_batch_size"], 1200)
        self.assertEqual(saved_params["llm_batch_size"], 300)
        self.assertEqual(saved_params["llm_model_id"], "model-x")
        self.assertFalse(saved_params["use_instructor"])
        self.assertFalse(saved_params["model_supports_tool_calling"])
        self.assertTrue(saved_params["enable_no_logs_hypothesis"])

    def test_build_saved_params_from_import_request_legacy_sql_fields(self) -> None:
        saved_params = _build_saved_params_from_import_request(
            request_payload={
                "sql_query": "SELECT * FROM logs_legacy",
                "metrics_query": "SELECT * FROM metrics_legacy",
                "period_mode": "start_end",
                "period_start": "2026-03-18T10:00:00+03:00",
                "period_end": "2026-03-18T11:00:00+03:00",
            },
            center_default="2026-03-01T00:00:00+03:00",
            start_default="2026-03-01T00:00:00+03:00",
            end_default="2026-03-01T01:00:00+03:00",
            default_query="SELECT default_query",
        )
        self.assertEqual(saved_params["logs_queries"], ["SELECT * FROM logs_legacy"])
        self.assertEqual(saved_params["metrics_queries"], ["SELECT * FROM metrics_legacy"])
        self.assertEqual(saved_params["period_mode"], "Явный диапазон (start/end)")

    def test_save_logs_summary_result_does_not_truncate_structured_text(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            long_summary = "x" * 20000
            saved = _save_logs_summary_result(
                output_dir=Path(tmp_dir),
                request_payload={"x": 1},
                result_state={
                    "status": "done",
                    "mode": "db",
                    "period_start": "2026-03-18T00:00:00Z",
                    "period_end": "2026-03-18T01:00:00Z",
                    "final_summary": long_summary,
                    "freeform_final_summary": "freeform summary",
                    "logs_processed": 10,
                    "logs_total": 10,
                    "stats": {"llm_calls": 2},
                    "error": None,
                },
            )
            structured_txt = Path(saved["structured_txt_path"]).read_text(encoding="utf-8")
            self.assertEqual(structured_txt, long_summary)
            structured_md = Path(saved["structured_md_path"]).read_text(encoding="utf-8")
            self.assertIn(long_summary, structured_md)

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

    def test_load_map_summaries_from_jsonl_for_source_filters_by_source(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "map_summaries.jsonl"
            p.write_text(
                "\n".join(
                    [
                        '{"source_name":"query_1","batch_summary":"q1-1"}',
                        '{"source_name":"query_2","batch_summary":"q2-1"}',
                        '{"source_name":"query_1","batch_summary":"q1-2"}',
                        '{"source_name":"query_1","batch_summary":"None"}',
                        "not json",
                    ]
                ),
                encoding="utf-8",
            )
            q1_items = _load_map_summaries_from_jsonl_for_source(str(p), "query_1")
            q2_items = _load_map_summaries_from_jsonl_for_source(str(p), "query_2")
            self.assertEqual(q1_items, ["q1-1", "q1-2"])
            self.assertEqual(q2_items, ["q2-1"])

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
                "final_summary_origin": "manual_rereduce",
                "instructor_report_status": "done",
                "instructor_structured_summary": "s2",
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
        self.assertEqual(payload["state"]["final_summary_origin"], "manual_rereduce")
        self.assertEqual(payload["state"]["instructor_report_status"], "done")
        self.assertEqual(payload["state"]["instructor_structured_summary"], "s2")

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
            self.assertTrue(ts.startswith("2026-03-30T13:05:00+03:00"))

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
            self.assertTrue(ts.startswith("2026-03-30T13:00:01+03:00"))

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
                "use_instructor": False,
                "model_supports_tool_calling": False,
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
        self.assertIn("legacy_alert_context", mapped["logs_sum_user_goal"])
        self.assertIn("описание: goal", mapped["logs_sum_user_goal"])
        self.assertTrue(isinstance(mapped.get("alerts"), list))
        self.assertGreaterEqual(len(mapped["alerts"]), 1)
        self.assertEqual(mapped["logs_sum_window_minutes"], 45)
        self.assertEqual(mapped["logs_sum_llm_batch"], 321)
        self.assertFalse(mapped["logs_sum_use_instructor"])
        self.assertFalse(mapped["logs_sum_model_supports_tool_calling"])
        self.assertFalse(mapped["logs_sum_parallel_map"])
        self.assertEqual(mapped["logs_sum_map_workers"], 1)
        self.assertTrue(mapped["logs_sum_demo_mode"])
        self.assertTrue(mapped["logs_sum_enable_no_logs_hypothesis"])

    def test_form_values_from_saved_params_uses_default_query_when_empty(self) -> None:
        mapped = _form_values_from_saved_params(
            saved_params={"logs_queries": []},
            default_query="SELECT fallback_query",
        )
        self.assertEqual(mapped["logs_queries"], ["SELECT fallback_query"])

    def test_form_values_from_saved_params_keeps_alert_time_fields_for_resume_buttons(self) -> None:
        saved_params = {
            "logs_queries": ["SELECT 1"],
            "alerts": [
                {
                    "id": "a1",
                    "title": "point_alert",
                    "time_mode": "point",
                    "time_point": "2026-03-18T18:42:00+03:00",
                    "details": "node down",
                },
                {
                    "id": "a2",
                    "title": "range_alert",
                    "time_mode": "range",
                    "time_start": "2026-03-18T18:42:00+03:00",
                    "time_end": "2026-03-18T19:10:00+03:00",
                    "details": "fs low",
                },
            ],
        }
        mapped = _form_values_from_saved_params(
            saved_params=saved_params,
            default_query="SELECT fallback_query",
        )
        alerts = mapped["alerts"]
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["title"], "point_alert")
        self.assertEqual(alerts[0]["time_mode"], "point")
        self.assertEqual(alerts[0]["time_point"], "2026-03-18T18:42:00+03:00")
        self.assertEqual(alerts[1]["title"], "range_alert")
        self.assertEqual(alerts[1]["time_mode"], "range")
        self.assertEqual(alerts[1]["time_start"], "2026-03-18T18:42:00+03:00")
        self.assertEqual(alerts[1]["time_end"], "2026-03-18T19:10:00+03:00")

    def test_extract_alerts_from_items_keeps_point_and_range(self) -> None:
        items = _normalize_alert_items(
            [
                {
                    "title": "alert_point",
                    "time_mode": "point",
                    "time_point": "2026-03-18T18:42:00+03:00",
                    "details": "node down",
                },
                {
                    "title": "alert_range",
                    "time_mode": "range",
                    "time_start": "2026-03-18T18:42:00+03:00",
                    "time_end": "2026-03-18T19:10:00+03:00",
                    "details": "filesystem low",
                },
            ],
            min_items=1,
        )
        alerts = _extract_alerts_from_items(items)
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["time_mode"], "point")
        self.assertEqual(alerts[1]["time_mode"], "range")
        rendered = _render_alerts_context(alerts)
        self.assertIn("alert_point", rendered)
        self.assertIn("alert_range", rendered)
        self.assertIn("период:", rendered)

    def test_default_alert_time_values_prefill_from_request_fields(self) -> None:
        values = _default_alert_time_values(
            center_dt_text="2026-03-27T14:30:00+03:00",
            start_dt_text="2026-03-27T13:30:00+03:00",
            end_dt_text="2026-03-27T15:30:00+03:00",
            center_default="2026-03-01T00:00:00+03:00",
            start_default="2026-03-01T00:00:00+03:00",
            end_default="2026-03-01T01:00:00+03:00",
        )
        self.assertEqual(values["time_point"], "2026-03-27T14:30:00+03:00")
        self.assertEqual(values["time_start"], "2026-03-27T13:30:00+03:00")
        self.assertEqual(values["time_end"], "2026-03-27T15:30:00+03:00")

    def test_default_alert_time_values_uses_defaults_when_fields_empty(self) -> None:
        values = _default_alert_time_values(
            center_dt_text="",
            start_dt_text="",
            end_dt_text="",
            center_default="2026-03-27T14:30:00+03:00",
            start_default="2026-03-27T13:30:00+03:00",
            end_default="2026-03-27T15:30:00+03:00",
        )
        self.assertEqual(values["time_point"], "2026-03-27T14:30:00+03:00")
        self.assertEqual(values["time_start"], "2026-03-27T13:30:00+03:00")
        self.assertEqual(values["time_end"], "2026-03-27T15:30:00+03:00")

    def test_build_models_url_from_chat_completions(self) -> None:
        url = _build_models_url("https://phoenix.example/api/v1/chat/completions")
        self.assertEqual(url, "https://phoenix.example/api/v1/models")

    def test_extract_model_ids_from_payload(self) -> None:
        payload = {
            "data": [
                {"id": "model-a"},
                {"name": "model-b"},
                {"model": "model-c"},
                {"id": "model-a"},
            ]
        }
        self.assertEqual(
            _extract_model_ids_from_payload(payload),
            ["model-a", "model-b", "model-c"],
        )

    def test_fetch_llm_model_candidates_from_api(self) -> None:
        class _Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            @staticmethod
            def json():
                return {"data": [{"id": "m1"}, {"id": "m2"}]}

        captured: Dict[str, Any] = {}

        def _fake_get(url: str, *, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            captured["timeout"] = timeout
            return _Resp()

        with patch.object(settings, "OPENAI_API_BASE_DB", "https://phoenix.example/api/v1/chat/completions"), patch.object(
            settings, "OPENAI_API_KEY_DB", "token-123"
        ), patch("ui.pages.logs_summary_page.requests.get", side_effect=_fake_get):
            models, err = _fetch_llm_model_candidates(timeout_seconds=7.0)

        self.assertEqual(models, ["m1", "m2"])
        self.assertEqual(err, "")
        self.assertEqual(captured["url"], "https://phoenix.example/api/v1/models")
        self.assertEqual(captured["headers"].get("Authorization"), "Bearer token-123")
        self.assertEqual(float(captured["timeout"]), 7.0)

    def test_build_config_passes_new_algorithm(self) -> None:
        deps = type(
            "_Deps",
            (),
            {
                "summarizer_config_cls": staticmethod(lambda **kwargs: kwargs),
            },
        )()
        overrides = {
            "CONTROL_PLANE_UI_LOGS_SUMMARY_USE_NEW_ALGORITHM": True,
            "CONTROL_PLANE_LLM_USE_NEW_ALGORITHM": True,
            "CONTROL_PLANE_LLM_REDUCE_TARGET_TOKEN_PCT": 44,
            "CONTROL_PLANE_LLM_COMPRESSION_TARGET_PCT": 33,
            "CONTROL_PLANE_LLM_COMPRESSION_IMPORTANCE_THRESHOLD": 0.81,
            "CONTROL_PLANE_LLM_USE_INSTRUCTOR": False,
            "CONTROL_PLANE_LLM_SUPPORTS_TOOL_CALLING": False,
        }
        with patch.multiple(settings, **overrides):
            cfg = _build_config(deps, db_batch_size=1000, llm_batch_size=250, map_workers=3)

        self.assertTrue(cfg["use_new_algorithm"])
        self.assertEqual(cfg["reduce_target_token_pct"], 44)
        self.assertEqual(cfg["compression_target_pct"], 33)
        self.assertAlmostEqual(float(cfg["compression_importance_threshold"]), 0.81, places=6)
        self.assertFalse(cfg["use_instructor"])
        self.assertFalse(cfg["model_supports_tool_calling"])
        self.assertEqual(cfg["page_limit"], 1000)
        self.assertEqual(cfg["llm_chunk_rows"], 250)
        self.assertEqual(cfg["map_workers"], 3)

    def test_build_config_clamps_min_llm_chunk_rows_to_user_cap(self) -> None:
        deps = type(
            "_Deps",
            (),
            {
                "summarizer_config_cls": staticmethod(lambda **kwargs: kwargs),
            },
        )()
        overrides = {
            "CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE": 200,
        }
        with patch.multiple(settings, **overrides):
            cfg = _build_config(deps, db_batch_size=5000, llm_batch_size=120, map_workers=1)

        self.assertEqual(cfg["llm_chunk_rows"], 120)
        # Even if env min is larger, lower bound must not exceed the user cap.
        self.assertEqual(cfg["min_llm_chunk_rows"], 120)

    def test_collect_timeline_events_from_map_batches(self) -> None:
        state = {
            "map_batches": [
                {
                    "batch_index": 0,
                    "batch_period_start": "2026-03-18T00:00:00+00:00",
                    "batch_period_end": "2026-03-18T00:05:00+00:00",
                    "batch_summary_structured": {
                        "timeline": [
                            {
                                "id": "evt-1",
                                "timestamp": "2026-03-18T00:01:00.123456+00:00",
                                "source": "api",
                                "description": "error",
                            }
                        ]
                    },
                }
            ]
        }
        rows = _collect_timeline_events(state)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "evt-1")
        self.assertEqual(rows[0]["batch_index"], 1)

    def test_build_causal_graph_dot(self) -> None:
        dot = _build_causal_graph_dot(
            events=[
                {"id": "evt-1", "timestamp": "t1", "description": "first"},
                {"id": "evt-2", "timestamp": "t2", "description": "second"},
            ],
            links=[
                {
                    "cause_event_id": "evt-1",
                    "effect_event_id": "evt-2",
                    "confidence": 0.9,
                }
            ],
        )
        self.assertTrue(isinstance(dot, str))
        self.assertIn("evt-1", dot)
        self.assertIn("evt-2", dot)
        self.assertIn("conf=0.90", dot)

    def test_build_zip_artifacts_bytes_contains_known_entries(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            report = base / "report.md"
            report_bundle = base / "report.bundle.json"
            report_html = base / "report.html"
            checkpoint = base / "checkpoint.json"
            report.write_text("report", encoding="utf-8")
            report_bundle.write_text('{"bundle_type":"logs_summary_portable_report"}', encoding="utf-8")
            report_html.write_text("<html></html>", encoding="utf-8")
            checkpoint.write_text("{}", encoding="utf-8")
            blob = _build_zip_artifacts_bytes(
                {
                    "result_summary_path": str(report),
                    "result_bundle_path": str(report_bundle),
                    "result_html_path": str(report_html),
                    "checkpoint_path": str(checkpoint),
                }
            )
            self.assertIsNotNone(blob)
            with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
                names = set(zf.namelist())
            self.assertIn("report/report.md", names)
            self.assertIn("report/report.bundle.json", names)
            self.assertIn("report/report.html", names)
            self.assertIn("runtime/checkpoint.json", names)

    def test_build_zip_artifacts_bytes_can_filter_by_prefix(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            report = base / "report.md"
            checkpoint = base / "checkpoint.json"
            report.write_text("report", encoding="utf-8")
            checkpoint.write_text("{}", encoding="utf-8")

            blob = _build_zip_artifacts_bytes(
                {
                    "result_summary_path": str(report),
                    "checkpoint_path": str(checkpoint),
                },
                include_prefixes=("report/",),
            )
            self.assertIsNotNone(blob)
            with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
                names = set(zf.namelist())
            self.assertIn("report/report.md", names)
            self.assertNotIn("runtime/checkpoint.json", names)


if __name__ == "__main__":
    unittest.main()
