import unittest
from unittest.mock import patch

from settings import settings
from ui.pages.logs_summary_page import (
    _build_no_logs_hypothesis_prompt,
    _build_freeform_summary_prompt,
    _enrich_stats_with_elapsed,
    _normalize_summary_text,
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


if __name__ == "__main__":
    unittest.main()
