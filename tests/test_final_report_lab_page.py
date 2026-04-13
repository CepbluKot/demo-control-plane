import unittest
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ui.pages.final_report_lab_page import (
    DEFAULT_LAB_MODEL,
    FINAL_REPORT_LAB_TOKENS_PER_MINUTE_LIMIT,
    _TokenPerMinuteLimiter,
    _build_chat_completions_url,
    _build_map_summaries_text,
    _build_synthetic_report_chunks,
    _estimate_tokens,
    _extract_openai_assistant_text,
    _make_groq_chat_call,
    _merge_synthetic_chunks,
    _persist_final_report_lab_artifacts,
)


class _FakeResponse:
    def __init__(self, *, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class TestFinalReportLabHelpers(unittest.TestCase):
    def test_defaults_for_model_and_tpm_limit(self) -> None:
        self.assertEqual(DEFAULT_LAB_MODEL, "PNX.QWEN3 235b a22b instruct")
        self.assertEqual(FINAL_REPORT_LAB_TOKENS_PER_MINUTE_LIMIT, 30_000)

    def test_estimate_tokens(self) -> None:
        self.assertEqual(_estimate_tokens(""), 0)
        self.assertEqual(_estimate_tokens("abc"), 1)
        self.assertEqual(_estimate_tokens("abcd"), 2)

    def test_token_limiter_prunes_after_one_minute(self) -> None:
        limiter = _TokenPerMinuteLimiter(tokens_per_minute=10)
        self.assertEqual(limiter.acquire(6, now=100.0), 0.0)
        self.assertEqual(limiter.acquire(4, now=100.0), 0.0)
        self.assertEqual(limiter.acquire(5, now=161.0), 0.0)

    def test_build_chat_completions_url_variants(self) -> None:
        self.assertEqual(
            _build_chat_completions_url("https://api.groq.com/openai/v1"),
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.assertEqual(
            _build_chat_completions_url("https://api.groq.com/openai/v1/chat/completions"),
            "https://api.groq.com/openai/v1/chat/completions",
        )

    def test_extract_openai_assistant_text_with_string_content(self) -> None:
        payload = {"choices": [{"message": {"content": "Привет, мир"}}]}
        self.assertEqual(_extract_openai_assistant_text(payload), "Привет, мир")

    def test_extract_openai_assistant_text_with_list_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Строка 1"},
                            {"type": "text", "text": "Строка 2"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(_extract_openai_assistant_text(payload), "Строка 1\nСтрока 2")

    def test_build_synthetic_chunks_and_merge(self) -> None:
        chunks = _build_synthetic_report_chunks(
            chunk_count=3,
            events_per_chunk=10,
            details_per_event=1,
            paragraphs_per_chunk=2,
            seed=42,
        )
        self.assertEqual(len(chunks), 3)
        merged = _merge_synthetic_chunks(chunks)
        map_text = _build_map_summaries_text(chunks)
        self.assertIn("L1 summary chunk 1", merged)
        self.assertIn("MAP BATCH 0001", map_text)
        self.assertGreater(len(merged), 1000)
        self.assertGreater(len(map_text), 1000)

    def test_make_groq_chat_call_retries_then_succeeds(self) -> None:
        responses = [
            _FakeResponse(status_code=500, text='{"error":"boom"}'),
            _FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"content": "ok"}}]},
            ),
        ]
        captured = {}

        def _fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            captured["json"] = dict(json or {})
            captured["timeout"] = timeout
            return responses.pop(0)

        with patch("ui.pages.final_report_lab_page.requests.post", side_effect=_fake_post):
            llm_call = _make_groq_chat_call(
                api_base="https://api.groq.com/openai/v1/chat/completions",
                api_key="secret",
                model="qwen/qwen3-32b",
                timeout_seconds=30.0,
                max_retries=2,
                temperature=0.2,
                max_tokens=512,
            )
            self.assertEqual(llm_call("ping"), "ok")
        self.assertEqual(captured["headers"].get("Authorization"), "Bearer secret")
        self.assertEqual(captured["json"].get("model"), "qwen/qwen3-32b")
        self.assertIn("messages", captured["json"])

    def test_persist_final_report_lab_artifacts_writes_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch("ui.pages.final_report_lab_page.settings.CONTROL_PLANE_ARTIFACTS_DIR", tmpdir):
                payload = _persist_final_report_lab_artifacts(
                    user_goal="goal",
                    metrics_context="metrics",
                    period_start="2026-03-18T00:00:00+03:00",
                    period_end="2026-03-18T06:00:00+03:00",
                    chunks=["chunk-1", "chunk-2"],
                    base_summary="base-summary",
                    map_summaries_text="map-summaries",
                    results={
                        "structured": {
                            "merged_text": "merged-structured",
                            "sections": [{"title": "1. Test", "text": "sec"}],
                            "llm_calls": [
                                {
                                    "call": 1,
                                    "section_index": 1,
                                    "section_title": "1. Test",
                                    "merge_section_label": "1/13 — 1. Test",
                                    "status": "ok",
                                    "error": "",
                                    "prompt_chars": 10,
                                    "response_chars": 20,
                                    "elapsed_sec": 0.2,
                                    "prompt_text": "prompt",
                                    "response_text": "response",
                                    "merge_previous_sections_text": "prev",
                                    "merge_base_summary_text": "base",
                                    "merge_map_summaries_text": "map",
                                }
                            ],
                            "elapsed_sec": 1.0,
                        }
                    },
                    logger=logging.getLogger("test.final_report_lab"),
                )
            run_dir = Path(str(payload["run_dir"]))
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "inputs" / "base_summary.txt").exists())
            self.assertTrue((run_dir / "outputs" / "structured" / "merged_report.md").exists())
            self.assertTrue((run_dir / "outputs" / "structured" / "llm_calls.jsonl").exists())
            self.assertGreater(len(payload.get("files") or []), 0)


if __name__ == "__main__":
    unittest.main()
