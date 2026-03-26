import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from control_plane.processing import (
    _call_summarizer_adapter,
    _extract_batch_summaries,
    _extract_map_batches,
    process_anomalies,
)


class TestProcessing(unittest.TestCase):
    def test_extract_batch_summaries_from_mixed_items(self) -> None:
        summary_result = {
            "chunk_summaries": [
                "plain chunk",
                {"summary": "summary chunk"},
                {"text": "text chunk"},
            ]
        }
        out = _extract_batch_summaries(summary_result, "fallback")
        self.assertEqual(out, ["plain chunk", "summary chunk", "text chunk"])

    def test_call_summarizer_adapter_fallbacks_to_start_end_dt(self) -> None:
        def summarize_only_dt(*, start_dt, end_dt, anomaly):
            return f"{start_dt.isoformat()}->{end_dt.isoformat()}::{anomaly['timestamp']}"

        period_start = datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc)
        period_end = datetime(2026, 3, 25, 10, 30, 0, tzinfo=timezone.utc)
        anomaly = {"timestamp": "2026-03-25T10:30:00Z"}

        out = _call_summarizer_adapter(
            summarize_only_dt,
            period_start_dt=period_start,
            period_end_dt=period_end,
            anomaly=anomaly,
        )
        self.assertIn(period_start.isoformat(), out)
        self.assertIn(period_end.isoformat(), out)
        self.assertIn(anomaly["timestamp"], out)

    def test_extract_map_batches_includes_logs_and_count(self) -> None:
        summary_result = {
            "map_batches": [
                {
                    "summary": "batch one",
                    "rows": [{"timestamp": "2026-03-25T10:00:00Z", "message": "m1"}],
                },
                "batch two",
            ]
        }
        out = _extract_map_batches(summary_result, "fallback")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["batch_summary"], "batch one")
        self.assertEqual(out[0]["batch_logs_count"], 1)
        self.assertEqual(out[0]["batch_logs"][0]["message"], "m1")
        self.assertEqual(out[0]["batch_period_start"], "2026-03-25T10:00:00Z")
        self.assertEqual(out[0]["batch_period_end"], "2026-03-25T10:00:00Z")
        self.assertEqual(out[1]["batch_summary"], "batch two")
        self.assertEqual(out[1]["batch_logs_count"], 0)

    def test_process_anomalies_test_mode_emits_map_reduce_flow(self) -> None:
        events = []

        def on_event(event, payload):
            events.append((event, payload))

        anomaly = {"timestamp": "2026-03-25T11:00:00Z", "value": 42.0, "source": "actual"}
        results = process_anomalies(
            [anomaly],
            lookback_minutes=20,
            continue_on_error=True,
            test_mode=True,
            on_event=on_event,
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])
        self.assertEqual(results[0]["alert_result"]["status"], "mocked")
        emitted_names = [name for name, _ in events]
        self.assertIn("map_start", emitted_names)
        self.assertIn("map_batch", emitted_names)
        self.assertIn("reduce_done", emitted_names)
        self.assertIn("notification_ready", emitted_names)
        self.assertIn("process_done", emitted_names)
        map_batch_payloads = [payload for name, payload in events if name == "map_batch"]
        self.assertTrue(map_batch_payloads)
        self.assertIn("batch_logs_count", map_batch_payloads[0])
        self.assertIn("batch_logs", map_batch_payloads[0])
        self.assertIn("batch_period_start", map_batch_payloads[0])
        self.assertIn("batch_period_end", map_batch_payloads[0])

    def test_process_anomalies_real_mode_streams_map_batches_before_summary_done(self) -> None:
        events = []

        def on_event(event, payload):
            events.append((event, payload))

        def fake_summarizer(*, start_dt, end_dt, anomaly, on_progress=None):
            _ = (start_dt, end_dt, anomaly)
            if on_progress is not None:
                on_progress("map_start", {"rows_processed": 0, "rows_total": 4})
                on_progress(
                    "map_batch",
                    {
                        "batch_index": 0,
                        "batch_total": 2,
                        "batch_summary": "chunk-1",
                        "batch_logs_count": 2,
                        "batch_logs": [{"timestamp": "t1", "message": "m1"}],
                        "batch_period_start": "2026-03-25T10:00:00Z",
                        "batch_period_end": "2026-03-25T10:00:30Z",
                        "rows_processed": 2,
                        "rows_total": 4,
                    },
                )
                on_progress(
                    "map_batch",
                    {
                        "batch_index": 1,
                        "batch_total": 2,
                        "batch_summary": "chunk-2",
                        "batch_logs_count": 2,
                        "batch_logs": [{"timestamp": "t2", "message": "m2"}],
                        "batch_period_start": "2026-03-25T10:00:31Z",
                        "batch_period_end": "2026-03-25T10:01:00Z",
                        "rows_processed": 4,
                        "rows_total": 4,
                    },
                )
                on_progress("map_done", {"batch_total": 2, "rows_processed": 4, "rows_total": 4})
                on_progress("reduce_start", {"rows_processed": 4, "rows_total": 4})
                on_progress("reduce_done", {"summary": "final", "rows_processed": 4, "rows_total": 4})
            return {
                "summary": "final",
                "map_batches": [
                    {"summary": "chunk-1", "rows": [{"timestamp": "t1", "message": "m1"}]},
                    {"summary": "chunk-2", "rows": [{"timestamp": "t2", "message": "m2"}]},
                ],
                "rows_processed": 4,
                "rows_total_estimate": 4,
            }

        def fake_alert(text: str):
            return {"status": "ok", "summary_text": text}

        def fake_loader(path: str):
            if "summarizer" in path:
                return fake_summarizer
            return fake_alert

        anomaly = {"timestamp": "2026-03-25T11:00:00Z", "value": 42.0, "source": "actual"}
        with patch("control_plane.processing.SUMMARIZER_CALLABLE", "fake.summarizer"), patch(
            "control_plane.processing.ALERT_CALLABLE",
            "fake.alert",
        ), patch(
            "control_plane.processing._load_callable",
            side_effect=fake_loader,
        ):
            results = process_anomalies(
                [anomaly],
                lookback_minutes=20,
                continue_on_error=True,
                test_mode=False,
                on_event=on_event,
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])
        emitted_names = [name for name, _ in events]
        self.assertIn("map_batch", emitted_names)
        self.assertIn("summary_done", emitted_names)
        self.assertIn("reduce_done", emitted_names)
        first_map_batch_idx = emitted_names.index("map_batch")
        summary_done_idx = emitted_names.index("summary_done")
        reduce_done_idx = emitted_names.index("reduce_done")
        self.assertLess(first_map_batch_idx, summary_done_idx)
        self.assertLess(summary_done_idx, reduce_done_idx)
        map_batch_payload = next(payload for name, payload in events if name == "map_batch")
        self.assertIn("batch_logs", map_batch_payload)
        self.assertIn("rows_processed", map_batch_payload)
        self.assertIn("batch_period_start", map_batch_payload)


if __name__ == "__main__":
    unittest.main()
