import unittest
from datetime import datetime, timezone

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
                    "rows": [{"timestamp": "t1", "message": "m1"}],
                },
                "batch two",
            ]
        }
        out = _extract_map_batches(summary_result, "fallback")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["batch_summary"], "batch one")
        self.assertEqual(out[0]["batch_logs_count"], 1)
        self.assertEqual(out[0]["batch_logs"][0]["message"], "m1")
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


if __name__ == "__main__":
    unittest.main()
