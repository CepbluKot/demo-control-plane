import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List

from control_plane.prometheus_io import calculate_max_range, fetch_metric_in_batches


class _FakePrometheus:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def custom_query_range(self, *, query: str, start_time: datetime, end_time: datetime, step: str):
        self.calls.append(
            {
                "query": query,
                "start_time": start_time,
                "end_time": end_time,
                "step": step,
            }
        )
        step_seconds = 60 if step == "1m" else 1
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        values = []
        ts = start_ts
        while ts <= end_ts:
            values.append((ts, "1.0"))
            ts += step_seconds
        return [{"metric": {"__name__": "up"}, "values": values}]


class TestPrometheusIO(unittest.TestCase):
    def test_calculate_max_range_respects_max_points(self) -> None:
        out = calculate_max_range(step_seconds=60, max_points=11000)
        self.assertEqual(out, 10999 * 60)

    def test_fetch_metric_in_batches_uses_max_points_sized_windows(self) -> None:
        prom = _FakePrometheus()
        start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 5, 0, 0, tzinfo=timezone.utc)
        result = fetch_metric_in_batches(
            prom=prom,
            query="up",
            start_time=start,
            end_time=end,
            step="1m",
            max_points=100,
        )
        self.assertGreater(len(prom.calls), 1)
        for call in prom.calls:
            window_seconds = int((call["end_time"] - call["start_time"]).total_seconds())
            points = window_seconds // 60 + 1
            self.assertLessEqual(points, 100)

        self.assertEqual(len(result), 1)
        # 5h range with 1m step includes both boundaries: 301 points.
        self.assertEqual(len(result[0]["values"]), 301)


if __name__ == "__main__":
    unittest.main()

