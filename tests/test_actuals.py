import unittest
from datetime import datetime, timezone

from control_plane.actuals import _render_clickhouse_query


class TestActualsQueryRender(unittest.TestCase):
    def test_render_clickhouse_query_formats_time_placeholders(self) -> None:
        start = datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 25, 11, 0, 0, tzinfo=timezone.utc)
        query = (
            "SELECT timestamp, value FROM t "
            "WHERE timestamp >= parseDateTimeBestEffort('{start}') "
            "AND timestamp < parseDateTimeBestEffort('{end}')"
        )

        rendered = _render_clickhouse_query(query, start, end)

        self.assertIn(start.isoformat(), rendered)
        self.assertIn(end.isoformat(), rendered)

    def test_render_clickhouse_query_keeps_unknown_placeholder(self) -> None:
        start = datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 25, 11, 0, 0, tzinfo=timezone.utc)
        query = "SELECT * FROM t WHERE x = '{unknown_placeholder}'"

        rendered = _render_clickhouse_query(query, start, end)

        self.assertIn("{unknown_placeholder}", rendered)


if __name__ == "__main__":
    unittest.main()
