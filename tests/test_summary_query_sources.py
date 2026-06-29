from __future__ import annotations

import unittest

from summary_backend.query_sources import ClickHouseQueryLogRecordSource, QuerySourceError, validate_read_query


class FakeStream:
    def __init__(self, column_names, blocks) -> None:
        self.source = type("Source", (), {"column_names": column_names})()
        self.blocks = blocks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __iter__(self):
        return iter(self.blocks)


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def query_row_block_stream(self, query: str):
        self.query = query
        return FakeStream(
            ("timestamp", "namespace", "container_name", "severity", "raw_line"),
            [[("2026-01-01", "ns", "api", "error", "failed")]],
        )

    def close(self) -> None:
        self.closed = True


class FakePreviewResult:
    column_names = ("timestamp", "message", "value")
    result_rows = (("2026-01-01", "ok", 42),)


class FakePreviewClient:
    def __init__(self) -> None:
        self.closed = False
        self.query_text = ""

    def query(self, query: str):
        self.query_text = query
        return FakePreviewResult()

    def close(self) -> None:
        self.closed = True


class SummaryQuerySourceTests(unittest.TestCase):
    def test_validate_read_query_allows_select_and_with(self) -> None:
        self.assertEqual(validate_read_query(" SELECT 1; "), "SELECT 1")
        self.assertEqual(validate_read_query("WITH x AS (SELECT 1) SELECT * FROM x"), "WITH x AS (SELECT 1) SELECT * FROM x")

    def test_validate_read_query_rejects_non_read_queries(self) -> None:
        with self.assertRaises(QuerySourceError):
            validate_read_query("DROP TABLE logs")

    def test_clickhouse_query_source_streams_rows_to_log_records(self) -> None:
        fake_client = FakeClient()
        source = ClickHouseQueryLogRecordSource(client_factory=lambda: fake_client)

        records = list(source.iter_log_records("SELECT * FROM logs"))

        self.assertTrue(fake_client.closed)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].timestamp, "2026-01-01")
        self.assertEqual(records[0].namespace, "ns")
        self.assertEqual(records[0].container_name, "api")
        self.assertEqual(records[0].attrs["severity"], "error")
        self.assertEqual(records[0].raw_line, "failed")

    def test_clickhouse_query_source_previews_limited_rows(self) -> None:
        fake_client = FakePreviewClient()
        source = ClickHouseQueryLogRecordSource(client_factory=lambda: fake_client)

        columns, rows = source.preview_rows("SELECT timestamp, message, value FROM logs", limit=25)

        self.assertTrue(fake_client.closed)
        self.assertEqual(
            fake_client.query_text,
            "SELECT * FROM (SELECT timestamp, message, value FROM logs) LIMIT 25",
        )
        self.assertEqual(columns, ["timestamp", "message", "value"])
        self.assertEqual(rows, [{"timestamp": "2026-01-01", "message": "ok", "value": 42}])


if __name__ == "__main__":
    unittest.main()
