"""Query-based input sources for summary jobs."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol

from .config import Settings, get_settings
from .input_models import LogRecord
from .input_parsers import InputParseError, normalize_log_record


class QuerySourceError(ValueError):
    """Raised when a query source cannot produce log records."""


class QueryLogRecordSource(Protocol):
    def iter_log_records(self, query: str, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        ...


def validate_read_query(query: str) -> str:
    normalized = query.strip().rstrip(";")
    lowered = normalized.lower()
    if not normalized:
        raise QuerySourceError("query is empty")
    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise QuerySourceError("only SELECT/WITH read queries are allowed")
    return normalized


class ClickHouseQueryLogRecordSource:
    """Streams ClickHouse query result rows and normalizes them into LogRecord."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client_factory = client_factory

    def iter_log_records(self, query: str, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        query = validate_read_query(query)
        client = self._make_client()
        try:
            with client.query_row_block_stream(query) as stream:
                column_names = [str(name) for name in stream.source.column_names]
                if not column_names:
                    raise QuerySourceError("query result has no columns")
                for block in stream:
                    for row in block:
                        try:
                            yield normalize_log_record(dict(zip(column_names, row)), raw_line_column=raw_line_column)
                        except InputParseError as exc:
                            raise QuerySourceError(str(exc)) from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def preview_rows(self, query: str, *, limit: int = 100) -> tuple[list[str], list[dict[str, Any]]]:
        query = validate_read_query(query)
        safe_limit = max(1, min(int(limit), 500))
        preview_query = f"SELECT * FROM ({query}) LIMIT {safe_limit}"
        client = self._make_client()
        try:
            result = client.query(preview_query)
            column_names = [str(name) for name in result.column_names]
            rows = [dict(zip(column_names, row)) for row in result.result_rows]
            return column_names, rows
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _make_client(self):
        if self._client_factory is not None:
            return self._client_factory()

        import clickhouse_connect

        return clickhouse_connect.get_client(
            host=self.settings.source_clickhouse_host,
            port=self.settings.source_clickhouse_port,
            username=self.settings.source_clickhouse_username,
            password=self.settings.source_clickhouse_password,
            database=self.settings.source_clickhouse_database,
            secure=self.settings.source_clickhouse_secure,
        )
