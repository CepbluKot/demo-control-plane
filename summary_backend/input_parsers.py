"""Uploaded log parsers.

Parsers normalize different DBeaver-style exports into LogRecord rows. They do
not decide how rows are chunked or persisted.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from .input_models import LogRecord


class InputParseError(ValueError):
    """Raised when an uploaded file cannot be parsed as the requested format."""


RAW_LINE_COLUMNS = ("raw_line", "log", "message", "line", "raw", "text")
TIMESTAMP_COLUMNS = ("timestamp", "time", "start_time", "event_time")
END_TIME_COLUMNS = ("end_time", "finish_time", "finished_at")
NAMESPACE_COLUMNS = ("namespace", "kubernetes_namespace_name", "k8s_namespace")
CONTAINER_COLUMNS = ("container_name", "kubernetes_container_name", "container")
POD_COLUMNS = ("pod_name", "kubernetes_pod_name", "pod")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _clean_row(row: dict[Any, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            if value:
                cleaned["_extra"] = " ".join(_as_text(item) for item in value)
            continue
        cleaned[str(key).strip()] = _as_text(value).strip()
    return cleaned


def _pick(row: dict[str, str], candidates: Iterable[str]) -> tuple[str, str]:
    by_lower = {key.lower(): key for key in row}
    for candidate in candidates:
        original = by_lower.get(candidate.lower())
        if original is not None:
            return original, row.get(original, "")
    return "", ""


def normalize_log_record(row: dict[str, Any], *, raw_line_column: str | None = None) -> LogRecord:
    cleaned = _clean_row(row)
    raw_key = ""
    raw_line = ""
    if raw_line_column:
        raw_key, raw_line = _pick(cleaned, (raw_line_column,))
        if not raw_key:
            raise InputParseError(f"raw_line column not found: {raw_line_column}")
    else:
        raw_key, raw_line = _pick(cleaned, RAW_LINE_COLUMNS)
        if not raw_key and len(cleaned) == 1:
            raw_key = next(iter(cleaned))
            raw_line = cleaned[raw_key]

    timestamp_key, timestamp = _pick(cleaned, TIMESTAMP_COLUMNS)
    end_time_key, end_time = _pick(cleaned, END_TIME_COLUMNS)
    namespace_key, namespace = _pick(cleaned, NAMESPACE_COLUMNS)
    container_key, container_name = _pick(cleaned, CONTAINER_COLUMNS)
    pod_key, pod_name = _pick(cleaned, POD_COLUMNS)

    used = {raw_key, timestamp_key, end_time_key, namespace_key, container_key, pod_key, ""}
    attrs = {key: value for key, value in cleaned.items() if key not in used and value != ""}

    if not raw_line:
        non_empty = [value for value in cleaned.values() if value]
        raw_line = " ".join(non_empty)
    if not raw_line:
        raise InputParseError("row does not contain log text")

    return LogRecord(
        raw_line=raw_line,
        timestamp=timestamp,
        end_time=end_time,
        namespace=namespace,
        container_name=container_name,
        pod_name=pod_name,
        attrs=attrs,
    )


class UploadedLogParser:
    source_format: str

    def parse_text_stream(self, stream: TextIO, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        raise NotImplementedError


class PlainTextLogParser(UploadedLogParser):
    source_format = "plain_text"

    def parse_text_stream(self, stream: TextIO, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        for line in stream:
            raw_line = line.rstrip("\r\n")
            if raw_line.strip():
                yield LogRecord(raw_line=raw_line)


class CsvLogParser(UploadedLogParser):
    source_format = "csv"

    def parse_text_stream(self, stream: TextIO, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        dialect = self._detect_dialect(stream)
        reader = csv.DictReader(stream, dialect=dialect)
        if not reader.fieldnames:
            raise InputParseError("CSV header is empty")
        for row in reader:
            if not row or not any(_as_text(value).strip() for value in row.values() if value is not None):
                continue
            yield normalize_log_record(row, raw_line_column=raw_line_column)

    @staticmethod
    def _detect_dialect(stream: TextIO) -> type[csv.Dialect] | csv.Dialect:
        try:
            position = stream.tell()
        except (OSError, AttributeError):
            position = None
        sample = stream.read(65536)
        if position is not None:
            stream.seek(position)
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            return csv.excel


class MarkdownTableLogParser(UploadedLogParser):
    source_format = "markdown"

    def parse_text_stream(self, stream: TextIO, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        rows = [line.strip() for line in stream if line.strip().startswith("|")]
        if len(rows) < 2:
            raise InputParseError("markdown table must contain a header and at least one data row")
        headers = self._split_row(rows[0])
        if not headers:
            raise InputParseError("markdown table header is empty")
        for line in rows[1:]:
            cells = self._split_row(line)
            if self._is_separator(cells):
                continue
            if len(cells) > len(headers):
                cells = cells[: len(headers) - 1] + ["|".join(cells[len(headers) - 1 :])]
            if len(cells) < len(headers):
                cells.extend([""] * (len(headers) - len(cells)))
            row = dict(zip(headers, cells))
            if not any(value.strip() for value in row.values()):
                continue
            yield normalize_log_record(row, raw_line_column=raw_line_column)

    @staticmethod
    def _split_row(line: str) -> list[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]

        cells: list[str] = []
        current: list[str] = []
        escaped = False
        for char in stripped:
            if escaped:
                if char == "|":
                    current.append(char)
                else:
                    current.append("\\")
                    current.append(char)
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "|":
                cells.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if escaped:
            current.append("\\")
        cells.append("".join(current).strip())
        return cells

    @staticmethod
    def _is_separator(cells: list[str]) -> bool:
        if not cells:
            return False
        return all(cell.strip().strip(":").replace("-", "") == "" and "-" in cell for cell in cells)


class JsonLogParser(UploadedLogParser):
    source_format = "json"

    def parse_text_stream(self, stream: TextIO, *, raw_line_column: str | None = None) -> Iterator[LogRecord]:
        for row in self._iter_first_array_items(stream):
            if isinstance(row, dict):
                yield normalize_log_record(row, raw_line_column=raw_line_column)
            else:
                yield LogRecord(raw_line=_as_text(row))

    @staticmethod
    def _iter_first_array_items(stream: TextIO) -> Iterator[Any]:
        JsonLogParser._seek_first_array(stream)
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False

        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                chunk = stream.read(65536)
                if chunk == "":
                    eof = True
                    break
                buffer += chunk
                buffer = buffer.lstrip()

            if not buffer:
                if eof:
                    raise InputParseError("JSON array is not closed")
                continue

            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue

            while True:
                try:
                    item, offset = decoder.raw_decode(buffer)
                    yield item
                    buffer = buffer[offset:]
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise InputParseError(f"invalid JSON array item: {exc}") from exc
                    chunk = stream.read(65536)
                    if chunk == "":
                        eof = True
                    else:
                        buffer += chunk

    @staticmethod
    def _seek_first_array(stream: TextIO) -> None:
        in_string = False
        escaped = False
        while True:
            char = stream.read(1)
            if char == "":
                raise InputParseError("JSON must be an array or an object containing an array")
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "[":
                return


@dataclass(frozen=True)
class ParserRegistry:
    parsers: dict[str, UploadedLogParser]

    @classmethod
    def default(cls) -> "ParserRegistry":
        parsers: list[UploadedLogParser] = [CsvLogParser(), MarkdownTableLogParser(), JsonLogParser(), PlainTextLogParser()]
        return cls({parser.source_format: parser for parser in parsers})

    def detect_format(self, *, filename: str, content_type: str, requested_format: str = "auto") -> str:
        requested = requested_format.lower().strip()
        if requested and requested != "auto":
            if requested not in self.parsers:
                raise InputParseError(f"unsupported source_format: {requested_format}")
            return requested

        suffix = Path(filename or "").suffix.lower()
        if suffix in {".json"}:
            return "json"
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".csv", ".tsv"}:
            return "csv"

        lowered_content_type = (content_type or "").lower()
        if "json" in lowered_content_type:
            return "json"
        if "markdown" in lowered_content_type:
            return "markdown"
        if suffix in {".txt", ".text", ".log"}:
            return "plain_text"
        if lowered_content_type.startswith("text/plain"):
            return "plain_text"
        return "csv"

    def get(self, source_format: str) -> UploadedLogParser:
        try:
            return self.parsers[source_format]
        except KeyError as exc:
            raise InputParseError(f"unsupported source_format: {source_format}") from exc
