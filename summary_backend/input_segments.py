"""Build durable input segments from normalized log records."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .input_models import InputSegment, LogRecord
from .text import estimate_tokens


class RowBudgetInputSegmenter:
    """Groups rendered log rows using the same no-tokenizer estimate as chunks."""

    def build_segments(
        self,
        records: Iterable[LogRecord],
        *,
        source_type: str,
        source_format: str,
        target_estimated_tokens: int,
    ) -> Iterator[InputSegment]:
        target_estimated_tokens = max(256, target_estimated_tokens)
        segment_index = 0
        rows: list[str] = []
        rows_count = 0
        estimated_tokens = 0
        first_timestamp = ""
        last_timestamp = ""

        for record in records:
            rendered = record.render()
            row_tokens = estimate_tokens(rendered)
            if rows and estimated_tokens + row_tokens > target_estimated_tokens:
                yield self._make_segment(
                    segment_index=segment_index,
                    source_type=source_type,
                    source_format=source_format,
                    rows=rows,
                    rows_count=rows_count,
                    first_timestamp=first_timestamp,
                    last_timestamp=last_timestamp,
                )
                segment_index += 1
                rows = []
                rows_count = 0
                estimated_tokens = 0
                first_timestamp = ""
                last_timestamp = ""

            if not first_timestamp and record.timestamp:
                first_timestamp = record.timestamp
            if record.timestamp:
                last_timestamp = record.timestamp
            rows.append(rendered)
            rows_count += 1
            estimated_tokens += row_tokens

        if rows:
            yield self._make_segment(
                segment_index=segment_index,
                source_type=source_type,
                source_format=source_format,
                rows=rows,
                rows_count=rows_count,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
            )

    @staticmethod
    def _make_segment(
        *,
        segment_index: int,
        source_type: str,
        source_format: str,
        rows: list[str],
        rows_count: int,
        first_timestamp: str,
        last_timestamp: str,
    ) -> InputSegment:
        metadata: dict[str, Any] = {
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        }
        return InputSegment(
            segment_index=segment_index,
            source_type=source_type,
            source_format=source_format,
            content="\n".join(rows),
            rows_count=rows_count,
            metadata=metadata,
        )
