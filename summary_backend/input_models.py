"""Normalized input models for uploaded log sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .ids import sha256_text


@dataclass(frozen=True)
class LogRecord:
    """One normalized log row from any supported input format."""

    raw_line: str
    timestamp: str = ""
    end_time: str = ""
    namespace: str = ""
    container_name: str = ""
    pod_name: str = ""
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        parts: list[str] = []
        if self.timestamp or self.end_time:
            parts.append(f"[{self.timestamp} -> {self.end_time or self.timestamp}]")
        if self.namespace:
            parts.append(f"namespace={self.namespace}")
        if self.container_name:
            parts.append(f"container={self.container_name}")
        if self.pod_name:
            parts.append(f"pod={self.pod_name}")
        for key, value in sorted(self.attrs.items()):
            if value not in {"", None}:
                parts.append(f"{key}={value}")
        parts.append(str(self.raw_line))
        return "  ".join(part for part in parts if part)


@dataclass(frozen=True)
class InputSegment:
    """A chunk of normalized input rows persisted before MAP nodes are created."""

    segment_index: int
    source_type: str
    source_format: str
    content: str
    rows_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.content)

    @property
    def content_hash(self) -> str:
        return sha256_text(self.content)
