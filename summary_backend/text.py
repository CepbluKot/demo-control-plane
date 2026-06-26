"""Text splitting without a tokenizer."""

from __future__ import annotations

import math


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 2.2))


def split_atomic_units(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return lines or [normalized]


def _split_large_unit(unit: str, target_chars: int) -> list[str]:
    if len(unit) <= target_chars:
        return [unit]
    pieces: list[str] = []
    start = 0
    while start < len(unit):
        end = min(len(unit), start + target_chars)
        pieces.append(unit[start:end].strip())
        start = end
    return [piece for piece in pieces if piece]


def build_chunks(text: str, target_estimated_tokens: int) -> list[str]:
    target_estimated_tokens = max(256, target_estimated_tokens)
    target_chars = int(target_estimated_tokens * 2.0)

    units: list[str] = []
    for unit in split_atomic_units(text):
        units.extend(_split_large_unit(unit, target_chars))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > target_estimated_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


class CharBudgetChunker:
    """Default no-tokenizer chunker used by the MVP pipeline."""

    def build_chunks(self, text: str, target_estimated_tokens: int) -> list[str]:
        return build_chunks(text, target_estimated_tokens)
