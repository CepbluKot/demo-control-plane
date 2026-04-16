"""Chunker — нарезка строк лога на батчи по токеновому бюджету.

Почему токены, а не строки:
  200 строк с "OK" → ~2k токенов.
  200 строк со stack trace → ~60k токенов.
  Фиксированное количество строк неэффективно.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from log_summarizer.models import Chunk, LogRow
from log_summarizer.utils.logging import get_logger
from log_summarizer.utils.tokens import estimate_tokens

logger = get_logger("chunker")


class Chunker:
    """Нарезка списка строк лога на Chunk'и с контролем токенов.

    Args:
        max_batch_tokens: Максимум токенов на один батч (55% от контекста).
        min_batch_lines: Минимальный батч при split (не делим меньше).
    """

    def __init__(
        self,
        max_batch_tokens: int,
        min_batch_lines: int = 20,
    ) -> None:
        self.max_batch_tokens = max_batch_tokens
        self.min_batch_lines = min_batch_lines

    def chunk(self, rows: list[LogRow]) -> list[Chunk]:
        """Нарезаем строки на Chunk'и.

        Набираем строки в текущий чанк пока estimate_tokens < max_batch_tokens.
        Когда лимит достигнут — закрываем чанк, начинаем новый.

        Args:
            rows: Все строки лога в хронологическом порядке.

        Returns:
            Список Chunk, каждый ≤ max_batch_tokens токенов.
        """
        if not rows:
            return []

        chunks: list[Chunk] = []
        current: list[LogRow] = []
        current_tokens = 0

        for row in rows:
            row_tokens = estimate_tokens(row.raw_line)

            if current and current_tokens + row_tokens > self.max_batch_tokens:
                chunks.append(self._make_chunk(current, current_tokens))
                current = []
                current_tokens = 0

            current.append(row)
            current_tokens += row_tokens

        if current:
            chunks.append(self._make_chunk(current, current_tokens))

        logger.info(
            "Chunked %d rows into %d chunks (max_batch_tokens=%d)",
            len(rows), len(chunks), self.max_batch_tokens,
        )
        return chunks

    def split_chunk(self, chunk: Chunk) -> tuple[Chunk, Chunk]:
        """Делим чанк пополам по строкам.

        Используется при ContextOverflowError в MapProcessor.

        Args:
            chunk: Исходный чанк.

        Returns:
            Два чанка: левая и правая половина.
        """
        rows = chunk.rows
        mid = max(len(rows) // 2, 1)
        left_rows = rows[:mid]
        right_rows = rows[mid:]
        left = self._make_chunk(
            left_rows,
            sum(estimate_tokens(r.raw_line) for r in left_rows),
        )
        right = self._make_chunk(
            right_rows,
            sum(estimate_tokens(r.raw_line) for r in right_rows),
        )
        return left, right

    # ── Статические методы ────────────────────────────────────────────

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Грубая оценка токенов: len(text) // 3.

        Статический метод — чтобы TreeReducer мог вызывать без инстанции.
        """
        return estimate_tokens(text)

    # ── Внутренние ───────────────────────────────────────────────────

    @staticmethod
    def _make_chunk(rows: list[LogRow], token_estimate: int) -> Chunk:
        """Создаём Chunk из списка строк."""
        if not rows:
            raise ValueError("Cannot make chunk from empty rows")
        first_ts = rows[0].timestamp
        last_ts = rows[-1].timestamp
        if first_ts > last_ts:
            first_ts, last_ts = last_ts, first_ts

        # batch_zone: если все строки одной зоны — эта зона, иначе "mixed"
        zones = {r.zone for r in rows}
        batch_zone = zones.pop() if len(zones) == 1 else "mixed"

        return Chunk(
            rows=rows,
            time_range=(first_ts, last_ts),
            token_estimate=token_estimate,
            batch_zone=batch_zone,
        )
