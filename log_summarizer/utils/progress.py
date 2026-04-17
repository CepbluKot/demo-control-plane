"""Утилиты для отображения прогресса пайплайна в логах."""

from __future__ import annotations

import time
from datetime import timezone
from typing import Any


def fmt_dur(seconds: float) -> str:
    """Форматирует длительность: '45s', '3m 12s', '1h 04m'."""
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def bar(done: int, total: int, width: int = 14) -> str:
    """ASCII прогресс-бар: '████████░░░░░░'."""
    if total <= 0:
        return "░" * width
    filled = min(int(width * done / total), width)
    return "█" * filled + "░" * (width - filled)


class ProgressTracker:
    """Трекер прогресса с ETA.

    Потокобезопасен в контексте asyncio (однопоточный event loop).

    Args:
        total: Общее количество элементов.
        label: Название фазы (например, 'MAP', 'REPORT').
    """

    def __init__(self, total: int, label: str) -> None:
        self.total = total
        self.label = label
        self.done = 0
        self._t0 = time.monotonic()

    def tick(self, detail: str = "") -> str:
        """Инкрементирует счётчик и возвращает строку прогресса."""
        self.done += 1
        return self.line(detail)

    def line(self, detail: str = "") -> str:
        """Возвращает строку прогресса без изменения счётчика."""
        elapsed = time.monotonic() - self._t0
        pct = 100.0 * self.done / self.total if self.total else 0.0

        if 0 < self.done < self.total:
            eta_sec = elapsed / self.done * (self.total - self.done)
            eta_str = f"~{fmt_dur(eta_sec)}"
        elif self.done >= self.total:
            eta_str = "✓ done"
        else:
            eta_str = "?"

        parts = [
            f"{self.label}",
            f"{self.done}/{self.total}",
            bar(self.done, self.total),
            f"{pct:.0f}%",
            f"elapsed {fmt_dur(elapsed)}",
            f"ETA {eta_str}",
        ]
        if detail:
            parts.append(detail)
        return "  ".join(parts)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def summary(self) -> str:
        """Итоговая строка после завершения фазы."""
        elapsed = time.monotonic() - self._t0
        return f"{self.label}  {self.total}/{self.total}  ✓  total {fmt_dur(elapsed)}"


class TimeProgress:
    """Прогресс выгрузки по временной оси (доля покрытого временного окна).

    Используется в DataLoader: прогресс = (last_ts - start) / (end - start).

    Args:
        start: Начало периода (datetime, tz-aware или naive UTC).
        end: Конец периода.
        label: Метка фазы.
    """

    def __init__(self, start: Any, end: Any, label: str = "LOAD") -> None:
        self.label = label
        self._t0 = time.monotonic()
        self._start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        self._end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end
        self._total_sec = max(1.0, (self._end - self._start).total_seconds())

    def line(self, current: Any, page: int, total_rows: int, query_sec: float, page_tokens: int = 0) -> str:
        """Строка прогресса после очередной страницы.

        Args:
            current: Текущий last_ts как datetime (последняя строка страницы).
            page: Номер страницы.
            total_rows: Накопленное кол-во групп за все страницы.
            query_sec: Время выполнения последнего SQL-запроса.
            page_tokens: Кол-во токенов в строках этой страницы.
        """
        elapsed = time.monotonic() - self._t0
        cur = current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current
        done_sec = max(0.0, (cur - self._start).total_seconds())
        pct = min(100.0, 100.0 * done_sec / self._total_sec)
        if 0 < pct < 100 and elapsed > 0:
            eta_sec = elapsed / (pct / 100.0) - elapsed
            eta_str = f"~{fmt_dur(eta_sec)}"
        else:
            eta_str = "?"
        parts = [
            self.label,
            f"стр.{page}",
            bar(int(pct), 100),
            f"{pct:.0f}%",
            f"{total_rows:,} гр.",
        ]
        if page_tokens:
            parts.append(f"~{page_tokens:,} tok")
        parts += [
            f"запрос {query_sec:.1f}s",
            f"elapsed {fmt_dur(elapsed)}",
            f"ETA {eta_str}",
        ]
        return "  ".join(parts)

    def summary(self, pages: int, total_rows: int) -> str:
        """Итоговая строка после завершения выгрузки."""
        elapsed = time.monotonic() - self._t0
        return f"{self.label}  100%  ✓  {pages} стр.  {total_rows:,} гр.  {fmt_dur(elapsed)}"
