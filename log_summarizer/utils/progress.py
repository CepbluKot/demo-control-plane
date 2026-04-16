"""Утилиты для отображения прогресса пайплайна в логах."""

from __future__ import annotations

import time


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
