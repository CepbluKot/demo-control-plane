"""Форматирование user-промпта для MAP-фазы."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from log_summarizer.utils.tokens import estimate_tokens, trim_rows_to_budget

if TYPE_CHECKING:
    from log_summarizer.models import Chunk, MetricRow


def format_map_user_prompt(
    chunk: "Chunk",
    metrics: Optional[list["MetricRow"]] = None,
    log_budget_tokens: int = 40_000,
    metrics_budget_tokens: int = 2_000,
) -> str:
    """Форматирует user-промпт для MAP-фазы.

    Гарантирует, что ни одна строка лога не будет обрезана на полуслове:
    использует trim_rows_to_budget, которая отбрасывает строки целиком.

    Args:
        chunk: Нарезанный батч логов.
        metrics: Опциональные метрики за тот же период.
        log_budget_tokens: Максимум токенов на блок логов.
        metrics_budget_tokens: Максимум токенов на блок метрик.
    """
    t_start = chunk.time_range[0].isoformat()
    t_end = chunk.time_range[1].isoformat()

    # Отбираем строки целиком, не обрезая ни одну на середине
    raw_lines = [row.raw_line for row in chunk.rows]
    kept_lines = trim_rows_to_budget(raw_lines, log_budget_tokens)

    dropped = len(raw_lines) - len(kept_lines)
    header_parts = [
        f"## Log batch  {t_start} → {t_end}",
        f"Lines in chunk: {len(chunk.rows)}"
        + (f"  (showing {len(kept_lines)}, {dropped} dropped — token budget)" if dropped else ""),
    ]

    log_block = "\n".join(kept_lines)

    parts: list[str] = [
        "\n".join(header_parts),
        "",
        "```",
        log_block,
        "```",
    ]

    if metrics:
        metric_lines = _format_metrics(metrics, metrics_budget_tokens)
        if metric_lines:
            parts += ["", "## Metrics", *metric_lines]

    return "\n".join(parts)


# ── Вспомогательные ────────────────────────────────────────────────────


def _format_metrics(
    metrics: list["MetricRow"],
    budget_tokens: int,
) -> list[str]:
    """Форматирует строки метрик с ограничением по токенам (целыми записями)."""
    formatted = [
        f"{m.timestamp.isoformat()}  {m.service}  {m.metric_name}={m.value}"
        for m in metrics
    ]
    return trim_rows_to_budget(formatted, budget_tokens)
