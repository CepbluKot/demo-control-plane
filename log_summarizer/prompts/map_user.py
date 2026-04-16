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
    incident_start: Optional[str] = None,
    incident_end: Optional[str] = None,
) -> str:
    """Форматирует user-промпт для MAP-фазы.

    Гарантирует, что ни одна строка лога не будет обрезана на полуслове:
    использует trim_rows_to_budget, которая отбрасывает строки целиком.

    Args:
        chunk: Нарезанный батч логов.
        metrics: Опциональные метрики за тот же период.
        log_budget_tokens: Максимум токенов на блок логов.
        metrics_budget_tokens: Максимум токенов на блок метрик.
        incident_start: ISO-строка начала окна инцидента (для zone-подсказки).
        incident_end: ISO-строка конца окна инцидента (для zone-подсказки).
    """
    t_start = chunk.time_range[0].isoformat()
    t_end = chunk.time_range[1].isoformat()
    batch_zone = chunk.batch_zone

    is_mixed = batch_zone == "mixed"

    # В mixed-батче добавляем префикс зоны к каждой строке
    if is_mixed:
        zone_prefix = {
            "context_before": "[CB] ",
            "incident":       "[INC] ",
            "context_after":  "[CA] ",
        }
        raw_lines = [
            zone_prefix.get(row.zone, "") + row.raw_line
            for row in chunk.rows
        ]
    else:
        raw_lines = [row.raw_line for row in chunk.rows]

    kept_lines = trim_rows_to_budget(raw_lines, log_budget_tokens)

    dropped = len(raw_lines) - len(kept_lines)
    header_parts = [
        f"## Log batch  {t_start} → {t_end}",
        f"Batch zone: {batch_zone}",
    ]
    if incident_start and incident_end:
        header_parts.append(f"Incident window: {incident_start} → {incident_end}")
    if is_mixed:
        header_parts.append("Zone prefixes: [CB]=context_before  [INC]=incident  [CA]=context_after")
    header_parts.append(
        f"Lines in chunk: {len(chunk.rows)}"
        + (f"  (showing {len(kept_lines)}, {dropped} dropped — token budget)" if dropped else "")
    )

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
