"""
Программный мерж alert_refs из нескольких summary.
Выполняется ДО отправки группы в LLM reduce.

Статусы мержатся детерминированно по приоритету.
related_events — union.
explanation — собираются в список для последующего синтеза LLM.
"""

from __future__ import annotations

from collections import defaultdict

from schemas import AlertRef, IncidentSummary


STATUS_PRIORITY = {
    "EXPLAINED": 4,
    "PARTIALLY": 3,
    "NOT_EXPLAINED": 2,
    "NOT_SEEN_IN_BATCH": 1,
}


def merge_alert_refs(
    summaries: list[IncidentSummary],
    id_remap: dict[str, str] | None = None,
) -> list[AlertRef]:
    """
    Мержит alert_refs из нескольких summary.

    Args:
        summaries: список summary из одной reduce-группы.
        id_remap: маппинг старых event_id → новые (если timeline
                  была перенумерована). Если None — id не меняются.

    Returns:
        Список смерженных AlertRef. Статусы — финальные.
        Поле explanation содержит собранные explanation из всех батчей,
        разделённые ' ||| ' — LLM в reduce синтезирует из них одно.
    """
    grouped: dict[str, list[AlertRef]] = defaultdict(list)

    for summary in summaries:
        for ref in summary.alert_refs:
            grouped[ref.alert_id].append(ref)

    merged: list[AlertRef] = []

    for alert_id, refs in grouped.items():
        # Статус: берём максимальный по приоритету
        best_status = max(
            (r.status for r in refs),
            key=lambda s: STATUS_PRIORITY[s],
        )

        # related_events: union всех, с ремапом id если нужно
        all_events: set[str] = set()
        for r in refs:
            for eid in r.related_events:
                if id_remap:
                    all_events.add(id_remap.get(eid, eid))
                else:
                    all_events.add(eid)

        # explanation: собираем все непустые
        explanations = [r.explanation for r in refs if r.explanation.strip()]
        combined_explanation = " ||| ".join(explanations) if explanations else ""

        merged.append(AlertRef(
            alert_id=alert_id,
            status=best_status,
            related_events=sorted(all_events),
            explanation=combined_explanation,
        ))

    return merged
