"""MarkdownRenderer — программный рендер MergedAnalysis → Markdown.

Не требует LLM. Детерминированный вывод всех данных из MergedAnalysis.
Использует _ru поля как первичный источник текста, English — как fallback.

Сохраняется рядом с LLM-отчётом как report_data.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from log_summarizer.config import PipelineConfig
from log_summarizer.models import AlertStatus, MergedAnalysis, Severity


# Порядок зон для хронологии
_ZONE_ORDER = ["context_before", "incident", "context_after"]
_ZONE_LABELS = {
    "context_before": "КОНТЕКСТ ДО ИНЦИДЕНТА",
    "incident":       "ОКНО ИНЦИДЕНТА",
    "context_after":  "КОНТЕКСТ ПОСЛЕ ИНЦИДЕНТА",
}

_SEV_EMOJI = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH:     "🟠",
    Severity.MEDIUM:   "🟡",
    Severity.LOW:      "🔵",
    Severity.INFO:     "⚪",
}

_STATUS_LABELS = {
    AlertStatus.EXPLAINED:     "✅ объяснён",
    AlertStatus.PARTIAL:       "⚡ частично",
    AlertStatus.NOT_EXPLAINED: "❓ не объяснён",
    AlertStatus.NOT_SEEN:      "➖ не замечен",
}

_CONF_LABELS = {
    "high":   "🟢 высокая",
    "medium": "🟡 средняя",
    "low":    "🔴 низкая",
}


def _ru(eng: Optional[str], ru: Optional[str]) -> str:
    """Возвращает ru если заполнен, иначе eng, иначе пустую строку."""
    return (ru or eng or "").strip()


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _ts_short(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


class MarkdownRenderer:
    """Рендерит MergedAnalysis в структурированный Markdown для SRE.

    Args:
        merged: Финальный результат REDUCE-фазы.
        config: Конфигурация пайплайна (для временных окон и контекста).
    """

    def __init__(self, merged: MergedAnalysis, config: PipelineConfig) -> None:
        self.merged = merged
        self.config = config
        self._has_context = config.has_context_window()

    def render(self) -> str:
        """Возвращает полный Markdown-документ."""
        parts: list[str] = []

        parts.append(self._header())
        parts.append(self._meta_block())
        parts.append(self._summary())

        if self.merged.alert_refs:
            parts.append(self._alert_coverage())

        parts.append(self._timeline())

        if self.merged.causal_chains:
            parts.append(self._causal_chains())

        parts.append(self._hypotheses())
        parts.append(self._impact())
        parts.append(self._recommendations())

        if self.merged.anomalies:
            parts.append(self._anomalies())

        if self.merged.evidence_bank:
            parts.append(self._evidence())

        if self.merged.gaps:
            parts.append(self._gaps())

        parts.append(self._footer())

        return "\n\n".join(p for p in parts if p.strip())

    # ── Секции ────────────────────────────────────────────────────────

    def _header(self) -> str:
        t0, t1 = self.merged.time_range
        return (
            f"# Анализ инцидента\n\n"
            f"**{_ts(t0)} → {_ts(t1)}**"
        )

    def _meta_block(self) -> str:
        cfg = self.config
        lines: list[str] = []

        inc_start = cfg.incident_start
        inc_end   = cfg.incident_end
        if inc_start and inc_end:
            lines.append(f"| **Окно инцидента** | {_ts(inc_start)} → {_ts(inc_end)} |")

        if self._has_context:
            cs = cfg.context_start_actual()
            ce = cfg.context_end_actual()
            if cs and ce:
                lines.append(f"| **Окно загрузки логов** | {_ts(cs)} → {_ts(ce)} |")

        zones = self.merged.zones_covered
        if zones:
            zone_labels = [_ZONE_LABELS.get(z, z) for z in zones]
            lines.append(f"| **Зоны анализа** | {' / '.join(zone_labels)} |")

        lines.append(f"| **Событий** | {len(self.merged.events)} |")
        lines.append(f"| **Причинно-следственных цепочек** | {len(self.merged.causal_chains)} |")
        lines.append(f"| **Гипотез** | {len(self.merged.hypotheses)} |")
        lines.append(f"| **Доказательств** | {len(self.merged.evidence_bank)} |")
        lines.append(f"| **Рекомендаций** | {len(self.merged.preliminary_recommendations)} |")

        if cfg.incident_context:
            ctx_preview = cfg.incident_context.strip().replace("\n", " ")[:200]
            lines.append(f"| **Контекст инцидента** | {ctx_preview} |")

        if not lines:
            return ""
        return "| Параметр | Значение |\n|----------|----------|\n" + "\n".join(lines)

    def _summary(self) -> str:
        text = _ru(self.merged.narrative, self.merged.narrative_ru)
        if not text:
            return ""
        return f"## Краткое резюме\n\n{text}"

    def _alert_coverage(self) -> str:
        if not self.merged.alert_refs:
            return ""

        # Строим имя алерта из конфига (alert_id → Alert)
        alert_map = {a.id: a.name for a in self.config.alerts}

        rows = ["| Alert | Статус | Комментарий |", "|-------|--------|-------------|"]
        for ref in sorted(self.merged.alert_refs, key=lambda r: AlertStatus.priority(r.status)):
            name = alert_map.get(ref.alert_id, ref.alert_id)
            status_label = _STATUS_LABELS.get(ref.status, ref.status.value)
            comment = ref.comment or "—"
            rows.append(f"| `{name}` | {status_label} | {comment} |")

        return "## Покрытие алертов\n\n" + "\n".join(rows)

    def _timeline(self) -> str:
        events = sorted(self.merged.events, key=lambda e: e.timestamp)
        if not events:
            return "## Хронология\n\n*Событий не обнаружено.*"

        parts = ["## Хронология"]

        if self._has_context and len(self.merged.zones_covered) > 1:
            # Разбиваем по зонам — используем временные границы инцидента как разделители
            inc_start = self.config.incident_start
            inc_end   = self.config.incident_end

            def event_zone(e) -> str:
                if inc_start and e.timestamp < inc_start:
                    return "context_before"
                if inc_end and e.timestamp > inc_end:
                    return "context_after"
                return "incident"

            zone_events: dict[str, list] = {z: [] for z in _ZONE_ORDER}
            for e in events:
                zone_events[event_zone(e)].append(e)

            for zone in _ZONE_ORDER:
                if zone not in self.merged.zones_covered:
                    continue
                label = _ZONE_LABELS[zone]
                parts.append(f"\n### — {label} —")
                evs = zone_events[zone]
                if not evs:
                    if zone == "context_before":
                        parts.append("*Событий не обнаружено. Возможно, причина инцидента"
                                     " находится за пределами окна загрузки.*")
                    else:
                        parts.append("*Событий не обнаружено.*")
                    continue
                parts.append(self._events_table(evs))
        else:
            parts.append(self._events_table(events))

        return "\n\n".join(parts)

    def _events_table(self, events: list) -> str:
        header = (
            "| Время | Важность | Сервис | Описание | Теги |\n"
            "|-------|---------|--------|---------|------|\n"
        )
        rows: list[str] = []
        for e in events:
            sev_icon = _SEV_EMOJI.get(e.severity, "")
            imp_flag = "⚠ " if e.importance >= 0.8 else ""
            desc = _ru(e.description, getattr(e, "description_ru", None))
            tags = ", ".join(e.tags) if e.tags else "—"
            rows.append(
                f"| `{_ts_short(e.timestamp)}` | {sev_icon} {e.severity.value} | "
                f"`{e.source}` | {imp_flag}{desc} | {tags} |"
            )
        return header + "\n".join(rows)

    def _causal_chains(self) -> str:
        if not self.merged.causal_chains:
            return ""

        parts = ["## Причинно-следственные цепочки"]

        # Строим карту event_id → description для человекочитаемых ссылок
        evt_map = {e.id: _ru(e.description, getattr(e, "description_ru", None))
                   for e in self.merged.events}

        conf_order = {"high": 0, "medium": 1, "low": 2}
        chains = sorted(
            self.merged.causal_chains,
            key=lambda c: conf_order.get(c.confidence, 9),
        )

        for chain in chains:
            conf_label = _CONF_LABELS.get(chain.confidence, chain.confidence)
            from_desc = evt_map.get(chain.from_event_id, chain.from_event_id)
            to_desc   = evt_map.get(chain.to_event_id,   chain.to_event_id)
            desc = _ru(chain.description, getattr(chain, "description_ru", None))
            mechanism = getattr(chain, "mechanism", None)
            mechanism_line = f"\n\n**Механизм:** {mechanism}" if mechanism else ""
            parts.append(
                f"**{conf_label}**\n\n"
                f"> `{chain.from_event_id}` {from_desc}\n"
                f"> → `{chain.to_event_id}` {to_desc}\n\n"
                f"{desc}{mechanism_line}"
            )

        return "\n\n".join(parts)

    def _hypotheses(self) -> str:
        if not self.merged.hypotheses:
            return "## Гипотезы\n\n*Гипотез не сформулировано.*"

        conf_order = {"high": 0, "medium": 1, "low": 2}
        hyps = sorted(
            self.merged.hypotheses,
            key=lambda h: conf_order.get(h.confidence, 9),
        )

        parts = ["## Гипотезы"]
        for i, h in enumerate(hyps, 1):
            conf_label = _CONF_LABELS.get(h.confidence, h.confidence)
            title = _ru(h.title, getattr(h, "title_ru", None))
            desc  = _ru(h.description, getattr(h, "description_ru", None))

            section = [f"### {i}. {title}  {conf_label}"]
            section.append(desc)

            if h.supporting_event_ids:
                section.append(
                    "**Подтверждающие события:** "
                    + ", ".join(f"`{eid}`" for eid in h.supporting_event_ids)
                )
            if h.contradicting_event_ids:
                section.append(
                    "**Опровергающие события:** "
                    + ", ".join(f"`{eid}`" for eid in h.contradicting_event_ids)
                )
            if h.related_alert_ids:
                section.append(
                    "**Связанные алерты:** "
                    + ", ".join(f"`{aid}`" for aid in h.related_alert_ids)
                )
            parts.append("\n\n".join(section))

        return "\n\n".join(parts)

    def _impact(self) -> str:
        text = _ru(self.merged.impact_summary, self.merged.impact_summary_ru)
        if not text:
            return ""
        return f"## Последствия\n\n{text}"

    def _recommendations(self) -> str:
        # Предпочитаем русские если есть
        recs_ru = self.merged.preliminary_recommendations_ru
        recs_en = self.merged.preliminary_recommendations

        recs = recs_ru if recs_ru else recs_en
        if not recs:
            return "## Рекомендации\n\n*Рекомендаций не сформулировано.*"

        items = "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))
        return f"## Рекомендации\n\n{items}"

    def _anomalies(self) -> str:
        if not self.merged.anomalies:
            return ""
        parts = ["## Аномалии"]
        for a in self.merged.anomalies:
            desc = _ru(a.description, getattr(a, "description_ru", None))
            related = ""
            if a.related_event_ids:
                related = " (" + ", ".join(f"`{eid}`" for eid in a.related_event_ids) + ")"
            parts.append(f"- {desc}{related}")
        return "\n".join(parts)

    def _evidence(self) -> str:
        if not self.merged.evidence_bank:
            return ""
        parts = ["## Доказательная база"]
        parts.append("Дословные строки логов, подтверждающие ключевые события:\n")

        # Группируем по severity (critical/high сначала)
        sev_order = {s: Severity.priority(s) for s in Severity}
        ev_sorted = sorted(
            self.merged.evidence_bank,
            key=lambda e: (Severity.priority(e.severity), e.timestamp),
        )

        parts.append("```")
        for ev in ev_sorted:
            parts.append(f"[{ev.timestamp.strftime('%H:%M:%S')}] [{ev.source}] {ev.raw_line}")
        parts.append("```")

        return "\n".join(parts)

    def _gaps(self) -> str:
        if not self.merged.gaps:
            return ""
        parts = ["## Пробелы в данных"]
        for g in self.merged.gaps:
            desc = _ru(g.description, getattr(g, "description_ru", None))
            parts.append(
                f"- **{_ts_short(g.start)} → {_ts_short(g.end)}**: {desc}"
            )
        return "\n".join(parts)

    def _footer(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"---\n\n*Сформировано автоматически: {now}*"
