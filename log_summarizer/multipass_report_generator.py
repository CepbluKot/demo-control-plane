"""MultipassReportGenerator — финальный отчёт: каждая секция — отдельный LLM-вызов.

Принцип: секции независимы, каждая получает только нужный кусок MergedAnalysis.
Секции 3–13 генерируются строго последовательно (1 запрос к LLM в каждый момент).
Секция 2 (Резюме) — последняя, получает «скелет» из готовых секций.

~11 LLM-вызовов при обычном прогоне (sec7 — программная заглушка, без LLM).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import ContextOverflowError, LLMClient
from log_summarizer.models import MergedAnalysis
from log_summarizer.utils.logging import get_logger

logger = get_logger("multipass_report")

_PLACEHOLDER = "[Секция не сгенерирована — данных нет или произошла ошибка]"

_LANG = (
    "Пиши на русском языке. "
    "Технические термины (имена сервисов, pod, namespace, Kubernetes-объекты, "
    "OOM, SIGTERM, коды ошибок, имена метрик, CLI-команды) оставляй как есть.\n\n"
)


def _ru(eng: Optional[str], ru: Optional[str]) -> str:
    return (ru or eng or "").strip()


def _conf_float(c: str) -> float:
    return {"high": 1.0, "medium": 0.5, "low": 0.2}.get(str(c), 0.5)


def _jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


class MultipassReportGenerator:
    """Генерирует финальный Markdown-отчёт: каждая секция — отдельный LLM-вызов.

    Args:
        llm: LLM-клиент.
        config: Конфигурация пайплайна.
        run_dir: Папка для сохранения промптов и результатов.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: PipelineConfig,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self._run_dir = run_dir

    # ── Публичный API ─────────────────────────────────────────────────

    async def generate(
        self,
        merged: MergedAnalysis,
        degradation: dict | None = None,
    ) -> str:
        """Генерирует финальный отчёт из 13 секций последовательно.

        LLM-вызовы строго по одному — без параллелизма.
        Секция 2 (Резюме) — последней, получает скелет из остальных.

        Args:
            merged: Финальный MergedAnalysis.
            degradation: Статистика деградации пайплайна
                         {"processing_error_batches": N, "programmatic_merges": N}.

        Returns:
            Готовый Markdown-документ.
        """
        # Общие данные, вычисляются один раз
        events_index = {
            e.id: _ru(e.description, getattr(e, "description_ru", None))
            for e in merged.events
        }
        hypotheses_short = [
            {
                "title": _ru(h.title, getattr(h, "title_ru", None)),
                "confidence": h.confidence,
            }
            for h in merged.hypotheses
        ]

        # Секция 1 — программно, без LLM
        sec1  = self._sec_incident_context()

        # Секции 3–13 — строго последовательно, по одному LLM-вызову
        sec3  = await self._sec_data_coverage(merged)
        sec4  = await self._sec_chronology(merged)
        sec5  = await self._sec_causal_chains(merged, events_index)
        sec6  = await self._sec_alert_links(merged, events_index)
        sec7  = await self._sec_metrics(merged)
        sec8  = await self._sec_hypotheses(merged, events_index)
        sec9  = await self._sec_conflicts(merged, events_index)
        sec10 = await self._sec_gaps(merged, events_index)
        sec11 = await self._sec_impact(merged)
        sec12 = await self._sec_recommendations(merged, hypotheses_short)
        sec13 = await self._sec_limitations(merged, hypotheses_short, degradation or {})

        # Секция 2 — последняя
        sec2  = await self._sec_summary(merged, sec13)

        # Сборка
        order = [
            ("1. Контекст инцидента",               sec1),
            ("2. Резюме инцидента",                  sec2),
            ("3. Покрытие данных",                   sec3),
            ("4. Хронология событий",                sec4),
            ("5. Причинно-следственные цепочки",     sec5),
            ("6. Связь с алертами",                  sec6),
            ("7. Аномалии метрик",                   sec7),
            ("8. Гипотезы первопричин",              sec8),
            ("9. Конфликтующие версии",              sec9),
            ("10. Разрывы в цепочках",               sec10),
            ("11. Масштаб и влияние",                sec11),
            ("12. Рекомендации для SRE",             sec12),
            ("13. Уровень уверенности и ограничения", sec13),
        ]

        parts = [f"## {title}\n\n{text}" for title, text in order]
        report = "\n\n---\n\n".join(parts)
        self._save("report_multipass.md", report)
        return report

    # ── Секции ────────────────────────────────────────────────────────

    def _sec_incident_context(self) -> str:
        """Секция 1 — программно, без LLM."""
        cfg = self.config
        lines = [cfg.incident_context.strip() or "Контекст не указан."]
        if cfg.incident_start and cfg.incident_end:
            lines.append(
                f"\n**Период инцидента:** {cfg.incident_start.isoformat()} "
                f"→ {cfg.incident_end.isoformat()}"
            )
        if cfg.has_context_window():
            cs = cfg.context_start_actual()
            ce = cfg.context_end_actual()
            lines.append(
                f"**Окно загрузки логов:** {cs.isoformat()} → {ce.isoformat()}"
            )
        if cfg.alerts:
            alert_lines = [
                f"- `{a.id}` **{a.name}** ({a.severity.value})"
                + (f" — {a.fired_at.isoformat()}" if a.fired_at else "")
                for a in cfg.alerts
            ]
            lines.append("\n**Алерты:**\n" + "\n".join(alert_lines))
        return "\n".join(lines)

    async def _sec_data_coverage(self, merged: MergedAnalysis) -> str:
        """Секция 3 — покрытие данных."""
        t0, t1 = merged.time_range
        zones = ", ".join(merged.zones_covered) if merged.zones_covered else "incident"
        sql_preview = (self.config.logs_sql_template or "")[:400].strip()
        user = (
            f"Период: {t0.isoformat()} → {t1.isoformat()}\n"
            f"Зоны анализа: {zones}\n"
            f"Событий извлечено: {len(merged.events)}\n"
            f"Доказательств (verbatim строк): {len(merged.evidence_bank)}\n"
            f"SQL-запрос (фрагмент):\n```sql\n{sql_preview}\n```\n"
        )
        system = _LANG + (
            "Ты генерируешь раздел «Покрытие данных» отчёта об инциденте.\n"
            "Укажи: период анализа, какие зоны покрыты (context_before/incident/context_after), "
            "объём обработанных данных, что НЕ покрыто если это следует из SQL-запроса. "
            "Будь кратким: 3–5 предложений."
        )
        return await self._call("data_coverage", system, user)

    async def _sec_chronology(self, merged: MergedAnalysis) -> str:
        """Секция 4 — хронология событий."""
        events = sorted(merged.events, key=lambda e: e.timestamp)

        # Обрезка если слишком много
        def _events_payload(evs) -> str:
            items = []
            for e in evs:
                eq = next(
                    (ev.raw_line for ev in merged.evidence_bank
                     if getattr(ev, "linked_event_id", None) == e.id),
                    None,
                )
                items.append({
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "source": e.source,
                    "description": _ru(e.description, getattr(e, "description_ru", None)),
                    "severity": e.severity.value,
                    "importance": e.importance,
                    "tags": e.tags,
                    **({"evidence_quote": eq} if eq else {}),
                })
            return _jdump(items)

        payload = _events_payload(events)
        trimmed_note = ""

        system = _LANG + (
            "Ты генерируешь раздел «Хронология событий» отчёта об инциденте.\n"
            "Для каждого события: точный timestamp, источник, описание, severity.\n"
            "Пометь [ФАКТ] если есть evidence_quote, иначе [СОБЫТИЕ].\n"
            "Для фактов приведи дословную цитату из лога в блоке цитаты.\n"
            "Сортировка строго хронологическая. События с importance ≥ 0.8 отмечай символом ⚠.\n"
            "Если есть зоны context_before/incident/context_after — разбей на подразделы."
        )
        user = f"События:\n{payload}"
        if trimmed_note:
            user += f"\n\n{trimmed_note}"

        try:
            return await self._call("chronology", system, user)
        except ContextOverflowError:
            # Обрезка 1: importance > 0.7 + top-20
            high = [e for e in events if e.importance > 0.7]
            rest = sorted(
                [e for e in events if e.importance <= 0.7],
                key=lambda e: -e.importance,
            )[:20]
            trimmed = sorted(high + rest, key=lambda e: e.timestamp)
            trimmed_note = f"[Показаны только ключевые события. Полная хронология в итоговом JSON. Показано {len(trimmed)} из {len(events)}]"
            user = f"События:\n{_events_payload(trimmed)}\n\n{trimmed_note}"
            try:
                return await self._call("chronology", system, user)
            except ContextOverflowError:
                # Обрезка 2: только importance > 0.7
                only_high = sorted([e for e in events if e.importance > 0.7], key=lambda e: e.timestamp)
                trimmed_note = f"[Показаны только события с importance > 0.7 ({len(only_high)} из {len(events)})]"
                user = f"События:\n{_events_payload(only_high)}\n\n{trimmed_note}"
                return await self._call_or_stub("chronology", system, user)

    async def _sec_causal_chains(self, merged: MergedAnalysis, events_index: dict) -> str:
        """Секция 5 — причинные цепочки."""
        if not merged.causal_chains:
            return "Причинно-следственных связей не обнаружено."

        chains = sorted(merged.causal_chains, key=lambda c: -_conf_float(c.confidence))

        def _payload(chs) -> str:
            return _jdump([{
                "from_event_id": c.from_event_id,
                "to_event_id": c.to_event_id,
                "description": _ru(c.description, getattr(c, "description_ru", None)),
                "mechanism": c.mechanism,
                "confidence": c.confidence,
            } for c in chs])

        system = _LANG + (
            "Ты генерируешь раздел «Причинно-следственные цепочки» отчёта об инциденте.\n"
            "Опиши каждую связь не как «A → B», а с конкретным механизмом: ПОЧЕМУ именно "
            "причина привела к следствию. Укажи уровень уверенности.\n"
            "Построй связный нарратив: сначала основная цепочка (root cause → следствия), "
            "затем побочные. Кросс-зональные связи (context_before → incident) — самые ценные, "
            "выдели их."
        )
        user = (
            f"Причинные связи:\n{_payload(chains)}\n\n"
            f"Индекс событий (id → описание):\n{_jdump(events_index)}"
        )

        try:
            return await self._call("causal_chains", system, user)
        except ContextOverflowError:
            # Оставить confidence > 0.5
            filtered = [c for c in chains if _conf_float(c.confidence) > 0.5]
            user = (
                f"Причинные связи (confidence > 0.5):\n{_payload(filtered)}\n\n"
                f"Индекс событий:\n{_jdump(events_index)}"
            )
            return await self._call_or_stub("causal_chains", system, user)

    async def _sec_alert_links(self, merged: MergedAnalysis, events_index: dict) -> str:
        """Секция 6 — связь с алертами."""
        if not merged.alert_refs:
            return "Алертов для анализа не было."

        alert_map = {a.id: a.name for a in self.config.alerts}
        payload = _jdump([{
            "alert_id": r.alert_id,
            "alert_name": alert_map.get(r.alert_id, r.alert_id),
            "status": r.status.value,
            "comment": r.comment,
        } for r in merged.alert_refs])

        system = _LANG + (
            "Ты генерируешь раздел «Связь с алертами» отчёта об инциденте.\n"
            "Для каждого алерта укажи статус одним из:\n"
            "- ОБЪЯСНЁН — найдены события, полностью объясняющие алерт\n"
            "- ЧАСТИЧНО ОБЪЯСНЁН — частичные данные; укажи чего не хватает\n"
            "- НЕ ОБЪЯСНЁН — объяснение не найдено; укажи где искали\n"
            "Ссылайся на конкретные события из индекса."
        )
        user = (
            f"Алерты:\n{payload}\n\n"
            f"Индекс событий:\n{_jdump(events_index)}"
        )
        return await self._call_or_stub("alert_links", system, user)

    async def _sec_metrics(self, merged: MergedAnalysis) -> str:
        """Секция 7 — метрики (программная заглушка — метрик нет в MergedAnalysis)."""
        return (
            "Метрики не предоставлены. "
            "Для более полного анализа рекомендуется повторить с метриками CPU, memory, latency "
            "для затронутых сервисов."
        )

    async def _sec_hypotheses(self, merged: MergedAnalysis, events_index: dict) -> str:
        """Секция 8 — гипотезы первопричин."""
        if not merged.hypotheses:
            return "Гипотез не сформулировано."

        hyps = sorted(merged.hypotheses, key=lambda h: -_conf_float(h.confidence))

        def _payload(hs) -> str:
            return _jdump([{
                "id": h.id,
                "title": _ru(h.title, getattr(h, "title_ru", None)),
                "description": _ru(h.description, getattr(h, "description_ru", None)),
                "confidence": h.confidence,
                "supporting_event_ids": h.supporting_event_ids,
                "contradicting_event_ids": h.contradicting_event_ids,
                "related_alert_ids": h.related_alert_ids,
            } for h in hs])

        system = _LANG + (
            "Ты генерируешь раздел «Гипотезы первопричин» отчёта об инциденте.\n"
            "Сгруппируй гипотезы по алертам (related_alert_ids). "
            "Внутри группы — ранжируй от высокого confidence к низкому.\n"
            "Для каждой гипотезы:\n"
            "- Название и уровень уверенности с ОБОСНОВАНИЕМ (не просто число)\n"
            "- Развёрнутое обоснование (3–5 предложений)\n"
            "- Подтверждающие и опровергающие события (ссылки из индекса)\n"
            "Гипотезы с низким confidence — в конце, свёрнуто."
        )
        user = f"Гипотезы:\n{_payload(hyps)}\n\nИндекс событий:\n{_jdump(events_index)}"

        try:
            return await self._call("hypotheses", system, user)
        except ContextOverflowError:
            # Убрать гипотезы с confidence "low"
            filtered = [h for h in hyps if _conf_float(h.confidence) >= 0.5]
            user = f"Гипотезы (только medium/high confidence):\n{_payload(filtered)}\n\nИндекс событий:\n{_jdump(events_index)}"
            return await self._call_or_stub("hypotheses", system, user)

    async def _sec_conflicts(self, merged: MergedAnalysis, events_index: dict) -> str:
        """Секция 9 — конфликтующие версии / аномалии."""
        if not merged.anomalies:
            return "Конфликтующих интерпретаций не обнаружено."

        payload = _jdump([{
            "description": _ru(a.description, getattr(a, "description_ru", None)),
            "related_event_ids": a.related_event_ids,
        } for a in merged.anomalies])

        system = _LANG + (
            "Ты генерируешь раздел «Конфликтующие версии» отчёта об инциденте.\n"
            "Опиши аномалии и противоречия в данных: что выглядит необычно или "
            "не вписывается в основную версию. Ссылайся на конкретные события из индекса."
        )
        user = f"Аномалии:\n{payload}\n\nИндекс событий:\n{_jdump(events_index)}"
        return await self._call_or_stub("conflicts", system, user)

    async def _sec_gaps(self, merged: MergedAnalysis, events_index: dict) -> str:
        """Секция 10 — разрывы в цепочках."""
        if not merged.gaps:
            return "Значимых разрывов в причинно-следственных цепочках не обнаружено."

        payload = _jdump([{
            "start": g.start.isoformat(),
            "end": g.end.isoformat(),
            "description": _ru(g.description, getattr(g, "description_ru", None)),
        } for g in merged.gaps])

        system = _LANG + (
            "Ты генерируешь раздел «Разрывы в цепочках» отчёта об инциденте.\n"
            "Для каждого разрыва объясни: что происходило до и после, "
            "какие данные отсутствуют и как это влияет на уверенность в диагнозе."
        )
        user = f"Разрывы:\n{payload}\n\nИндекс событий:\n{_jdump(events_index)}"
        return await self._call_or_stub("gaps", system, user)

    async def _sec_impact(self, merged: MergedAnalysis) -> str:
        """Секция 11 — масштаб и влияние."""
        impact_text = _ru(merged.impact_summary, merged.impact_summary_ru) or "Данных о масштабе нет."
        system = _LANG + (
            "Ты генерируешь раздел «Масштаб и влияние» отчёта об инциденте.\n"
            "Укажи: затронутые сервисы, пострадавшие пользовательские сценарии, "
            "количественные данные об ошибках, общую длительность, общую оценку серьёзности. "
            "Будь конкретным, используй цифры и названия сервисов."
        )
        user = f"Данные о влиянии:\n{impact_text}"
        return await self._call_or_stub("impact", system, user)

    async def _sec_recommendations(self, merged: MergedAnalysis, hypotheses_short: list) -> str:
        """Секция 12 — рекомендации для SRE."""
        recs = merged.preliminary_recommendations_ru or merged.preliminary_recommendations
        if not recs:
            return "Рекомендаций не сформулировано."

        system = _LANG + (
            "Ты генерируешь раздел «Рекомендации для SRE» отчёта об инциденте.\n"
            "Сгруппируй по приоритету:\n"
            "- P0 — немедленно (инцидент продолжается или повторится)\n"
            "- P1 — в ближайшее время (серьёзный риск)\n"
            "- P2 — улучшения (предотвращение повторения)\n"
            "Для каждой рекомендации: конкретное действие, зачем (ссылка на гипотезу), ожидаемый эффект.\n"
            "«Увеличить пул соединений с 50 до 150» — хорошо. «Посмотреть на базу данных» — плохо."
        )
        user = (
            f"Черновые рекомендации:\n{_jdump(recs)}\n\n"
            f"Гипотезы для привязки (title, confidence):\n{_jdump(hypotheses_short)}"
        )
        return await self._call_or_stub("recommendations", system, user)

    async def _sec_limitations(self, merged: MergedAnalysis, hypotheses_short: list, degradation: dict) -> str:
        """Секция 13 — уровень уверенности и ограничения анализа."""
        active_hyp = hypotheses_short  # все считаем активными
        confidences = [_conf_float(h["confidence"]) for h in active_hyp]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        if avg_conf > 0.7 and len(merged.gaps) < 3:
            conf_level = "высокий"
        elif avg_conf > 0.4:
            conf_level = "средний"
        else:
            conf_level = "низкий"

        gaps_payload = _jdump([{
            "start": g.start.isoformat(),
            "end": g.end.isoformat(),
            "description": _ru(g.description, getattr(g, "description_ru", None)),
        } for g in merged.gaps])

        system = _LANG + (
            "Ты генерируешь раздел «Уровень уверенности и ограничения анализа» отчёта.\n"
            "Будь честным. Укажи:\n"
            "- Общий уровень уверенности с обоснованием\n"
            "- Конкретные ограничения: какие сервисы не покрыты, где данных мало, где высокий шум\n"
            "- Какие гипотезы имеют низкий confidence и почему\n"
            "- Если были технические проблемы при анализе (processing_error_batches > 0 или "
            "programmatic_merges > 0) — явно упомяни это как ограничение: какие части логов "
            "могли быть проанализированы неполно или без LLM\n"
            "В конце явная пометка: «Этот отчёт — помощь в расследовании, он не заменяет инженерное суждение SRE.»"
        )

        degradation_note = ""
        if degradation.get("processing_error_batches", 0) or degradation.get("programmatic_merges", 0):
            degradation_note = f"\nТехнические проблемы при анализе:\n{_jdump(degradation)}"

        user = (
            f"Предварительный уровень уверенности: {conf_level}\n"
            f"Разрывы в данных:\n{gaps_payload}\n"
            f"Уровни уверенности гипотез:\n{_jdump(hypotheses_short)}"
            f"{degradation_note}"
        )
        return await self._call_or_stub("limitations", system, user)

    async def _sec_summary(self, merged: MergedAnalysis, limitations_section: str) -> str:
        """Секция 2 — резюме. Генерируется последней, получает скелет."""
        # Скелет из программно вычисленных фактов
        events_sorted = sorted(merged.events, key=lambda e: e.timestamp)
        first_evt = events_sorted[0] if events_sorted else None
        last_evt  = events_sorted[-1] if events_sorted else None

        leading_hyp = max(
            merged.hypotheses,
            key=lambda h: _conf_float(h.confidence),
            default=None,
        )

        from log_summarizer.models import AlertStatus
        count_explained   = sum(1 for r in merged.alert_refs if r.status == AlertStatus.EXPLAINED)
        count_partial     = sum(1 for r in merged.alert_refs if r.status == AlertStatus.PARTIAL)
        count_unexplained = sum(1 for r in merged.alert_refs if r.status in (AlertStatus.NOT_EXPLAINED, AlertStatus.NOT_SEEN))

        skeleton = {
            "первое_событие": (
                f"{first_evt.timestamp.isoformat()} — "
                + _ru(first_evt.description, getattr(first_evt, "description_ru", None))
            ) if first_evt else "нет данных",
            "последнее_событие": (
                f"{last_evt.timestamp.isoformat()} — "
                + _ru(last_evt.description, getattr(last_evt, "description_ru", None))
            ) if last_evt else "нет данных",
            "ведущая_гипотеза": (
                f"{_ru(leading_hyp.title, getattr(leading_hyp, 'title_ru', None))} "
                f"(уверенность: {leading_hyp.confidence})"
            ) if leading_hyp else "нет гипотез",
            "статусы_алертов": (
                f"объяснено: {count_explained}, частично: {count_partial}, "
                f"не объяснено: {count_unexplained}"
            ),
            "затронутые_сервисы": merged.impact_summary_ru or merged.impact_summary or "нет данных",
        }

        # Извлекаем уровень уверенности из секции 13 (первое слово после "уровень уверенности")
        conf_hint = ""
        lower_lim = limitations_section.lower()
        for marker in ("высокий", "средний", "низкий"):
            if marker in lower_lim:
                conf_hint = marker
                break
        skeleton["общая_уверенность"] = conf_hint or "средний"

        system = _LANG + (
            "Ты генерируешь раздел «Резюме инцидента» — первый раздел, который читает SRE.\n"
            "Напиши 3–5 предложений:\n"
            "- Что произошло (кратко)\n"
            "- Когда (временные рамки)\n"
            "- Какие сервисы затронуты\n"
            "- Наиболее вероятная первопричина (со ссылкой на ведущую гипотезу)\n"
            "- Текущий статус (разрешён / продолжается / неизвестен)\n"
            "Будь предельно кратким. Это TL;DR для занятого SRE."
        )
        user = _jdump(skeleton)
        return await self._call_or_stub("summary", system, user)

    # ── Вспомогательные ───────────────────────────────────────────────

    async def _call(self, name: str, system: str, user: str) -> str:
        """LLM-вызов; пробрасывает ContextOverflowError для обработки выше."""
        result = await self.llm.call_text(
            system=system,
            user=user,
            temperature=self.config.temperature_report,
        )
        self._save(f"multipass_{name}.md", result)
        return result

    async def _call_or_stub(self, name: str, system: str, user: str) -> str:
        """LLM-вызов с fallback на заглушку при любой ошибке."""
        try:
            return await self._call(name, system, user)
        except Exception as exc:
            logger.error("Section '%s' failed: %s — using placeholder", name, exc)
            return _PLACEHOLDER

    def _save(self, filename: str, content: str) -> None:
        if self._run_dir is None:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / filename).write_text(content, encoding="utf-8")
