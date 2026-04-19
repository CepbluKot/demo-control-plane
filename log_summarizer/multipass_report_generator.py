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
from log_summarizer.utils.progress import ProgressTracker, fmt_dur

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
        self._progress: Optional[ProgressTracker] = None  # живёт только во время generate()

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
        import time as _time
        _t0_report = _time.monotonic()

        # 13 LLM-вызовов: sec3,4,5,5a,6,8,9,10,11,12,13,14 + sec2 (последний)
        _LLM_TOTAL = 13
        self._progress = ProgressTracker(total=_LLM_TOTAL, label="REPORT")

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

        t0_rep, t1_rep = merged.time_range
        logger.info(
            "  Генерация отчёта: %d событий · %d гипотез · %d цепочек  |  период %s→%s",
            len(merged.events), len(merged.hypotheses), len(merged.causal_chains),
            t0_rep.strftime("%H:%M:%S"), t1_rep.strftime("%H:%M:%S"),
        )

        # Секция 1 — программно, без LLM
        sec1  = self._sec_incident_context()

        # Секции 3–14 — строго последовательно, по одному LLM-вызову.
        # sec5 (технические цепочки) и sec5a (объяснение для человека) генерируются
        # первыми, чтобы все последующие секции могли использовать их как контекст.
        sec3  = await self._sec_data_coverage(merged)
        sec4  = await self._sec_chronology(merged)
        sec5  = await self._sec_causal_chains(merged, events_index)
        sec5a = await self._sec_root_cause_explanation(merged, sec5)

        # sec6–sec11 получают цепочку как контекст
        sec6  = await self._sec_alert_links(merged, events_index, sec5a)
        sec7  = await self._sec_metrics(merged)
        sec8  = await self._sec_hypotheses(merged, events_index, sec5a)
        sec9  = await self._sec_conflicts(merged, events_index, sec5a)
        sec10 = await self._sec_gaps(merged, events_index, sec5a)
        sec11 = await self._sec_impact(merged, sec5a)
        sec12 = await self._sec_recommendations(merged, hypotheses_short, sec5a)
        sec13 = await self._sec_limitations(merged, hypotheses_short, degradation or {})
        sec14 = await self._sec_coverage_recommendations(merged, events_index)

        # Секция 2 — последняя
        sec2  = await self._sec_summary(merged, sec13, sec5a)

        # Справочник событий — программно, без LLM
        events_ref = self._sec_events_reference(merged)

        # Сборка
        order = [
            ("1. Контекст инцидента",               sec1),
            ("2. Резюме инцидента",                  sec2),
            ("3. Покрытие данных",                   sec3),
            ("4. Хронология событий",                sec4),
            ("5. Причинно-следственные цепочки",     sec5),
            ("5а. Объяснение первопричины",          sec5a),
            ("6. Связь с алертами",                  sec6),
            ("7. Аномалии метрик",                   sec7),
            ("8. Гипотезы первопричин",              sec8),
            ("9. Конфликтующие версии",              sec9),
            ("10. Разрывы в цепочках",               sec10),
            ("11. Масштаб и влияние",                sec11),
            ("12. Рекомендации для SRE",             sec12),
            ("13. Уровень уверенности и ограничения", sec13),
            ("14. Рекомендации по расширению анализа", sec14),
            ("Приложение: Справочник событий",       events_ref),
        ]

        parts = [f"## {title}\n\n{text}" for title, text in order if text.strip()]
        report = "\n\n---\n\n".join(parts)

        if self._run_dir:
            rpath = self._run_dir / "report_multipass.md"
            self._save("report_multipass.md", report)
            logger.info(
                "  %s  →  %s",
                self._progress.summary() if self._progress else "REPORT done",
                rpath,
            )
        else:
            self._save("report_multipass.md", report)
            logger.info(
                "  %s",
                self._progress.summary() if self._progress else "REPORT done",
            )

        self._progress = None  # сброс
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
        if cfg.total_log_rows:
            lines.append(f"**Обработано строк логов:** {cfg.total_log_rows:,}")
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
        sql_preview = (self.config.logs_sql_templates[0] if self.config.logs_sql_templates else "")[:400].strip()
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
            "Ты генерируешь раздел «Причинно-следственные цепочки» отчёта об инциденте.\n\n"
            "ОБЯЗАТЕЛЬНАЯ СТРУКТУРА РАЗДЕЛА:\n\n"
            "### Цепочка возникновения инцидента\n"
            "Пронумерованная последовательность от первопричины до пика инцидента.\n"
            "Каждый шаг: время → событие → ПОЧЕМУ оно привело к следующему.\n"
            "Если есть кросс-зональные связи (context_before → incident) — они идут первыми,\n"
            "это самые ценные ссылки: изменение/деплой ДО инцидента, которое его породило.\n"
            "Пример формата:\n"
            "1. `HH:MM` **[evt-id]** Название события — механизм перехода к следующему шагу\n"
            "2. `HH:MM` **[evt-id]** ...\n\n"
            "### Побочные цепочки\n"
            "Остальные причинно-следственные связи, не входящие в основную цепочку.\n"
            "Для каждой: уровень уверенности и конкретный механизм.\n\n"
            "Если данных для построения цепочки недостаточно — явно укажи, какого звена не хватает."
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

    async def _sec_root_cause_explanation(self, merged: MergedAnalysis, sec5_text: str) -> str:
        """Секция 5а — объяснение первопричины для человека (3-4 абзаца).

        Берёт sec5 (причинно-следственные цепочки) как контекст и пишет
        понятное объяснение: симптомы → деградация → корень.
        """
        leading_hyp = max(
            merged.hypotheses,
            key=lambda h: _conf_float(h.confidence),
            default=None,
        )
        hyp_text = ""
        if leading_hyp:
            hyp_text = (
                f"Ведущая гипотеза: {_ru(leading_hyp.title, getattr(leading_hyp, 'title_ru', None))} "
                f"(уверенность: {leading_hyp.confidence})\n"
                f"{_ru(leading_hyp.description, getattr(leading_hyp, 'description_ru', None))}"
            )

        system = _LANG + (
            "Ты пишешь раздел «Объяснение первопричины» для SRE-отчёта.\n\n"
            "Задача: понятно объяснить дежурному инженеру ЧТО произошло и ПОЧЕМУ.\n"
            "Не используй технический жаргон без расшифровки.\n\n"
            "Структура (4 абзаца):\n"
            "1. **Симптомы** — что наблюдал дежурный (алерты, ошибки, жалобы)\n"
            "2. **Цепочка деградации** — как одно событие привело к другому (шаг за шагом)\n"
            "3. **Корень проблемы** — где и почему система дала сбой\n"
            "4. **Почему именно сейчас** — что изменилось или что достигло предела\n\n"
            "Каждый абзац: 2–4 предложения. Говори уверенно там, где данные есть; "
            "явно обозначай неопределённость там, где данных нет.\n"
            "Не повторяй технические ID событий — пиши человеческим языком."
        )
        user = f"Причинно-следственные цепочки:\n{sec5_text}"
        if hyp_text:
            user += f"\n\n{hyp_text}"

        return await self._call_or_stub("root_cause_explanation", system, user)

    async def _sec_alert_links(
        self,
        merged: MergedAnalysis,
        events_index: dict,
        root_cause_text: str = "",
    ) -> str:
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
            "Ссылайся на конкретные события из индекса.\n"
            "Если доступна цепочка первопричины — покажи, в каком звене цепочки "
            "алерт сработал: это самая ценная информация для SRE."
        )
        user = (
            f"Алерты:\n{payload}\n\n"
            f"Индекс событий:\n{_jdump(events_index)}"
        )
        if root_cause_text.strip():
            user += f"\n\nЦепочка деградации (контекст):\n{root_cause_text}"

        return await self._call_or_stub("alert_links", system, user)

    async def _sec_metrics(self, merged: MergedAnalysis) -> str:
        """Секция 7 — метрики (программная заглушка — метрик нет в MergedAnalysis)."""
        return (
            "Метрики не предоставлены. "
            "Для более полного анализа рекомендуется повторить с метриками CPU, memory, latency "
            "для затронутых сервисов."
        )

    async def _sec_hypotheses(
        self,
        merged: MergedAnalysis,
        events_index: dict,
        root_cause_text: str = "",
    ) -> str:
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
            "Гипотезы с низким confidence — в конце, свёрнуто.\n"
            "Если доступна цепочка деградации — сверь каждую гипотезу с ней: "
            "подтверждает ли цепочка гипотезу или опровергает? "
            "Явно укажи это в обосновании."
        )
        user = f"Гипотезы:\n{_payload(hyps)}\n\nИндекс событий:\n{_jdump(events_index)}"
        if root_cause_text.strip():
            user += f"\n\nЦепочка деградации (для сверки с гипотезами):\n{root_cause_text}"

        try:
            return await self._call("hypotheses", system, user)
        except ContextOverflowError:
            filtered = [h for h in hyps if _conf_float(h.confidence) >= 0.5]
            user = f"Гипотезы (только medium/high confidence):\n{_payload(filtered)}\n\nИндекс событий:\n{_jdump(events_index)}"
            if root_cause_text.strip():
                user += f"\n\nЦепочка деградации:\n{root_cause_text}"
            return await self._call_or_stub("hypotheses", system, user)

    async def _sec_conflicts(
        self,
        merged: MergedAnalysis,
        events_index: dict,
        root_cause_text: str = "",
    ) -> str:
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
            "не вписывается в основную версию. Ссылайся на конкретные события из индекса.\n"
            "Если доступна цепочка деградации — для каждой аномалии явно укажи: "
            "вписывается ли она в эту цепочку, или это альтернативная версия событий? "
            "Если альтернативная — чем она опасна и почему не стала основной?"
        )
        user = f"Аномалии:\n{payload}\n\nИндекс событий:\n{_jdump(events_index)}"
        if root_cause_text.strip():
            user += f"\n\nОсновная цепочка деградации:\n{root_cause_text}"

        return await self._call_or_stub("conflicts", system, user)

    async def _sec_gaps(
        self,
        merged: MergedAnalysis,
        events_index: dict,
        root_cause_text: str = "",
    ) -> str:
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
            "какие данные отсутствуют и как это влияет на уверенность в диагнозе.\n"
            "Если доступна цепочка деградации — покажи, в каком звене цепочки "
            "находится этот разрыв: пропущено звено в начале (неизвестна первопричина), "
            "в середине (неизвестен механизм перехода) или в конце (неизвестно разрешение)? "
            "Это определяет, насколько критичен разрыв для диагноза."
        )
        user = f"Разрывы:\n{payload}\n\nИндекс событий:\n{_jdump(events_index)}"
        if root_cause_text.strip():
            user += f"\n\nЦепочка деградации (для локализации разрыва):\n{root_cause_text}"

        return await self._call_or_stub("gaps", system, user)

    async def _sec_impact(
        self,
        merged: MergedAnalysis,
        root_cause_text: str = "",
    ) -> str:
        """Секция 11 — масштаб и влияние."""
        impact_text = _ru(merged.impact_summary, merged.impact_summary_ru) or "Данных о масштабе нет."
        system = _LANG + (
            "Ты генерируешь раздел «Масштаб и влияние» отчёта об инциденте.\n"
            "Укажи: затронутые сервисы, пострадавшие пользовательские сценарии, "
            "количественные данные об ошибках, общую длительность, общую оценку серьёзности. "
            "Будь конкретным, используй цифры и названия сервисов.\n"
            "Если доступна цепочка деградации — объясни, как именно первопричина "
            "распространилась и какие сервисы затронула на каждом шаге."
        )
        user = f"Данные о влиянии:\n{impact_text}"
        if root_cause_text.strip():
            user += f"\n\nЦепочка деградации (как распространялась проблема):\n{root_cause_text}"

        return await self._call_or_stub("impact", system, user)

    async def _sec_recommendations(
        self,
        merged: MergedAnalysis,
        hypotheses_short: list,
        root_cause_text: str = "",
    ) -> str:
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
            "Для каждой рекомендации: конкретное действие, зачем (привязка к конкретному звену "
            "причинно-следственной цепочки), ожидаемый эффект.\n"
            "«Увеличить пул соединений с 50 до 150» — хорошо. «Посмотреть на базу данных» — плохо.\n"
            "Если объяснение первопричины доступно — каждая рекомендация должна явно прерывать "
            "одно из звеньев описанной цепочки деградации."
        )
        user = (
            f"Черновые рекомендации:\n{_jdump(recs)}\n\n"
            f"Гипотезы для привязки (title, confidence):\n{_jdump(hypotheses_short)}"
        )
        if root_cause_text.strip():
            user += f"\n\nОбъяснение первопричины (цепочка деградации):\n{root_cause_text}"

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

    async def _sec_summary(
        self,
        merged: MergedAnalysis,
        limitations_section: str,
        root_cause_text: str = "",
    ) -> str:
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
            "- Наиболее вероятная первопричина — используй объяснение цепочки деградации "
            "если оно доступно, не пересказывай технические ID\n"
            "- Текущий статус (разрешён / продолжается / неизвестен)\n"
            "Будь предельно кратким. Это TL;DR для занятого SRE."
        )
        user = _jdump(skeleton)
        if root_cause_text.strip():
            user += f"\n\nОбъяснение первопричины (цепочка деградации):\n{root_cause_text}"

        return await self._call_or_stub("summary", system, user)

    # ── Вспомогательные ───────────────────────────────────────────────

    async def _sec_coverage_recommendations(
        self, merged: MergedAnalysis, events_index: dict
    ) -> str:
        """Секция 14 — рекомендации по расширению анализа.

        Отвечает на два вопроса:
        1. Нужно ли расширить временное окно (и в какую сторону)?
        2. Нужно ли добавить логи других сервисов?
        """
        cfg = self.config
        ctx_start = cfg.context_start_actual()
        ctx_end   = cfg.context_end_actual()
        inc_start = cfg.incident_start
        inc_end   = cfg.incident_end

        events_sorted = sorted(merged.events, key=lambda e: e.timestamp)
        first_evt = events_sorted[0]  if events_sorted else None
        last_evt  = events_sorted[-1] if events_sorted else None

        # Сервисы, реально присутствующие в событиях
        seen_services = sorted({e.source for e in merged.events if e.source})

        # Близость первого события к левой границе окна: сигнал что надо смотреть глубже
        proximity_warning = ""
        if first_evt and ctx_start:
            try:
                # Нормализуем tzinfo для сравнения
                fe_ts = first_evt.timestamp
                cs    = ctx_start
                if fe_ts.tzinfo is None and cs.tzinfo is not None:
                    from datetime import timezone as _tz
                    fe_ts = fe_ts.replace(tzinfo=_tz.utc)
                delta_min = (fe_ts - cs).total_seconds() / 60
                if 0 <= delta_min < 10:
                    proximity_warning = (
                        f"⚠ Первое событие ({fe_ts.strftime('%H:%M')}) "
                        f"всего в {delta_min:.0f} мин от начала окна загрузки — "
                        "возможно, root cause возник раньше."
                    )
            except Exception:
                pass

        user = _jdump({
            "окно_загрузки_логов": {
                "start": ctx_start.isoformat() if ctx_start else None,
                "end":   ctx_end.isoformat()   if ctx_end   else None,
            },
            "окно_инцидента": {
                "start": inc_start.isoformat() if inc_start else None,
                "end":   inc_end.isoformat()   if inc_end   else None,
            },
            "зоны_покрытия": merged.zones_covered,
            "первое_событие": (
                {"id": first_evt.id, "time": first_evt.timestamp.isoformat(),
                 "desc": events_index.get(first_evt.id, "")}
                if first_evt else None
            ),
            "последнее_событие": (
                {"id": last_evt.id, "time": last_evt.timestamp.isoformat(),
                 "desc": events_index.get(last_evt.id, "")}
                if last_evt else None
            ),
            "сервисы_в_анализе": seen_services,
            "разрывы_в_данных": [
                {"start": g.start.isoformat(), "end": g.end.isoformat(),
                 "desc": _ru(g.description, getattr(g, "description_ru", None))}
                for g in merged.gaps
            ],
            "гипотезы": [
                {"title": _ru(h.title, getattr(h, "title_ru", None)),
                 "confidence": h.confidence,
                 "description": _ru(h.description, getattr(h, "description_ru", None))}
                for h in merged.hypotheses
            ],
            "предупреждение": proximity_warning or None,
        })

        system = _LANG + (
            "Ты генерируешь раздел «Рекомендации по расширению анализа» отчёта об инциденте.\n\n"
            "Ответь на два вопроса — конкретно, с обоснованием:\n\n"
            "### 1. Временное окно\n"
            "Нужно ли расширить окно загрузки логов? В какую сторону и на сколько?\n"
            "Сигналы что надо расширить НАЗАД:\n"
            "- первое событие анализа близко к началу окна (< 10 мин)\n"
            "- в гипотезах упоминаются изменения/деплои «до инцидента» без конкретных событий\n"
            "- root cause цепочка оборвана: начинается с середины без предшествующего триггера\n"
            "- зона context_before не покрыта или покрыта мало\n"
            "Сигналы что надо расширить ВПЕРЁД:\n"
            "- последнее событие близко к концу окна\n"
            "- нет события «восстановление» / «сервис вернулся в норму»\n"
            "Если окно достаточное — прямо скажи: «Окно достаточное, расширять не нужно».\n\n"
            "### 2. Набор сервисов\n"
            "Нужно ли добавить логи других сервисов?\n"
            "Смотри: какие сервисы упоминаются в гипотезах и описаниях событий, "
            "но ОТСУТСТВУЮТ в списке сервисов анализа?\n"
            "Для каждого предложенного сервиса: почему его логи важны и что они могут объяснить.\n"
            "Если набор сервисов достаточный — скажи об этом явно.\n\n"
            "Формат: два подраздела с конкретными выводами. "
            "«Рассмотреть возможность» — плохо. «Расширить окно на 2 часа назад» — хорошо."
        )

        return await self._call_or_stub("coverage_recommendations", system, user)

    @staticmethod
    def _sec_events_reference(merged: MergedAnalysis) -> str:
        """Программный справочник событий: полная таблица ID → детали.

        Генерируется без LLM. Позволяет по ID из любой секции найти
        время, сервис и описание события.
        """
        events = sorted(merged.events, key=lambda e: e.timestamp)
        if not events:
            return ""

        rows = [
            "| ID | Время (UTC) | Сервис | Описание | Severity | Важность |",
            "|-------|-------------|--------|---------|----------|----------|",
        ]
        for e in events:
            desc = _ru(e.description, getattr(e, "description_ru", None))
            imp_flag = "⚠" if e.importance >= 0.8 else ""
            rows.append(
                f"| `{e.id}` | `{e.timestamp.strftime('%H:%M:%S')}` "
                f"| `{e.source}` | {desc} | {e.severity.value} | {imp_flag} {e.importance:.1f} |"
            )
        return "\n".join(rows)

    async def _call(self, name: str, system: str, user: str) -> str:
        """LLM-вызов; пробрасывает ContextOverflowError для обработки выше."""
        import time as _time

        if self._progress:
            next_n = self._progress.done + 1
            logger.info(
                "  [секция %d/%d] ▶  %s",
                next_n, self._progress.total, name,
            )

        t_sec = _time.monotonic()
        result = await self.llm.call_text(
            system=system,
            user=user,
            temperature=self.config.temperature_report,
        )
        elapsed_sec = _time.monotonic() - t_sec

        path = self._save(f"multipass_{name}.md", result)

        if self._progress:
            status = self._progress.tick()
            logger.info(
                "  %s  [%s]  →  %s",
                status, fmt_dur(elapsed_sec),
                path if path else f"multipass_{name}.md",
            )

        return result

    async def _call_or_stub(self, name: str, system: str, user: str) -> str:
        """LLM-вызов с fallback на заглушку при любой ошибке."""
        try:
            return await self._call(name, system, user)
        except Exception as exc:
            if self._progress:
                self._progress.tick()  # считаем как выполненный шаг
            logger.error("  Section '%s' failed: %s — using placeholder", name, exc)
            return _PLACEHOLDER

    def _save(self, filename: str, content: str) -> Optional[Path]:
        if self._run_dir is None:
            return None
        self._run_dir.mkdir(parents=True, exist_ok=True)
        path = self._run_dir / filename
        path.write_text(content, encoding="utf-8")
        return path
