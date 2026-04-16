"""CrossIncidentAnalyzer — объединённый анализ нескольких инцидентов.

Публичный API:
  generate_combined_report(incidents_with_config) → Markdown в той же
      14-секционной структуре, что и per-incident отчёты.
      Сохраняет {runs_dir}/combined_report.md.

  generate(incidents) → вспомогательный кросс-инцидентный текст
      (мета-цепочка + резюме). Используется как контекст в combined_report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from log_summarizer.llm_client import LLMClient
from log_summarizer.models import MergedAnalysis
from log_summarizer.utils.logging import get_logger
from log_summarizer.utils.progress import fmt_dur

logger = get_logger("cross_incident")

_LANG = (
    "Пиши на русском языке. "
    "Технические термины (имена сервисов, pod, namespace, Kubernetes-объекты, "
    "OOM, SIGTERM, коды ошибок, имена метрик, CLI-команды) оставляй как есть.\n\n"
)


def _ru(eng: Optional[str], ru: Optional[str]) -> str:
    return (ru or eng or "").strip()


def _jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _conf_float(c: str) -> float:
    return {"high": 1.0, "medium": 0.5, "low": 0.2}.get(str(c), 0.5)


class CrossIncidentAnalyzer:
    """Анализирует связи между несколькими инцидентами из списка INCIDENTS.

    Args:
        llm: LLM-клиент (тот же, что использовался для отдельных инцидентов).
        runs_dir: Корневая папка runs/ — отчёт сохраняется туда напрямую.
        temperature: Температура для LLM-вызовов.
    """

    def __init__(
        self,
        llm: LLMClient,
        runs_dir: str,
        temperature: float = 0.3,
    ) -> None:
        self.llm = llm
        self._runs_dir = Path(runs_dir) if runs_dir else None
        self._temperature = temperature

    # ── Публичный API ─────────────────────────────────────────────────

    async def generate(
        self,
        incidents: list[tuple[str, MergedAnalysis]],
    ) -> str:
        """Генерирует кросс-инцидентный анализ.

        Args:
            incidents: Список (name, MergedAnalysis). Порядок не важен —
                       сортируем внутри по времени начала.

        Returns:
            Markdown-документ.
        """
        if not incidents:
            return "# Кросс-инцидентный анализ\n\nНет данных для анализа."

        if len(incidents) == 1:
            return (
                "# Кросс-инцидентный анализ\n\n"
                "Только один инцидент — кросс-инцидентный анализ неприменим.\n"
                "Для анализа связей добавьте минимум два инцидента в INCIDENTS."
            )

        # Сортируем по времени начала (по первому событию)
        sorted_incidents = sorted(
            incidents,
            key=lambda pair: pair[1].time_range[0] if pair[1].time_range else datetime.min.replace(tzinfo=timezone.utc),
        )

        import time as _time
        _SEP = "─" * 68
        t_cross = _time.monotonic()

        logger.info("")
        logger.info(_SEP)
        logger.info("КРОСС-ИНЦИДЕНТНЫЙ АНАЛИЗ  ▶  %d инцидентов", len(sorted_incidents))
        for i, (name, merged) in enumerate(sorted_incidents, 1):
            t0m, t1m = merged.time_range
            logger.info(
                "  %d. %s  |  %s → %s  |  %d событий  %d гипотез",
                i, name,
                t0m.strftime("%Y-%m-%d %H:%M %Z"),
                t1m.strftime("%H:%M %Z"),
                len(merged.events), len(merged.hypotheses),
            )
        logger.info(_SEP)

        # Шаг 1: резюме каждого инцидента с root cause
        logger.info("  [шаг 1/2] ▶  Резюме каждого инцидента (root cause + механизм)...")
        t1 = _time.monotonic()
        per_incident_text = await self._summarize_each(sorted_incidents)
        logger.info("  [шаг 1/2] ✓  Резюме готовы  [%s]", fmt_dur(_time.monotonic() - t1))

        # Шаг 2: мета-цепочка между инцидентами
        logger.info("  [шаг 2/2] ▶  Построение мета-цепочки между инцидентами...")
        t2 = _time.monotonic()
        meta_chain_text = await self._build_meta_chain(sorted_incidents, per_incident_text)
        logger.info("  [шаг 2/2] ✓  Мета-цепочка готова  [%s]", fmt_dur(_time.monotonic() - t2))

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        parts = [
            "# Кросс-инцидентный анализ\n",
            f"*{len(sorted_incidents)} инцидентов | {now}*\n",
            "---\n",
            "## Инциденты\n",
            per_incident_text,
            "\n---\n",
            "## Связи между инцидентами\n",
            meta_chain_text,
            "\n---\n",
            f"*Сгенерировано автоматически по {len(sorted_incidents)} инцидентам*",
        ]
        report = "\n".join(p for p in parts if p.strip())

        self._save(report)
        logger.info(
            "КРОСС-ИНЦИДЕНТНЫЙ АНАЛИЗ  ✓  Общее время: %s",
            fmt_dur(_time.monotonic() - t_cross),
        )
        logger.info(_SEP)
        return report

    # ── Внутренние методы ─────────────────────────────────────────────

    async def _summarize_each(
        self, incidents: list[tuple[str, MergedAnalysis]]
    ) -> str:
        """Шаг 1: краткая карточка для каждого инцидента."""
        payloads = []
        for name, merged in incidents:
            t0, t1 = merged.time_range

            leading_hyp = max(
                merged.hypotheses,
                key=lambda h: _conf_float(h.confidence),
                default=None,
            )

            top_chain = None
            if merged.causal_chains:
                top_chain = max(
                    merged.causal_chains,
                    key=lambda c: _conf_float(c.confidence),
                )

            recs = (merged.preliminary_recommendations_ru or merged.preliminary_recommendations)[:3]

            payloads.append({
                "name": name,
                "time_range": {
                    "start": t0.isoformat(),
                    "end":   t1.isoformat(),
                },
                "events_count": len(merged.events),
                "narrative": _ru(merged.narrative, merged.narrative_ru),
                "leading_hypothesis": {
                    "title":       _ru(leading_hyp.title, getattr(leading_hyp, "title_ru", None)),
                    "confidence":  leading_hyp.confidence,
                    "description": _ru(leading_hyp.description, getattr(leading_hyp, "description_ru", None)),
                } if leading_hyp else None,
                "top_causal_chain": {
                    "from_event":  top_chain.from_event_id,
                    "to_event":    top_chain.to_event_id,
                    "description": _ru(top_chain.description, getattr(top_chain, "description_ru", None)),
                    "mechanism":   top_chain.mechanism,
                    "confidence":  top_chain.confidence,
                } if top_chain else None,
                "top_recommendations": recs,
            })

        system = _LANG + (
            "Ты генерируешь раздел «Инциденты» кросс-инцидентного анализа.\n"
            "Для каждого инцидента напиши карточку:\n\n"
            "### {name} ({start} → {end})\n"
            "- **Описание:** 2–3 предложения что произошло\n"
            "- **Первопричина:** наиболее вероятная гипотеза с уровнем уверенности\n"
            "- **Ключевой механизм:** как конкретно одно событие привело к деградации\n"
            "- **Топ-рекомендации:** 1–3 конкретных действия\n\n"
            "Используй именно такой формат заголовка. Будь конкретным."
        )
        user = _jdump(payloads)

        try:
            result = await self.llm.call_text(
                system=system, user=user, temperature=self._temperature
            )
            return result
        except Exception as exc:
            logger.error("    Шаг 1 завершился с ошибкой: %s — программный fallback", exc)
            return self._fallback_summaries(payloads)

    async def _build_meta_chain(
        self,
        incidents: list[tuple[str, MergedAnalysis]],
        per_incident_text: str,
    ) -> str:
        """Шаг 2: ищет причинно-следственные связи между инцидентами."""
        brief = []
        for name, merged in incidents:
            t0, t1 = merged.time_range

            leading_hyp = max(
                merged.hypotheses,
                key=lambda h: _conf_float(h.confidence),
                default=None,
            )

            brief.append({
                "name":          name,
                "start":         t0.isoformat(),
                "end":           t1.isoformat(),
                "key_services":  sorted({e.source for e in merged.events if e.source})[:10],
                "root_cause":    _ru(
                    getattr(leading_hyp, "title", None),
                    getattr(leading_hyp, "title_ru", None),
                ) if leading_hyp else "неизвестно",
                "narrative_short": _ru(merged.narrative, merged.narrative_ru)[:300],
            })

        system = _LANG + (
            "Ты строишь мета-цепочку между несколькими инцидентами.\n\n"
            "Задача: найти РЕАЛЬНЫЕ причинно-следственные связи.\n"
            "Критерии связи:\n"
            "- Один инцидент создал условия для следующего (истощение ресурса, накопленная нагрузка)\n"
            "- Одна первопричина проявляется в разных формах в разное время\n"
            "- Инцидент является побочным эффектом восстановления после предыдущего\n\n"
            "Если инциденты независимы — явно укажи: "
            "«Связей между инцидентами не обнаружено» и объясни почему.\n"
            "Не придумывай связи там, где их нет.\n\n"
            "### Структура ответа\n\n"
            "**Если связи есть:**\n"
            "Пронумерованная цепочка от первого инцидента к последнему.\n"
            "Для каждой связи: механизм + уровень уверенности (высокий/средний/низкий).\n\n"
            "**Если связей нет:**\n"
            "2–3 предложения почему инциденты независимы.\n\n"
            "**В любом случае — раздел «Системные паттерны»:**\n"
            "Что общего у всех инцидентов (общие сервисы, типы ошибок, временные паттерны), "
            "даже если прямых причинно-следственных связей нет."
        )
        user = (
            f"Инциденты (хронологически):\n{_jdump(brief)}\n\n"
            f"Резюме каждого инцидента:\n{per_incident_text[:4000]}"
        )

        try:
            result = await self.llm.call_text(
                system=system, user=user, temperature=self._temperature
            )
            return result
        except Exception as exc:
            logger.error("    Шаг 2 завершился с ошибкой: %s", exc)
            return "[Мета-цепочка не сгенерирована — произошла ошибка при обращении к LLM]"

    # ── Комбинированный мультипасс-отчёт ─────────────────────────────

    async def generate_combined_report(
        self,
        incidents: list[tuple[str, MergedAnalysis, object]],  # (name, merged, config)
    ) -> str:
        """Генерирует комбинированный отчёт по всем инцидентам.

        Структура идентична per-incident мультипасс-отчёту (14 секций).
        Сохраняет runs/combined_report.md.

        Args:
            incidents: Список (name, MergedAnalysis, PipelineConfig).
        """
        from log_summarizer.config import PipelineConfig
        from log_summarizer.multipass_report_generator import MultipassReportGenerator
        from log_summarizer.tree_reducer import TreeReducer

        import time as _time
        _SEP = "─" * 68

        if not incidents:
            return "# Комбинированный отчёт\n\nНет данных."

        if len(incidents) == 1:
            # Один инцидент — просто возвращаем его отчёт
            name, merged, cfg = incidents[0]
            logger.info("Только один инцидент — combined report совпадает с per-incident.")
            generator = MultipassReportGenerator(
                llm=self.llm,
                config=cfg,  # type: ignore[arg-type]
                run_dir=self._runs_dir,
            )
            return await generator.generate(merged, degradation={})

        # Сортируем по времени начала
        sorted_incidents = sorted(
            incidents,
            key=lambda t: t[1].time_range[0],
        )

        t_combined = _time.monotonic()
        logger.info("")
        logger.info(_SEP)
        logger.info(
            "КОМБИНИРОВАННЫЙ ОТЧЁТ  ▶  объединяем %d инцидентов",
            len(sorted_incidents),
        )

        # Шаг 1: краткий кросс-инцидентный текст для контекста
        logger.info("  [шаг 1] Кросс-инцидентный анализ (контекст для LLM)...")
        t1 = _time.monotonic()
        pairs = [(name, merged) for name, merged, _ in sorted_incidents]
        cross_text = await self.generate(pairs)
        logger.info("  [шаг 1] ✓  [%s]", fmt_dur(_time.monotonic() - t1))

        # Шаг 2: программный merge всех MergedAnalysis
        logger.info("  [шаг 2] Программное объединение MergedAnalysis...")
        all_merged = [merged for _, merged, _ in sorted_incidents]
        combined_merged = TreeReducer._programmatic_merge(all_merged)
        logger.info(
            "  [шаг 2] ✓  %d событий  ·  %d гипотез  ·  %d цепочек",
            len(combined_merged.events),
            len(combined_merged.hypotheses),
            len(combined_merged.causal_chains),
        )

        # Шаг 3: создаём combined PipelineConfig
        combined_config = self._make_combined_config(sorted_incidents, cross_text)

        # Шаг 4: генерируем мультипасс-отчёт
        logger.info("  [шаг 3] Генерация комбинированного мультипасс-отчёта...")
        combined_run_dir = self._runs_dir / "_combined" if self._runs_dir else None
        if combined_run_dir:
            combined_run_dir.mkdir(parents=True, exist_ok=True)

        generator = MultipassReportGenerator(
            llm=self.llm,
            config=combined_config,
            run_dir=combined_run_dir,
        )
        report = await generator.generate(combined_merged, degradation={})

        # Сохраняем
        if self._runs_dir:
            path = self._runs_dir / "combined_report.md"
            path.write_text(report, encoding="utf-8")
            logger.info("  Комбинированный отчёт → %s", path.resolve())

        logger.info(
            "КОМБИНИРОВАННЫЙ ОТЧЁТ  ✓  Общее время: %s",
            fmt_dur(_time.monotonic() - t_combined),
        )
        logger.info(_SEP)
        return report

    def _make_combined_config(
        self,
        incidents: list[tuple[str, MergedAnalysis, object]],
        cross_text: str = "",
    ) -> object:
        """Создаёт PipelineConfig, охватывающий все инциденты."""
        from log_summarizer.config import PipelineConfig

        names = [name for name, _, _ in incidents]
        first_cfg = incidents[0][2]  # type: PipelineConfig

        # Временные границы — union всех инцидентов
        inc_starts = [cfg.incident_start for _, _, cfg in incidents if cfg.incident_start]  # type: ignore[attr-defined]
        inc_ends   = [cfg.incident_end   for _, _, cfg in incidents if cfg.incident_end]    # type: ignore[attr-defined]
        ctx_starts = [s for _, _, cfg in incidents if (s := cfg.context_start_actual())]    # type: ignore[attr-defined]
        ctx_ends   = [e for _, _, cfg in incidents if (e := cfg.context_end_actual())]      # type: ignore[attr-defined]

        inc_start = min(inc_starts) if inc_starts else None
        inc_end   = max(inc_ends)   if inc_ends   else None
        ctx_start = min(ctx_starts) if ctx_starts else None
        ctx_end   = max(ctx_ends)   if ctx_ends   else None

        # Сводный контекст
        names_str = ", ".join(names)
        combined_context = (
            f"Комбинированный анализ {len(incidents)} инцидентов: {names_str}\n\n"
        )
        for name, _, cfg in incidents:  # type: ignore[misc]
            ctx = cfg.incident_context.strip()[:300].replace("\n", " ")  # type: ignore[attr-defined]
            combined_context += f"• [{name}]: {ctx}\n"

        if cross_text.strip():
            combined_context += f"\n---\nКросс-инцидентный анализ:\n{cross_text[:1500]}"

        # Алерты — union без дублей по id
        all_alerts = []
        seen_ids: set[str] = set()
        for _, _, cfg in incidents:  # type: ignore[misc]
            for alert in cfg.alerts:  # type: ignore[attr-defined]
                if alert.id not in seen_ids:
                    seen_ids.add(alert.id)
                    all_alerts.append(alert)

        # Суммарные строки логов
        total_rows = sum(getattr(cfg, "total_log_rows", 0) for _, _, cfg in incidents)  # type: ignore[misc]

        return PipelineConfig(
            incident_context=combined_context,
            incident_start=inc_start,
            incident_end=inc_end,
            context_start=ctx_start,
            context_end=ctx_end,
            context_auto_expand_hours=0,  # окна уже заданы явно
            alerts=all_alerts,
            model=first_cfg.model,              # type: ignore[attr-defined]
            api_base=first_cfg.api_base,        # type: ignore[attr-defined]
            api_key=first_cfg.api_key,          # type: ignore[attr-defined]
            max_context_tokens=first_cfg.max_context_tokens,  # type: ignore[attr-defined]
            model_supports_tool_calling=first_cfg.model_supports_tool_calling,  # type: ignore[attr-defined]
            temperature_report=first_cfg.temperature_report,  # type: ignore[attr-defined]
            total_log_rows=total_rows,
            runs_dir="",  # артефакты пишем сами
        )

    # ── Вспомогательные ───────────────────────────────────────────────

    @staticmethod
    def _fallback_summaries(payloads: list[dict]) -> str:
        """Программный fallback если LLM недоступен."""
        lines = []
        for p in payloads:
            t0 = p["time_range"]["start"][:16]
            t1 = p["time_range"]["end"][:16]
            lines.append(f"### {p['name']} ({t0} → {t1})")
            lines.append(p["narrative"] or "Нарратив не сгенерирован.")
            if p.get("leading_hypothesis"):
                hyp = p["leading_hypothesis"]
                lines.append(
                    f"\n**Первопричина:** {hyp['title']} "
                    f"(уверенность: {hyp['confidence']})"
                )
            if p.get("top_recommendations"):
                lines.append("\n**Рекомендации:**")
                for r in p["top_recommendations"]:
                    lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)

    def _save(self, content: str) -> None:
        if not self._runs_dir:
            return
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        path = self._runs_dir / "cross_incident_report.md"
        path.write_text(content, encoding="utf-8")
        logger.info("  Отчёт сохранён → %s", path.resolve())
