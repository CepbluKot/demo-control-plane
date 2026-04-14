"""ReportGenerator — финальный отчёт из MergedAnalysis.

Бюджетирование:
  available = max_context_tokens - response_reserve - system_prompt_tokens
  analysis  = available × report_budget_analysis_pct  (MergedAnalysis JSON)
  evidence  = available × report_budget_evidence_pct  (evidence_bank строки)
  early     = available × report_budget_early_pct     (ранние саммари)

Секционный fallback:
  Если полный user-промпт > available_tokens → три отдельных LLM-вызова:
    1. analysis section  (executive summary + timeline + RCA + impact)
    2. evidence section  (evidence bank)
    3. recommendations section (contributing factors + hypotheses + recommendations)
  Секции конкатенируются в конечный отчёт.
"""

from __future__ import annotations

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import ContextOverflowError, LLMClient
from log_summarizer.models import MergedAnalysis, ReportBudget
from log_summarizer.prompts.report import (
    build_report_system_prompt,
    format_report_user_prompt,
)
from log_summarizer.utils.logging import get_logger
from log_summarizer.utils.tokens import (
    estimate_tokens,
    trim_lines_to_budget,
    trim_rows_to_budget,
    trim_to_budget,
    tokens_to_chars,
)

logger = get_logger("report_generator")


class ReportGenerator:
    """Генерация финального Markdown-отчёта.

    Args:
        llm: LLM-клиент.
        config: Конфигурация пайплайна.
    """

    def __init__(self, llm: LLMClient, config: PipelineConfig) -> None:
        self.llm = llm
        self.config = config

    # ── Публичный API ─────────────────────────────────────────────────

    async def generate(
        self,
        merged: MergedAnalysis,
        early_summaries: list[str],
    ) -> str:
        """Генерирует финальный отчёт.

        Пробует сначала полный режим (один LLM-вызов).
        При ContextOverflowError или если user-промпт не влезает
        переходит к секционному режиму (3 вызова).

        Args:
            merged: Финальный MergedAnalysis из TreeReducer.
            early_summaries: Текстовые саммари из ранних раундов reduce.

        Returns:
            Готовый Markdown-отчёт.
        """
        budget = self._calculate_budget(merged, early_summaries)

        analysis_json = self._trim_analysis(merged, budget.analysis_tokens)
        evidence_text = self._trim_evidence(merged, budget.evidence_tokens)
        early_text = self._trim_early(early_summaries, budget.early_tokens)

        user = format_report_user_prompt(
            analysis_json=analysis_json,
            evidence_text=evidence_text,
            early_summaries_text=early_text,
            incident_context=self.config.incident_context or "",
            incident_start=(
                self.config.incident_start.isoformat()
                if self.config.incident_start
                else "unknown"
            ),
            incident_end=(
                self.config.incident_end.isoformat()
                if self.config.incident_end
                else "unknown"
            ),
        )

        user_tokens = estimate_tokens(user)
        system_tokens = self.config.report_system_prompt_tokens
        total_tokens = user_tokens + system_tokens + self.config.report_response_reserve_tokens

        if total_tokens > self.config.max_context_tokens:
            logger.warning(
                "Full report prompt too large (%d tokens) — using sectional mode",
                total_tokens,
            )
            return await self._generate_sectional(
                analysis_json=analysis_json,
                evidence_text=evidence_text,
                early_text=early_text,
            )

        try:
            return await self.llm.call_text(
                system=build_report_system_prompt(),
                user=user,
                temperature=self.config.temperature_report,
            )
        except ContextOverflowError:
            logger.warning("ContextOverflow on full report — falling back to sectional mode")
            return await self._generate_sectional(
                analysis_json=analysis_json,
                evidence_text=evidence_text,
                early_text=early_text,
            )

    # ── Бюджетирование ────────────────────────────────────────────────

    def _calculate_budget(
        self,
        merged: MergedAnalysis,
        early_summaries: list[str],
    ) -> ReportBudget:
        """Рассчитывает токеновые бюджеты для каждого компонента."""
        available = (
            self.config.max_context_tokens
            - self.config.report_response_reserve_tokens
            - self.config.report_system_prompt_tokens
        )
        available = max(available, 1000)  # safety floor

        analysis_tokens = int(available * self.config.report_budget_analysis_pct)
        evidence_tokens = int(available * self.config.report_budget_evidence_pct)
        early_tokens = int(available * self.config.report_budget_early_pct)

        logger.debug(
            "Report budget: available=%d, analysis=%d, evidence=%d, early=%d",
            available,
            analysis_tokens,
            evidence_tokens,
            early_tokens,
        )
        return ReportBudget(
            analysis_tokens=analysis_tokens,
            evidence_tokens=evidence_tokens,
            early_tokens=early_tokens,
        )

    # ── Тримминг компонентов ──────────────────────────────────────────

    @staticmethod
    def _trim_analysis(merged: MergedAnalysis, budget_tokens: int) -> str:
        """Обрезает сериализованный MergedAnalysis до бюджета (по символам)."""
        json_str = merged.to_json_str()
        return trim_to_budget(json_str, budget_tokens)

    @staticmethod
    def _trim_evidence(merged: MergedAnalysis, budget_tokens: int) -> str:
        """Форматирует evidence_bank и обрезает целыми записями.

        Каждая запись — отдельная строка вида:
            [timestamp] [source] raw_line
        Никогда не обрезает запись на полуслове: использует trim_rows_to_budget.
        """
        lines = [
            f"[{ev.timestamp.isoformat()}] [{ev.source}] {ev.raw_line}"
            for ev in merged.evidence_bank
        ]
        kept = trim_rows_to_budget(lines, budget_tokens)
        dropped = len(lines) - len(kept)
        if dropped:
            logger.debug("Evidence trimmed: %d → %d entries (budget)", len(lines), len(kept))
        return "\n".join(kept)

    @staticmethod
    def _trim_early(early_summaries: list[str], budget_tokens: int) -> str:
        """Конкатенирует ранние саммари с обрезкой целыми записями."""
        kept = trim_rows_to_budget(early_summaries, budget_tokens)
        return "\n\n---\n\n".join(kept)

    # ── Секционный режим ──────────────────────────────────────────────

    async def _generate_sectional(
        self,
        analysis_json: str,
        evidence_text: str,
        early_text: str,
    ) -> str:
        """Три отдельных LLM-вызова — один на секцию.

        Разбиваем на: analysis_section, evidence_section, recommendations_section.
        """
        base_ctx = (
            f"Incident: {self.config.incident_context or 'N/A'}\n"
            f"Period: "
            + (self.config.incident_start.isoformat() if self.config.incident_start else "?")
            + " → "
            + (self.config.incident_end.isoformat() if self.config.incident_end else "?")
        )

        # Секция 1: executive summary + timeline + RCA + impact
        user_analysis = (
            f"{base_ctx}\n\n"
            "## Analysis JSON\n```json\n"
            + analysis_json
            + "\n```"
            + (f"\n\n## Early summaries\n{early_text}" if early_text.strip() else "")
        )
        # Секция 2: evidence
        user_evidence = (
            f"{base_ctx}\n\n"
            "## Evidence (verbatim log lines)\n```\n"
            + evidence_text
            + "\n```"
        )
        # Секция 3: contributing factors + hypotheses + recommendations
        user_recommendations = (
            f"{base_ctx}\n\n"
            "## Analysis JSON\n```json\n"
            + analysis_json
            + "\n```"
        )

        logger.info("Generating report in sectional mode (3 LLM calls)")

        section_analysis, section_evidence, section_recommendations = (
            await self._call_section("analysis", user_analysis),
            await self._call_section("evidence", user_evidence),
            await self._call_section("recommendations", user_recommendations),
        )

        return "\n\n".join(
            part for part in [section_analysis, section_evidence, section_recommendations]
            if part.strip()
        )

    async def _call_section(self, section: str, user: str) -> str:
        """Один LLM-вызов для одной секции."""
        try:
            return await self.llm.call_text(
                system=build_report_system_prompt(section=section),
                user=user,
                temperature=self.config.temperature_report,
            )
        except ContextOverflowError:
            logger.error(
                "ContextOverflow even in sectional mode for section=%s — returning empty",
                section,
            )
            return f"<!-- {section} section unavailable: context overflow -->"
