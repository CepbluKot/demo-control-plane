"""TreeReducer — итеративная REDUCE-фаза (MAP-результаты → один MergedAnalysis).

Алгоритм:
  1. Группируем результаты по adaptive_group_size (сколько влезает в 55% контекста).
  2. Каждую группу отправляем в LLM → MergedAnalysis.
  3. После merge: программно обрезаем events до max_events_per_merge (без LLM).
  4. Если результат после merge > compression_target_pct — сжимаем через LLM.
  5. Повторяем до тех пор, пока не останется один элемент.
  6. evidence_bank никогда не проходит через LLM — только concat + dedup.
"""

from __future__ import annotations

import json
from typing import Union

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import ContextOverflowError, LLMClient
from log_summarizer.models import (
    BatchAnalysis,
    Evidence,
    MergedAnalysis,
    Severity,
)
from log_summarizer.prompts.reduce_compress import REDUCE_COMPRESS_SYSTEM_TEMPLATE
from log_summarizer.prompts.reduce_merge import REDUCE_MERGE_SYSTEM_TEMPLATE
from log_summarizer.utils.logging import get_logger
from log_summarizer.utils.tokens import estimate_tokens, tokens_to_chars

logger = get_logger("tree_reducer")

# Union type для элементов очереди reduce
_Item = Union[BatchAnalysis, MergedAnalysis]


class TreeReducer:
    """Итеративная свёртка списка BatchAnalysis → один MergedAnalysis.

    Args:
        llm: LLM-клиент.
        config: Конфигурация пайплайна.
    """

    def __init__(self, llm: LLMClient, config: PipelineConfig) -> None:
        self.llm = llm
        self.config = config

    # ── Публичный API ─────────────────────────────────────────────────

    async def reduce(
        self,
        batch_results: list[BatchAnalysis],
        early_summaries: list[str],
    ) -> MergedAnalysis:
        """Свёртка списка BatchAnalysis в один MergedAnalysis.

        Args:
            batch_results: Выходы MAP-фазы.
            early_summaries: Список коротких текстовых саммари (накапливаются
                             если очередь не сходится достаточно быстро).

        Returns:
            Финальный MergedAnalysis с заполненным evidence_bank.
        """
        if not batch_results:
            raise ValueError("Cannot reduce empty list of batch results")

        # Collect all evidence upfront — never pass through LLM
        evidence_bank = self._collect_evidence(batch_results)

        items: list[_Item] = list(batch_results)

        for round_num in range(1, self.config.max_reduce_rounds + 1):
            if len(items) == 1:
                break

            logger.info(
                "Reduce round %d: %d items remaining", round_num, len(items)
            )

            group_size = self._adaptive_group_size(items)
            groups = self._make_groups(items, group_size)

            next_items: list[_Item] = []
            for group in groups:
                if len(group) == 1:
                    next_items.append(group[0])
                    continue
                merged = await self._merge_group(group, round_num)
                merged = self._trim_events(merged)
                merged = await self._maybe_compress(merged)
                next_items.append(merged)

            items = next_items
        else:
            logger.warning(
                "Reduce did not converge in %d rounds — forcing final merge",
                self.config.max_reduce_rounds,
            )
            if len(items) > 1:
                items = [await self._merge_group(items, self.config.max_reduce_rounds + 1)]

        result = items[0]

        # Если результат — BatchAnalysis (edge case: один батч), конвертируем
        if isinstance(result, BatchAnalysis):
            result = self._batch_to_merged(result)

        # Присоединяем evidence_bank — никогда не проходит через LLM
        evidence_bank = self._dedup_evidence(
            evidence_bank + list(result.evidence_bank)
        )
        result = result.model_copy(update={"evidence_bank": evidence_bank})

        logger.info(
            "Reduce complete: %d events, %d evidence, %d hypotheses",
            len(result.events),
            len(result.evidence_bank),
            len(result.hypotheses),
        )
        return result

    # ── Группировка ───────────────────────────────────────────────────

    def _adaptive_group_size(self, items: list[_Item]) -> int:
        """Вычисляет group_size: сколько items влезает в 55% контекста.

        Берём среднее по первым min(5, len) элементам, чтобы не переполнить
        контекст при merge.
        """
        sample = items[: min(5, len(items))]
        avg_tokens = sum(estimate_tokens(self._item_to_json(i)) for i in sample) / len(
            sample
        )
        budget = int(self.config.max_context_tokens * 0.55)
        adaptive = max(2, int(budget // avg_tokens))
        # Ограничиваем сверху max_group_size из конфига
        size = min(adaptive, self.config.max_group_size)
        logger.debug(
            "adaptive_group_size: avg_tokens=%.0f, budget=%d, computed=%d, used=%d",
            avg_tokens,
            budget,
            adaptive,
            size,
        )
        return size

    @staticmethod
    def _make_groups(items: list[_Item], group_size: int) -> list[list[_Item]]:
        """Нарезаем список на группы по group_size."""
        return [items[i : i + group_size] for i in range(0, len(items), group_size)]

    # ── Merge ─────────────────────────────────────────────────────────

    async def _merge_group(
        self,
        group: list[_Item],
        round_num: int,
    ) -> MergedAnalysis:
        """Отправляем группу в LLM, получаем MergedAnalysis.

        Если ContextOverflowError — делаем попарный merge рекурсивно.
        """
        system = self._build_merge_system()
        user = self._build_merge_user(group)

        try:
            result: MergedAnalysis = await self.llm.call_json(
                system=system,
                user=user,
                response_model=MergedAnalysis,
                temperature=self.config.temperature_reduce,
            )
            logger.debug(
                "Round %d merge: %d items → %d events",
                round_num,
                len(group),
                len(result.events),
            )
            return result
        except ContextOverflowError:
            logger.warning(
                "ContextOverflow in merge group of %d — splitting pair-wise",
                len(group),
            )
            if len(group) <= 2:
                # Fallback: программный merge без LLM
                return self._programmatic_merge(group)
            # Рекурсивно сворачиваем попарно
            mid = len(group) // 2
            left = await self._merge_group(group[:mid], round_num)
            right = await self._merge_group(group[mid:], round_num)
            return await self._merge_group([left, right], round_num)

    # ── Events trimming ───────────────────────────────────────────────

    def _trim_events(self, analysis: MergedAnalysis) -> MergedAnalysis:
        """Программно обрезает список events до max_events_per_merge.

        Не через LLM. Сортируем по severity (critical → info), берём топ-N.
        IDs в causal_chains / hypotheses могут стать "висячими" — это допустимо,
        LLM на следующем раунде их проигнорирует.
        """
        limit = self.config.max_events_per_merge
        if len(analysis.events) <= limit:
            return analysis

        sorted_events = sorted(
            analysis.events,
            key=lambda e: Severity.priority(e.severity),
        )
        trimmed = sorted_events[:limit]
        logger.debug(
            "Trimmed events %d → %d (max_events_per_merge=%d)",
            len(analysis.events),
            len(trimmed),
            limit,
        )
        return analysis.model_copy(update={"events": trimmed})

    # ── Compression ───────────────────────────────────────────────────

    async def _maybe_compress(self, analysis: MergedAnalysis) -> MergedAnalysis:
        """Сжимает MergedAnalysis если он превышает compression_target_pct контекста."""
        target_chars = tokens_to_chars(
            int(self.config.max_context_tokens * self.config.compression_target_pct / 100)
        )
        json_size = len(analysis.to_json_str())
        if json_size <= target_chars:
            return analysis

        logger.info(
            "Analysis too large (%d chars > %d target) — compressing",
            json_size,
            target_chars,
        )
        return await self._compress(analysis)

    async def _compress(self, analysis: MergedAnalysis) -> MergedAnalysis:
        """LLM-сжатие текстов в MergedAnalysis (события, нарратив, гипотезы)."""
        user = (
            "Compress the following MergedAnalysis JSON:\n\n"
            "```json\n"
            + analysis.to_json_str()
            + "\n```"
        )
        try:
            compressed: MergedAnalysis = await self.llm.call_json(
                system=REDUCE_COMPRESS_SYSTEM_TEMPLATE,
                user=user,
                response_model=MergedAnalysis,
                temperature=self.config.temperature_reduce,
            )
            # evidence_bank не трогаем — восстанавливаем из оригинала
            return compressed.model_copy(
                update={"evidence_bank": analysis.evidence_bank}
            )
        except ContextOverflowError:
            logger.error("ContextOverflow during compression — returning uncompressed")
            return analysis

    # ── Evidence bank ─────────────────────────────────────────────────

    @staticmethod
    def _collect_evidence(batch_results: list[BatchAnalysis]) -> list[Evidence]:
        """Собираем все Evidence из MAP-результатов."""
        all_evidence: list[Evidence] = []
        for batch in batch_results:
            all_evidence.extend(batch.evidence)
        return all_evidence

    @staticmethod
    def _dedup_evidence(items: list[Evidence]) -> list[Evidence]:
        """Дедупликация Evidence по raw_line."""
        seen: set[str] = set()
        result: list[Evidence] = []
        for ev in items:
            if ev.raw_line not in seen:
                seen.add(ev.raw_line)
                result.append(ev)
        return result

    # ── Промпты ───────────────────────────────────────────────────────

    def _build_merge_system(self) -> str:
        start = (
            self.config.incident_start.isoformat()
            if self.config.incident_start
            else "unknown"
        )
        end = (
            self.config.incident_end.isoformat()
            if self.config.incident_end
            else "unknown"
        )
        return REDUCE_MERGE_SYSTEM_TEMPLATE.format(
            incident_context=self.config.incident_context or "No additional context provided.",
            incident_start=start,
            incident_end=end,
        )

    @staticmethod
    def _build_merge_user(group: list[_Item]) -> str:
        """Форматируем user-промпт: список JSON-объектов для merge."""
        parts = [
            f"Merge the following {len(group)} analyses into one MergedAnalysis.\n"
        ]
        for i, item in enumerate(group, 1):
            item_json = TreeReducer._item_to_json(item)
            parts.append(f"### Analysis {i}")
            parts.append("```json")
            parts.append(item_json)
            parts.append("```")
        return "\n".join(parts)

    # ── Вспомогательные ───────────────────────────────────────────────

    @staticmethod
    def _item_to_json(item: _Item) -> str:
        """Сериализуем item БЕЗ evidence_bank для оценки размера / промптов."""
        if isinstance(item, MergedAnalysis):
            return item.to_json_str()
        return item.to_json_str()

    @staticmethod
    def _batch_to_merged(batch: BatchAnalysis) -> MergedAnalysis:
        """Конвертируем одиночный BatchAnalysis в MergedAnalysis."""
        return MergedAnalysis(
            time_range=batch.time_range,
            narrative=batch.narrative,
            events=batch.events,
            causal_chains=[],
            hypotheses=batch.hypotheses,
            anomalies=batch.anomalies,
            gaps=[],
            impact_summary=batch.metrics_context or "",
            evidence_bank=batch.evidence,
        )

    @staticmethod
    def _programmatic_merge(group: list[_Item]) -> MergedAnalysis:
        """Программный merge без LLM — fallback при ContextOverflow на паре."""
        all_ts = []
        all_events = {}
        all_hyp = {}
        all_anomalies = {}
        all_causal = []
        all_gaps = []
        narratives = []
        impacts = []

        for item in group:
            if isinstance(item, BatchAnalysis):
                all_ts.extend([item.time_range[0], item.time_range[1]])
                for e in item.events:
                    all_events.setdefault(e.id, e)
                for h in item.hypotheses:
                    all_hyp.setdefault(h.id, h)
                for a in item.anomalies:
                    all_anomalies.setdefault(a.description, a)
                if item.narrative:
                    narratives.append(item.narrative)
            else:
                all_ts.extend([item.time_range[0], item.time_range[1]])
                for e in item.events:
                    all_events.setdefault(e.id, e)
                for h in item.hypotheses:
                    all_hyp.setdefault(h.id, h)
                for a in item.anomalies:
                    all_anomalies.setdefault(a.description, a)
                all_causal.extend(item.causal_chains)
                all_gaps.extend(item.gaps)
                if item.narrative:
                    narratives.append(item.narrative)
                if item.impact_summary:
                    impacts.append(item.impact_summary)

        return MergedAnalysis(
            time_range=(min(all_ts), max(all_ts)),
            narrative=" ".join(narratives),
            events=list(all_events.values()),
            causal_chains=all_causal,
            hypotheses=list(all_hyp.values()),
            anomalies=list(all_anomalies.values()),
            gaps=all_gaps,
            impact_summary="; ".join(impacts),
        )
