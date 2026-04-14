"""MapProcessor — параллельная обработка чанков (MAP-фаза).

Каждый чанк → один LLM-вызов → BatchAnalysis.
При ContextOverflowError чанк делится пополам рекурсивно.
Результаты двух половин сливаются программно (без LLM).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from log_summarizer.chunker import Chunker
from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import ContextOverflowError, LLMClient
from log_summarizer.models import (
    Anomaly,
    BatchAnalysis,
    Chunk,
    Evidence,
    Event,
    Hypothesis,
    MetricRow,
)
from log_summarizer.prompts.map_system import MAP_SYSTEM_TEMPLATE
from log_summarizer.prompts.map_user import format_map_user_prompt
from log_summarizer.utils.logging import get_logger
from log_summarizer.utils.tokens import estimate_tokens

logger = get_logger("map_processor")


class MapProcessor:
    """Параллельная MAP-фаза: чанки → BatchAnalysis.

    Args:
        llm: LLM-клиент.
        chunker: Для split при ContextOverflowError.
        config: Конфигурация пайплайна.
    """

    def __init__(
        self,
        llm: LLMClient,
        chunker: Chunker,
        config: PipelineConfig,
    ) -> None:
        self.llm = llm
        self.chunker = chunker
        self.config = config

    # ── Публичный API ─────────────────────────────────────────────────

    async def process_all(
        self,
        chunks: list[Chunk],
        metrics: Optional[list[MetricRow]] = None,
    ) -> list[BatchAnalysis]:
        """Параллельная обработка всех чанков с ограничением конкурентности.

        Args:
            chunks: Нарезанные батчи логов.
            metrics: Опциональные метрики за весь период.

        Returns:
            BatchAnalysis для каждого чанка в том же порядке.
        """
        semaphore = asyncio.Semaphore(self.config.map_concurrency)

        async def _bounded(chunk: Chunk, idx: int) -> BatchAnalysis:
            async with semaphore:
                return await self.process_chunk(chunk, metrics=metrics, chunk_id=idx)

        tasks = [_bounded(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        logger.info("MAP phase complete: %d chunks processed", len(results))
        return list(results)

    async def process_chunk(
        self,
        chunk: Chunk,
        metrics: Optional[list[MetricRow]] = None,
        chunk_id: int = 0,
        _depth: int = 0,
    ) -> BatchAnalysis:
        """Обрабатывает один чанк. При ContextOverflowError делит пополам.

        Args:
            chunk: Батч логов.
            metrics: Метрики за период.
            chunk_id: Номер чанка (для логирования и генерации ID).
            _depth: Внутренний счётчик рекурсии (защита от бесконечного split).
        """
        if _depth > self.config.max_split_depth:
            logger.error(
                "Max split depth %d reached for chunk %d — returning empty analysis",
                self.config.max_split_depth,
                chunk_id,
            )
            return self._empty_analysis(chunk)

        log_budget = int(self.config.max_context_tokens * 0.55)
        system = self._build_system_prompt()
        user = format_map_user_prompt(
            chunk,
            metrics=metrics,
            log_budget_tokens=log_budget,
        )

        try:
            result: BatchAnalysis = await self.llm.call_json(
                system=system,
                user=user,
                response_model=BatchAnalysis,
                temperature=self.config.temperature_map,
            )
            logger.debug("Chunk %d → %d events, %d evidence", chunk_id, len(result.events), len(result.evidence))
            return result
        except ContextOverflowError:
            if len(chunk.rows) <= self.config.min_batch_lines:
                logger.warning(
                    "Chunk %d overflow but too small to split (%d rows) — returning empty",
                    chunk_id,
                    len(chunk.rows),
                )
                return self._empty_analysis(chunk)

            logger.warning(
                "Chunk %d context overflow (%d rows) — splitting in half (depth=%d)",
                chunk_id,
                len(chunk.rows),
                _depth,
            )
            left_chunk, right_chunk = self.chunker.split_chunk(chunk)
            left, right = await asyncio.gather(
                self.process_chunk(left_chunk, metrics=metrics, chunk_id=chunk_id * 10, _depth=_depth + 1),
                self.process_chunk(right_chunk, metrics=metrics, chunk_id=chunk_id * 10 + 1, _depth=_depth + 1),
            )
            return self._merge_halves(left, right)

    # ── Вспомогательные ───────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
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
        return MAP_SYSTEM_TEMPLATE.format(
            incident_context=self.config.incident_context or "No additional context provided.",
            incident_start=start,
            incident_end=end,
        )

    @staticmethod
    def _merge_halves(left: BatchAnalysis, right: BatchAnalysis) -> BatchAnalysis:
        """Программно сливает два BatchAnalysis в один (без LLM).

        Объединяет события, evidence и гипотезы; дедуплицирует по id.
        """
        # time_range: берём минимум/максимум
        all_ts = [left.time_range[0], left.time_range[1], right.time_range[0], right.time_range[1]]
        merged_range = (min(all_ts), max(all_ts))

        # События: дедуп по id (левый имеет приоритет)
        events: dict[str, Event] = {}
        for ev in (*left.events, *right.events):
            events.setdefault(ev.id, ev)

        # Evidence: дедуп по raw_line
        evidence: dict[str, Evidence] = {}
        for e in (*left.evidence, *right.evidence):
            evidence.setdefault(e.raw_line, e)

        # Гипотезы: дедуп по id
        hypotheses: dict[str, Hypothesis] = {}
        for h in (*left.hypotheses, *right.hypotheses):
            hypotheses.setdefault(h.id, h)

        # Аномалии: дедуп по description
        anomalies: dict[str, Anomaly] = {}
        for a in (*left.anomalies, *right.anomalies):
            anomalies.setdefault(a.description, a)

        # Нарратив: объединяем
        narratives = [n for n in (left.narrative, right.narrative) if n.strip()]
        narrative = " ".join(narratives)

        # Метрики и качество данных
        metrics_parts = [m for m in (left.metrics_context, right.metrics_context) if m]
        quality_parts = [q for q in (left.data_quality, right.data_quality) if q]

        return BatchAnalysis(
            time_range=merged_range,
            narrative=narrative,
            events=list(events.values()),
            evidence=list(evidence.values()),
            hypotheses=list(hypotheses.values()),
            anomalies=list(anomalies.values()),
            metrics_context="; ".join(metrics_parts) if metrics_parts else None,
            data_quality="; ".join(quality_parts) if quality_parts else None,
        )

    @staticmethod
    def _empty_analysis(chunk: Chunk) -> BatchAnalysis:
        """Пустой BatchAnalysis — заглушка при неустранимых ошибках."""
        return BatchAnalysis(
            time_range=chunk.time_range,
            narrative="[Analysis unavailable — chunk too large or LLM error]",
            events=[],
            evidence=[],
            hypotheses=[],
            anomalies=[],
            data_quality="processing_error",
        )
