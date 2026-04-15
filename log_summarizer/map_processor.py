"""MapProcessor — параллельная обработка чанков (MAP-фаза).

Каждый чанк → один LLM-вызов → BatchAnalysis.
При ContextOverflowError чанк делится пополам рекурсивно.
Оба результата возвращаются отдельно — REDUCE-фаза свяжет их через LLM.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from log_summarizer.chunker import Chunker
from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import ContextOverflowError, LLMClient
from log_summarizer.models import (
    BatchAnalysis,
    Chunk,
    MetricRow,
)
from log_summarizer.prompts.map_system import MAP_SYSTEM_TEMPLATE, format_alerts_section
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
        run_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.chunker = chunker
        self.config = config
        self._run_dir = run_dir

    # ── Публичный API ─────────────────────────────────────────────────

    async def process_all(
        self,
        chunks: list[Chunk],
        metrics: Optional[list[MetricRow]] = None,
    ) -> list[BatchAnalysis]:
        """Параллельная обработка всех чанков с ограничением конкурентности.

        При split чанка возвращаются два отдельных BatchAnalysis — их смержит
        REDUCE-фаза через LLM, а не программно здесь.

        Args:
            chunks: Нарезанные батчи логов.
            metrics: Опциональные метрики за весь период.

        Returns:
            Плоский список BatchAnalysis (может быть больше len(chunks) при split).
        """
        semaphore = asyncio.Semaphore(self.config.map_concurrency)

        async def _bounded(chunk: Chunk, idx: int) -> list[BatchAnalysis]:
            async with semaphore:
                return await self.process_chunk(chunk, metrics=metrics, chunk_id=idx)

        tasks = [_bounded(chunk, i) for i, chunk in enumerate(chunks)]
        nested = await asyncio.gather(*tasks)
        results = [item for sublist in nested for item in sublist]
        logger.info("MAP phase complete: %d chunks → %d results", len(chunks), len(results))
        return results

    async def process_chunk(
        self,
        chunk: Chunk,
        metrics: Optional[list[MetricRow]] = None,
        chunk_id: int = 0,
        _depth: int = 0,
    ) -> list[BatchAnalysis]:
        """Обрабатывает один чанк. При ContextOverflowError делит пополам.

        Возвращает список: обычно из одного элемента, при split — два отдельных
        BatchAnalysis, которые REDUCE-фаза потом свяжет через LLM.

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
            return [self._empty_analysis(chunk)]

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
            self._save_chunk_result(chunk_id, result)
            return [result]
        except ContextOverflowError:
            if len(chunk.rows) <= self.config.min_batch_lines:
                logger.warning(
                    "Chunk %d overflow but too small to split (%d rows) — returning empty",
                    chunk_id,
                    len(chunk.rows),
                )
                return [self._empty_analysis(chunk)]

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
            # Оба результата идут в REDUCE как отдельные элементы — LLM их смержит
            return left + right

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
            alerts_section=format_alerts_section(self.config.alerts),
        )

    def _save_chunk_result(self, chunk_id: int, result: BatchAnalysis) -> None:
        if self._run_dir is None:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)
        path = self._run_dir / f"chunk_{chunk_id:03d}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("MAP chunk %d saved → %s", chunk_id, path)

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
