"""PipelineOrchestrator — точка входа пайплайна.

Последовательность:
  1. DataLoader.iter_log_pages → страницы логов
  2. Chunker.chunk → list[Chunk] по токеновому бюджету
  3. DataLoader.fetch_metrics → list[MetricRow] (опционально)
  4. MapProcessor.process_all → list[BatchAnalysis]  (параллельно)
  5. TreeReducer.reduce → MergedAnalysis
  6. ReportGenerator.generate → str (Markdown)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from log_summarizer.chunker import Chunker
from log_summarizer.config import PipelineConfig
from log_summarizer.data_loader import DataLoader
from log_summarizer.llm_client import LLMClient
from log_summarizer.map_processor import MapProcessor
from log_summarizer.models import BatchAnalysis, Chunk, MetricRow
from log_summarizer.report_generator import ReportGenerator
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import get_logger, setup_pipeline_logging

logger = get_logger("orchestrator")


class PipelineOrchestrator:
    """Запуск полного пайплайна анализа инцидента.

    Args:
        clickhouse_client: Готовый клиент clickhouse_connect.
        config: Конфигурация пайплайна.
    """

    def __init__(self, clickhouse_client: Any, config: PipelineConfig) -> None:
        self.config = config

        # Инициализируем компоненты
        max_batch_tokens = int(config.max_context_tokens * 0.55)
        self._chunker = Chunker(
            max_batch_tokens=max_batch_tokens,
            min_batch_lines=config.min_batch_lines,
        )
        self._llm = LLMClient(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model,
            max_retries=config.max_retries,
            retry_backoff_base=config.retry_backoff_base,
            use_instructor=config.use_instructor,
            model_supports_tool_calling=config.model_supports_tool_calling,
        )
        self._data_loader = DataLoader(clickhouse_client, config)
        self._map_processor = MapProcessor(self._llm, self._chunker, config)
        self._tree_reducer = TreeReducer(self._llm, config)
        self._report_generator = ReportGenerator(self._llm, config)

    # ── Публичный API ─────────────────────────────────────────────────

    async def run(self) -> str:
        """Запускает пайплайн и возвращает Markdown-отчёт.

        Returns:
            Финальный отчёт в формате Markdown.
        """
        t0 = time.monotonic()
        logger.info(
            "Pipeline starting: model=%s, incident=%s → %s",
            self.config.model,
            self.config.incident_start,
            self.config.incident_end,
        )

        # ── 1. Загрузка данных ────────────────────────────────────────
        chunks, metrics = await asyncio.get_event_loop().run_in_executor(
            None, self._load_data
        )

        if not chunks:
            logger.warning("No log data found for the specified period")
            return "# Incident Analysis\n\nNo log data found for the specified period."

        logger.info(
            "Data loaded: %d chunks, %d metric rows",
            len(chunks),
            len(metrics) if metrics else 0,
        )

        # ── 2. MAP-фаза ───────────────────────────────────────────────
        batch_results: list[BatchAnalysis] = await self._map_processor.process_all(
            chunks, metrics=metrics
        )

        # Отфильтровываем пустые результаты (ошибки загрузки)
        batch_results = [b for b in batch_results if b.events or b.hypotheses or b.narrative.strip()]
        if not batch_results:
            logger.warning("MAP phase produced no useful results")
            return "# Incident Analysis\n\nNo significant events found in the log data."

        # ── 3. REDUCE-фаза ────────────────────────────────────────────
        merged = await self._tree_reducer.reduce(
            batch_results=batch_results,
            early_summaries=[],
        )

        # ── 4. Финальный отчёт ────────────────────────────────────────
        report = await self._report_generator.generate(
            merged=merged,
            early_summaries=[],
        )

        elapsed = time.monotonic() - t0
        logger.info("Pipeline complete in %.1fs", elapsed)
        return report

    def run_sync(self) -> str:
        """Синхронная обёртка для использования без asyncio.run().

        Создаёт новый event loop, запускает пайплайн, закрывает loop.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run())
        finally:
            loop.close()

    # ── Загрузка данных (sync, запускается в executor) ────────────────

    def _load_data(self) -> tuple[list[Chunk], Optional[list[MetricRow]]]:
        """Загрузка логов и метрик (синхронная, вызывается из executor)."""
        # Загружаем все строки лога постранично и сразу чанкуем
        all_chunks: list[Chunk] = []
        total_rows = 0

        for page in self._data_loader.iter_log_pages(page_size=self.config.batch_size):
            if not page:
                continue
            total_rows += len(page)
            page_chunks = self._chunker.chunk(page)
            all_chunks.extend(page_chunks)
            logger.debug("Loaded page: %d rows → %d chunks", len(page), len(page_chunks))

        logger.info(
            "Log loading complete: %d total rows → %d chunks",
            total_rows,
            len(all_chunks),
        )

        # Метрики загружаем одним запросом
        metrics = self._data_loader.fetch_metrics(
            start=self.config.incident_start,
            end=self.config.incident_end,
        )

        return all_chunks, metrics
