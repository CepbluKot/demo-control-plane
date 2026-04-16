"""PipelineOrchestrator — точка входа пайплайна.

Последовательность:
  1. DataLoader.iter_log_pages → страницы логов
  2. Chunker.chunk → list[Chunk] по токеновому бюджету
  3. DataLoader.fetch_metrics → list[MetricRow] (опционально)
  4. MapProcessor.process_all → list[BatchAnalysis]  (параллельно)
  5. TreeReducer.reduce → MergedAnalysis
  6. ReportGenerator.generate → str (Markdown)

Артефакты каждого прогона сохраняются в {runs_dir}/{timestamp}/:
  llm/          — промпты и ответы LLM (call_NNNN_*.txt)
  map/          — BatchAnalysis по каждому чанку
  reduce/       — MergedAnalysis после каждого merge-шага
  chunks_meta.json   — метаданные чанков (без сырых строк)
  report.md          — финальный отчёт
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from log_summarizer.chunker import Chunker
from log_summarizer.config import PipelineConfig
from log_summarizer.data_loader import DataLoader
from log_summarizer.llm_client import LLMClient
from log_summarizer.map_processor import MapProcessor
from log_summarizer.markdown_renderer import MarkdownRenderer
from log_summarizer.models import BatchAnalysis, Chunk, MergedAnalysis, MetricRow
from log_summarizer.report_generator import ReportGenerator
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import get_logger

logger = get_logger("orchestrator")


class PipelineOrchestrator:
    """Запуск полного пайплайна анализа инцидента.

    Args:
        clickhouse_client: Готовый клиент clickhouse_connect.
        config: Конфигурация пайплайна.
    """

    def __init__(self, clickhouse_client: Any, config: PipelineConfig) -> None:
        self.config = config

        # Создаём папку для артефактов этого прогона
        self._run_dir: Optional[Path] = self._make_run_dir(config.runs_dir)

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
            audit_dir=self._run_dir / "llm" if self._run_dir else None,
        )
        self._data_loader = DataLoader(clickhouse_client, config)
        self._map_processor = MapProcessor(
            self._llm, self._chunker, config,
            run_dir=self._run_dir / "map" if self._run_dir else None,
        )
        self._tree_reducer = TreeReducer(
            self._llm, config,
            run_dir=self._run_dir / "reduce" if self._run_dir else None,
        )
        self._report_generator = ReportGenerator(
            self._llm, config,
            run_dir=self._run_dir,
        )

    # ── Публичный API ─────────────────────────────────────────────────

    async def run(self) -> str:
        """Запускает пайплайн и возвращает Markdown-отчёт."""
        t0 = time.monotonic()

        # Валидация временных окон
        errors = self.config.validate_windows()
        for err in errors:
            logger.warning("Window validation: %s", err)

        if self.config.has_context_window():
            logger.info(
                "Pipeline starting: model=%s  context=%s → %s  incident=%s → %s",
                self.config.model,
                self.config.context_start_actual(),
                self.config.context_end_actual(),
                self.config.incident_start,
                self.config.incident_end,
            )
        else:
            logger.info(
                "Pipeline starting: model=%s, incident=%s → %s",
                self.config.model,
                self.config.incident_start,
                self.config.incident_end,
            )
        if self._run_dir:
            logger.info("Run artifacts → %s", self._run_dir.resolve())

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
        self._save_chunks_meta(chunks)

        # ── 2. MAP-фаза ───────────────────────────────────────────────
        batch_results: list[BatchAnalysis] = await self._map_processor.process_all(
            chunks, metrics=metrics
        )

        batch_results = [b for b in batch_results if b.events or b.hypotheses or b.narrative.strip()]
        if not batch_results:
            logger.warning("MAP phase produced no useful results")
            return "# Incident Analysis\n\nNo significant events found in the log data."

        self._log_map_summary(batch_results)

        # ── 3. REDUCE-фаза ────────────────────────────────────────────
        merged = await self._tree_reducer.reduce(
            batch_results=batch_results,
            early_summaries=[],
        )

        self._log_merged_summary(merged)

        # ── 4а. Программный Markdown-отчёт (детерминированный) ───────
        self._save_structured_report(merged)

        # ── 4б. LLM-отчёт (нарративный) ──────────────────────────────
        report = await self._report_generator.generate(
            merged=merged,
            early_summaries=[],
        )

        elapsed = time.monotonic() - t0
        logger.info("Pipeline complete in %.1fs", elapsed)
        if self._run_dir:
            logger.info("All artifacts saved in %s", self._run_dir.resolve())
        return report

    def run_sync(self) -> str:
        """Синхронная обёртка для использования без asyncio.run()."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run())
        finally:
            loop.close()

    # ── Вспомогательные ───────────────────────────────────────────────

    @staticmethod
    def _make_run_dir(runs_dir: str) -> Optional[Path]:
        """Создаёт папку runs/{timestamp}/ для артефактов прогона."""
        if not runs_dir:
            return None
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = Path(runs_dir) / ts
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_chunks_meta(self, chunks: list[Chunk]) -> None:
        """Сохраняет метаданные чанков (без сырых строк логов)."""
        if self._run_dir is None:
            return
        meta = [
            {
                "index": i,
                "rows": len(c.rows),
                "token_estimate": c.token_estimate,
                "time_from": c.time_range[0].isoformat(),
                "time_to": c.time_range[1].isoformat(),
            }
            for i, c in enumerate(chunks)
        ]
        path = self._run_dir / "chunks_meta.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Chunks meta saved → %s  (%d chunks)", path, len(chunks))

    # ── Сводные логи ──────────────────────────────────────────────────

    @staticmethod
    def _log_map_summary(batch_results: list[BatchAnalysis]) -> None:
        """Сводная таблица MAP-результатов в лог."""
        total_events = sum(len(b.events) for b in batch_results)
        total_evidence = sum(len(b.evidence) for b in batch_results)
        total_hyp = sum(len(b.hypotheses) for b in batch_results)

        logger.info(
            "MAP summary: %d batches | events=%d | evidence=%d | hypotheses=%d",
            len(batch_results), total_events, total_evidence, total_hyp,
        )
        sep = "─" * 80
        logger.info(sep)
        logger.info(
            "  %-5s  %-17s  %-14s  %-6s  %-8s  %-3s  %s",
            "batch", "period", "zone", "events", "evidence", "hyp", "top severity",
        )
        logger.info(sep)
        for i, b in enumerate(batch_results):
            t0, t1 = b.time_range
            sev: dict[str, int] = {}
            for e in b.events:
                sev[e.severity.value] = sev.get(e.severity.value, 0) + 1
            sev_str = " ".join(f"{k}={v}" for k, v in sorted(sev.items()))
            # zone берём из первого события или из нарратива — если нет, "?"
            zone = getattr(b, "batch_zone", "?") if hasattr(b, "batch_zone") else "?"
            logger.info(
                "  %03d    %s→%s  %-14s  %-6d  %-8d  %-3d  %s",
                i,
                t0.strftime("%H:%M:%S"), t1.strftime("%H:%M:%S"),
                zone,
                len(b.events), len(b.evidence), len(b.hypotheses),
                sev_str or "—",
            )
        logger.info(sep)

    @staticmethod
    def _log_merged_summary(merged: MergedAnalysis) -> None:
        """Сводка финального MergedAnalysis в лог."""
        t0, t1 = merged.time_range
        sev: dict[str, int] = {}
        for e in merged.events:
            sev[e.severity.value] = sev.get(e.severity.value, 0) + 1
        conf: dict[str, int] = {}
        for h in merged.hypotheses:
            conf[h.confidence] = conf.get(h.confidence, 0) + 1

        sep = "═" * 72
        logger.info(sep)
        logger.info("REDUCE result: %s → %s", t0.strftime("%H:%M:%S"), t1.strftime("%H:%M:%S"))
        logger.info(
            "  events=%d (%s) | causal=%d | hypotheses=%d (%s) | gaps=%d",
            len(merged.events),
            " ".join(f"{k}={v}" for k, v in sorted(sev.items())),
            len(merged.causal_chains),
            len(merged.hypotheses),
            " ".join(f"{k}×{v}" for k, v in sorted(conf.items())),
            len(merged.gaps),
        )
        logger.info(
            "  evidence_bank=%d | alert_refs=%d | recommendations=%d",
            len(merged.evidence_bank),
            len(merged.alert_refs),
            len(merged.preliminary_recommendations),
        )
        if merged.alert_refs:
            for ref in merged.alert_refs:
                logger.info("  alert %s → %s", ref.alert_id, ref.status.value)
        narrative = (merged.narrative_ru or merged.narrative)[:120].replace("\n", " ")
        logger.info("  narrative: %s…", narrative)
        logger.info(sep)

    def _save_structured_report(self, merged: MergedAnalysis) -> None:
        """Сохраняет программный Markdown-отчёт из MergedAnalysis (без LLM)."""
        try:
            md = MarkdownRenderer(merged, self.config).render()
        except Exception as exc:
            logger.warning("MarkdownRenderer failed: %s — skipping report_data.md", exc)
            return

        if self._run_dir:
            path = self._run_dir / "report_data.md"
            path.write_text(md, encoding="utf-8")
            logger.info("Structured report → %s", path.resolve())
        else:
            logger.debug("report_data.md not saved (no run_dir configured)")

    def _load_data(self) -> tuple[list[Chunk], Optional[list[MetricRow]]]:
        """Загрузка логов и метрик (синхронная, вызывается из executor)."""
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

        metrics = self._data_loader.fetch_metrics(
            start=self.config.incident_start,
            end=self.config.incident_end,
        )
        return all_chunks, metrics
