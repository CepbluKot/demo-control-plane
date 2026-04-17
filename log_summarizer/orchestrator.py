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
from log_summarizer.multipass_report_generator import MultipassReportGenerator
from log_summarizer.report_generator import ReportGenerator
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import get_logger
from log_summarizer.utils.progress import fmt_dur

logger = get_logger("orchestrator")


class PipelineOrchestrator:
    """Запуск полного пайплайна анализа инцидента.

    Args:
        clickhouse_client: Готовый клиент clickhouse_connect.
        config: Конфигурация пайплайна.
    """

    def __init__(self, clickhouse_client: Any, config: PipelineConfig) -> None:
        self.config = config

        # Финальный MergedAnalysis после REDUCE-фазы; доступен после run().
        self.last_merged: Optional[MergedAnalysis] = None

        # Создаём папку для артефактов этого прогона
        self._run_dir: Optional[Path] = self._make_run_dir(config.runs_dir)

        # Публичный псевдоним — доступен снаружи без обращения к приватному атрибуту
        self.run_dir: Optional[Path] = self._run_dir

        self._chunker = Chunker(
            max_batch_tokens=config.map_batch_tokens(),
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
        self._multipass_generator = MultipassReportGenerator(
            self._llm, config,
            run_dir=self._run_dir,
        )

    # ── Публичный API ─────────────────────────────────────────────────

    async def run(self) -> str:
        """Запускает пайплайн и возвращает Markdown-отчёт."""
        _SEP  = "═" * 72
        _SEP2 = "─" * 72
        t_pipeline = time.monotonic()
        _TOTAL_STAGES = 6

        # ── Шапка пайплайна ───────────────────────────────────────────
        errors = self.config.validate_windows()
        for err in errors:
            logger.warning("Window validation: %s", err)

        cfg = self.config
        ctx_start = cfg.context_start_actual()
        ctx_end   = cfg.context_end_actual()

        logger.info(_SEP)
        logger.info("ПАЙПЛАЙН ИНЦИДЕНТНОЙ СУММАРИЗАЦИИ")
        logger.info("  Модель    : %s", cfg.model)
        logger.info("  API       : %s", cfg.api_base)
        if cfg.incident_start and cfg.incident_end:
            logger.info(
                "  Инцидент  : %s  →  %s",
                cfg.incident_start.strftime("%Y-%m-%d %H:%M %Z"),
                cfg.incident_end.strftime("%H:%M %Z"),
            )
        if cfg.has_context_window() and ctx_start and ctx_end:
            logger.info(
                "  Окно логов: %s  →  %s",
                ctx_start.strftime("%Y-%m-%d %H:%M %Z"),
                ctx_end.strftime("%H:%M %Z"),
            )
        if cfg.incident_context:
            logger.info("  Контекст  : %s", cfg.incident_context.strip()[:120].replace("\n", " "))
        if self._run_dir:
            logger.info("  Артефакты : %s", self._run_dir.resolve())
        logger.info(_SEP)

        # ── СТАДИЯ 1: Загрузка данных ─────────────────────────────────
        logger.info("")
        logger.info("СТАДИЯ 1/%d  ▶  Загрузка данных из ClickHouse", _TOTAL_STAGES)
        if ctx_start and ctx_end:
            logger.info(
                "  Запрашиваем: %s  →  %s",
                ctx_start.strftime("%Y-%m-%d %H:%M %Z"),
                ctx_end.strftime("%H:%M %Z"),
            )
        t1 = time.monotonic()
        chunks, metrics = await asyncio.get_event_loop().run_in_executor(
            None, self._load_data
        )
        t1_elapsed = time.monotonic() - t1

        if not chunks:
            logger.warning("No log data found for the specified period")
            return "# Incident Analysis\n\nNo log data found for the specified period."

        total_rows = sum(len(c.rows) for c in chunks)
        self.config.total_log_rows = total_rows  # доступно снаружи через orchestrator.config
        logger.info(
            "СТАДИЯ 1/%d  ✓  Данные загружены  [%s]  →  %d строк  ·  %d чанков%s",
            _TOTAL_STAGES, fmt_dur(t1_elapsed), total_rows, len(chunks),
            f"  ·  {len(metrics)} метрик" if metrics else "",
        )
        self._save_chunks_meta(chunks)

        # ── СТАДИЯ 2: MAP-фаза ────────────────────────────────────────
        logger.info("")
        logger.info(
            "СТАДИЯ 2/%d  ▶  MAP-фаза  (%d чанков  ·  параллельность %d)",
            _TOTAL_STAGES, len(chunks), cfg.map_concurrency,
        )
        if chunks:
            t0_map = chunks[0].time_range[0]
            t1_map = chunks[-1].time_range[1]
            logger.info(
                "  Период MAP: %s  →  %s",
                t0_map.strftime("%Y-%m-%d %H:%M %Z"),
                t1_map.strftime("%H:%M %Z"),
            )
        t2 = time.monotonic()
        batch_results: list[BatchAnalysis] = await self._map_processor.process_all(
            chunks, metrics=metrics
        )
        t2_elapsed = time.monotonic() - t2

        processing_error_batches = sum(
            1 for b in batch_results if b.data_quality == "processing_error"
        )
        batch_results = [b for b in batch_results if b.events or b.hypotheses or b.narrative.strip()]
        if not batch_results:
            logger.warning("MAP phase produced no useful results")
            return "# Incident Analysis\n\nNo significant events found in the log data."

        total_events_map = sum(len(b.events) for b in batch_results)
        total_hyp_map    = sum(len(b.hypotheses) for b in batch_results)
        logger.info(
            "СТАДИЯ 2/%d  ✓  MAP завершён  [%s]  →  %d батчей  ·  %d событий  ·  %d гипотез%s",
            _TOTAL_STAGES, fmt_dur(t2_elapsed), len(batch_results),
            total_events_map, total_hyp_map,
            f"  ·  ⚠ ошибок обработки: {processing_error_batches}" if processing_error_batches else "",
        )
        self._log_map_summary(batch_results)

        # ── СТАДИЯ 3: REDUCE-фаза ─────────────────────────────────────
        logger.info("")
        logger.info(
            "СТАДИЯ 3/%d  ▶  REDUCE-фаза  (%d батчей → 1 MergedAnalysis)",
            _TOTAL_STAGES, len(batch_results),
        )
        t3 = time.monotonic()
        merged = await self._tree_reducer.reduce(
            batch_results=batch_results,
            early_summaries=[],
        )
        self.last_merged = merged
        t3_elapsed = time.monotonic() - t3

        logger.info(
            "СТАДИЯ 3/%d  ✓  REDUCE завершён  [%s]  →  %d событий  ·  %d цепочек  ·  %d гипотез",
            _TOTAL_STAGES, fmt_dur(t3_elapsed),
            len(merged.events), len(merged.causal_chains), len(merged.hypotheses),
        )
        self._log_merged_summary(merged)

        degradation = {
            "processing_error_batches": processing_error_batches,
            "programmatic_merges": self._tree_reducer.programmatic_merge_count,
        }
        if any(degradation.values()):
            logger.warning("Degradation stats: %s", degradation)

        # ── СТАДИЯ 4: Программный отчёт ───────────────────────────────
        logger.info("")
        logger.info("СТАДИЯ 4/%d  ▶  Программный Markdown-отчёт (детерминированный, без LLM)", _TOTAL_STAGES)
        t4 = time.monotonic()
        self._save_structured_report(merged)
        t4_elapsed = time.monotonic() - t4
        logger.info("СТАДИЯ 4/%d  ✓  report_data.md  [%s]", _TOTAL_STAGES, fmt_dur(t4_elapsed))

        # ── СТАДИЯ 5: Многопроходный LLM-отчёт ───────────────────────
        logger.info("")
        logger.info("СТАДИЯ 5/%d  ▶  Многопроходный LLM-отчёт (14 секций последовательно)", _TOTAL_STAGES)
        t5 = time.monotonic()
        await self._multipass_generator.generate(merged, degradation=degradation)
        t5_elapsed = time.monotonic() - t5
        logger.info("СТАДИЯ 5/%d  ✓  report_multipass.md  [%s]", _TOTAL_STAGES, fmt_dur(t5_elapsed))

        # ── СТАДИЯ 6: Монолитный LLM-отчёт ───────────────────────────
        logger.info("")
        logger.info("СТАДИЯ 6/%d  ▶  Монолитный LLM-отчёт (нарративный, один вызов)", _TOTAL_STAGES)
        t6 = time.monotonic()
        report = await self._report_generator.generate(
            merged=merged,
            early_summaries=[],
        )
        t6_elapsed = time.monotonic() - t6
        logger.info("СТАДИЯ 6/%d  ✓  report.md  [%s]", _TOTAL_STAGES, fmt_dur(t6_elapsed))

        # ── Итог ──────────────────────────────────────────────────────
        total_elapsed = time.monotonic() - t_pipeline
        logger.info("")
        logger.info(_SEP)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЁН  ✓  Общее время: %s", fmt_dur(total_elapsed))
        logger.info(
            "  Загрузка: %s  ·  MAP: %s  ·  REDUCE: %s  ·  Отчёты: %s",
            fmt_dur(t1_elapsed), fmt_dur(t2_elapsed), fmt_dur(t3_elapsed),
            fmt_dur(t4_elapsed + t5_elapsed + t6_elapsed),
        )
        if self._run_dir:
            logger.info("  Все артефакты: %s", self._run_dir.resolve())
        logger.info(_SEP)
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
        from log_summarizer.config import MSK
        ts = datetime.now(MSK).strftime("%Y-%m-%dT%H-%M-%S")
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
        page_count = 0

        raw_dir = self._run_dir / "raw" if self._run_dir else None
        if raw_dir:
            raw_dir.mkdir(exist_ok=True)

        for page in self._data_loader.iter_log_pages(page_size=self.config.batch_size):
            if not page:
                continue
            page_count += 1
            total_rows += len(page)
            page_chunks = self._chunker.chunk(page)
            all_chunks.extend(page_chunks)
            logger.debug("Loaded page %d: %d rows → %d chunks", page_count, len(page), len(page_chunks))

            if raw_dir:
                path = raw_dir / f"page_{page_count:03d}.txt"
                with path.open("w", encoding="utf-8") as f:
                    for row in page:
                        f.write(row.raw_line)
                        if not row.raw_line.endswith("\n"):
                            f.write("\n")

        logger.info(
            "Log loading complete: %d страниц · %d строк → %d чанков%s",
            page_count, total_rows, len(all_chunks),
            f"  ·  raw/ → {raw_dir}" if raw_dir else "",
        )

        metrics = self._data_loader.fetch_metrics(
            start=self.config.incident_start,
            end=self.config.incident_end,
        )
        return all_chunks, metrics
