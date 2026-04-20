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
            timeout=config.llm_timeout,
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

    async def run_from_map_dir(self, map_dir: Path) -> str:
        """Продолжает пайплайн с той стадии, где он остановился.

        Каждый шаг проверяет наличие своего выходного файла и пропускает его
        если файл уже есть. Повторные resume используют результаты предыдущих.
        Чтобы перегенерировать конкретный шаг — удали его файл:
          reduce/final_merged.json → перезапустит REDUCE
          report_data.md          → перезапустит программный отчёт
          report_multipass.md     → перезапустит многопроходный LLM-отчёт
          report.md               → перезапустит монолитный LLM-отчёт

        Args:
            map_dir: Папка с chunk_NNN.json (обычно <artifact_dir>/map/).
                     reduce/ ищется рядом: map_dir.parent / "reduce".
        """
        from log_summarizer.models import BatchAnalysis, MergedAnalysis

        artifact_dir = map_dir.parent
        reduce_dir = artifact_dir / "reduce"
        final_merged_path = reduce_dir / "final_merged.json"

        degradation: dict = {"processing_error_batches": 0, "programmatic_merges": 0}

        # ── СТАДИЯ 3: REDUCE ─────────────────────────────────────────────
        if final_merged_path.exists():
            logger.info("RESUME: reduce/final_merged.json найден — пропускаем REDUCE")
            merged = MergedAnalysis.model_validate_json(
                final_merged_path.read_text(encoding="utf-8")
            )
            self.last_merged = merged
            self._log_merged_summary(merged)
        else:
            from log_summarizer.tree_reducer import TreeReducer as _TR
            checkpoint = _TR.load_latest_checkpoint(reduce_dir) if reduce_dir.exists() else None

            logger.info("")
            t3 = time.monotonic()

            if checkpoint is not None:
                round_completed, ck_items, ck_evidence, ck_alert_refs = checkpoint
                logger.info(
                    "СТАДИЯ 3  ▶  REDUCE с checkpoint раунда %d  (%d items)",
                    round_completed, len(ck_items),
                )
                merged = await self._tree_reducer.reduce_from_checkpoint(
                    items=ck_items,
                    start_round=round_completed + 1,
                    evidence_bank=ck_evidence,
                    alert_refs=ck_alert_refs,
                )
            else:
                chunk_files = sorted(map_dir.glob("chunk_*.json"))
                if not chunk_files:
                    raise FileNotFoundError(f"Нет ни MAP-чанков, ни REDUCE-checkpoint в {map_dir.parent}")

                batch_results: list[BatchAnalysis] = [
                    BatchAnalysis.model_validate_json(f.read_text(encoding="utf-8"))
                    for f in chunk_files
                ]
                logger.info("RESUME: загружено %d MAP-чанков из %s", len(batch_results), map_dir)

                degradation["processing_error_batches"] = sum(
                    1 for b in batch_results if b.data_quality == "processing_error"
                )
                batch_results = [b for b in batch_results if b.events or b.hypotheses or b.narrative.strip()]
                if not batch_results:
                    return "# Incident Analysis\n\nNo significant events found in the log data."

                logger.info("СТАДИЯ 3  ▶  REDUCE с нуля  (%d батчей)", len(batch_results))
                merged = await self._tree_reducer.reduce(batch_results=batch_results, early_summaries=[])

            self.last_merged = merged
            degradation["programmatic_merges"] = self._tree_reducer.programmatic_merge_count
            logger.info("СТАДИЯ 3  ✓  REDUCE завершён  [%.1fс]", time.monotonic() - t3)
            self._log_merged_summary(merged)

        # ── СТАДИЯ 4: Программный отчёт ──────────────────────────────────
        report_data_path = artifact_dir / "report_data.md"
        if report_data_path.exists():
            logger.info("RESUME: report_data.md найден — пропускаем")
        else:
            logger.info("")
            logger.info("СТАДИЯ 4  ▶  Программный отчёт")
            self._save_structured_report(merged)

        # ── СТАДИЯ 5: Многопроходный LLM-отчёт ──────────────────────────
        multipass_path = artifact_dir / "report_multipass.md"
        if multipass_path.exists():
            logger.info("RESUME: report_multipass.md найден — пропускаем")
        else:
            logger.info("")
            logger.info("СТАДИЯ 5  ▶  Многопроходный LLM-отчёт")
            await self._multipass_generator.generate(merged, degradation=degradation)

        # ── СТАДИЯ 6: Монолитный LLM-отчёт ──────────────────────────────
        report_path = artifact_dir / "report.md"
        if report_path.exists():
            logger.info("RESUME: report.md найден — пропускаем")
            report = report_path.read_text(encoding="utf-8")
        else:
            logger.info("")
            logger.info("СТАДИЯ 6  ▶  Монолитный LLM-отчёт")
            report = await self._report_generator.generate(merged=merged, early_summaries=[])

        logger.info("RESUME завершён ✓")
        return report

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
        all_rows: list = []
        page_count = 0

        raw_dir = self._run_dir / "raw" if self._run_dir else None
        if raw_dir:
            raw_dir.mkdir(exist_ok=True)

        for page in self._data_loader.iter_log_pages(page_size=self.config.batch_size):
            if not page:
                continue
            page_count += 1
            all_rows.extend(page)
            logger.debug("Loaded page %d: %d rows (total so far: %d)", page_count, len(page), len(all_rows))

            if raw_dir:
                path = raw_dir / f"page_{page_count:03d}.txt"
                with path.open("w", encoding="utf-8") as f:
                    for row in page:
                        f.write(row.raw_line)
                        if not row.raw_line.endswith("\n"):
                            f.write("\n")

        # Чанкуем все строки разом — каждый чанк заполняется до токенового бюджета.
        # Если чанковать по страницам, чанки получаются маленькими (~4k токенов
        # вместо ~55k бюджета) и MAP делает в 10× больше вызовов чем нужно.
        all_chunks = self._chunker.chunk(all_rows)

        logger.info(
            "Log loading complete: %d страниц · %d строк → %d чанков (бюджет ~%d tok/чанк)%s",
            page_count, len(all_rows), len(all_chunks), self.config.map_batch_tokens(),
            f"  ·  raw/ → {raw_dir}" if raw_dir else "",
        )

        metrics = self._data_loader.fetch_metrics(
            start=self.config.incident_start,
            end=self.config.incident_end,
        )
        return all_chunks, metrics
