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
from pathlib import Path
from typing import Optional, Union

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import ContextOverflowError, LLMClient
from log_summarizer.models import (
    AlertRef,
    AlertStatus,
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

    def __init__(
        self,
        llm: LLMClient,
        config: PipelineConfig,
        run_dir: Optional[Path] = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self._run_dir = run_dir
        self.programmatic_merge_count: int = 0  # сколько раз сработал fallback без LLM

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

        # Collect evidence and alert_refs upfront — never pass through LLM
        evidence_bank = self._collect_evidence(batch_results)
        alert_refs = self._merge_alert_refs(batch_results)

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
            for g_idx, group in enumerate(groups):
                if len(group) == 1:
                    next_items.append(group[0])
                    continue
                merged = await self._merge_group(group, round_num, g_idx)
                merged = self._trim_events(merged)
                merged = await self._maybe_compress(merged)
                self._save_reduce_result(round_num, g_idx, merged)
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
        self._save_reduce_result(0, 0, result, name="final_merged")  # перед evidence

        # Если результат — BatchAnalysis (edge case: один батч), конвертируем
        if isinstance(result, BatchAnalysis):
            result = self._batch_to_merged(result)

        # Присоединяем evidence_bank и alert_refs — никогда не проходят через LLM
        evidence_bank = self._dedup_evidence(
            evidence_bank + list(result.evidence_bank)
        )
        result = result.model_copy(update={
            "evidence_bank": evidence_bank,
            "alert_refs": alert_refs,
        })

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

    def _save_reduce_result(
        self,
        round_num: int,
        group_idx: int,
        result: MergedAnalysis,
        name: Optional[str] = None,
    ) -> None:
        if self._run_dir is None:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)
        fname = name or f"round_{round_num:02d}_group_{group_idx:02d}"
        path = self._run_dir / f"{fname}.json"
        path.write_text(result.to_json_str(), encoding="utf-8")
        logger.info("REDUCE %s saved → %s", fname, path)

    def _save_merge_report(
        self,
        round_num: int,
        group_idx: int,
        group: list[_Item],
        result: MergedAnalysis,
    ) -> None:
        """Сохраняет читаемый отчёт: входы → результат одного merge-шага."""
        if self._run_dir is None:
            return
        self._run_dir.mkdir(parents=True, exist_ok=True)

        sep  = "═" * 68
        sep2 = "─" * 68
        lines: list[str] = [
            sep,
            f"MERGE  round={round_num}  group={group_idx}  ({len(group)} inputs → 1 result)",
            sep,
        ]

        for i, item in enumerate(group, 1):
            t0 = item.time_range[0].strftime("%H:%M:%S")
            t1 = item.time_range[1].strftime("%H:%M:%S")
            events     = item.events
            hypotheses = item.hypotheses
            narrative  = item.narrative

            sev_counts: dict[str, int] = {}
            for e in events:
                sev_counts[e.severity.value] = sev_counts.get(e.severity.value, 0) + 1
            sev_str = "  ".join(f"{k}={v}" for k, v in sorted(sev_counts.items()))

            conf_counts: dict[str, int] = {}
            for h in hypotheses:
                conf_counts[h.confidence] = conf_counts.get(h.confidence, 0) + 1
            conf_str = "  ".join(f"{k}×{v}" for k, v in sorted(conf_counts.items()))

            kind = "BatchAnalysis" if isinstance(item, BatchAnalysis) else "MergedAnalysis"
            causal_str = ""
            if isinstance(item, MergedAnalysis) and item.causal_chains:
                causal_str = f"  causal={len(item.causal_chains)}"

            # Сохраняем входной JSON
            in_path = self._run_dir / f"round_{round_num:02d}_group_{group_idx:02d}_in_{i}.json"
            in_path.write_text(item.to_json_str(), encoding="utf-8")

            lines += [
                "",
                f"── INPUT {i}  [{t0} → {t1}]  {kind}  {sep2[len(f'── INPUT {i}  [{t0} → {t1}]  {kind}  '):]}" ,
                f"   events     : {len(events)}   {sev_str}",
                f"   hypotheses : {len(hypotheses)}   {conf_str}{causal_str}",
                f"   narrative  : {narrative[:200].strip()}",
                f"   → {in_path}",
            ]

        # Результат
        t0r = result.time_range[0].strftime("%H:%M:%S")
        t1r = result.time_range[1].strftime("%H:%M:%S")
        sev_r: dict[str, int] = {}
        for e in result.events:
            sev_r[e.severity.value] = sev_r.get(e.severity.value, 0) + 1
        conf_r: dict[str, int] = {}
        for h in result.hypotheses:
            conf_r[h.confidence] = conf_r.get(h.confidence, 0) + 1

        result_path = self._run_dir / f"round_{round_num:02d}_group_{group_idx:02d}.json"
        lines += [
            "",
            f"── RESULT   [{t0r} → {t1r}]  MergedAnalysis  {sep2[len(f'── RESULT   [{t0r} → {t1r}]  MergedAnalysis  '):]}" ,
            f"   events     : {len(result.events)}   {'  '.join(f'{k}={v}' for k, v in sorted(sev_r.items()))}",
            f"   causal     : {len(result.causal_chains)}",
            f"   hypotheses : {len(result.hypotheses)}   {'  '.join(f'{k}×{v}' for k, v in sorted(conf_r.items()))}",
            f"   gaps       : {len(result.gaps)}",
            f"   narrative  : {result.narrative[:200].strip()}",
            f"   → {result_path}",
            "",
            sep,
        ]

        report_path = self._run_dir / f"round_{round_num:02d}_group_{group_idx:02d}_merge.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Merge report → %s", report_path)

    async def _merge_group(
        self,
        group: list[_Item],
        round_num: int,
        group_idx: int = 0,
    ) -> MergedAnalysis:
        """Отправляем группу в LLM, получаем MergedAnalysis.

        Если payload превышает pre_compress_threshold — превентивно сжимаем входы.
        Если ContextOverflowError — делаем попарный merge рекурсивно.
        """
        system = self._build_merge_system()

        # Превентивная компрессия: если суммарный payload "опасного" размера —
        # сжимаем входы ДО LLM-вызова, не дожидаясь 400.
        threshold = self.config.pre_compress_threshold
        if threshold > 0:
            payload_size = sum(len(self._item_to_json(i)) for i in group)
            if payload_size > threshold:
                logger.info(
                    "Pre-compressing: payload=%d chars > threshold=%d, items=%d",
                    payload_size, threshold, len(group),
                )
                group = await self._compress_group_inputs(group)

        user = self._build_merge_user(group)

        try:
            result: MergedAnalysis = await self.llm.call_json(
                system=system,
                user=user,
                response_model=MergedAnalysis,
                temperature=self.config.temperature_reduce,
            )
            # zones_covered: union of all input zones — set programmatically, not by LLM
            result = result.model_copy(update={"zones_covered": self._merge_zones(group)})
            logger.debug(
                "Round %d merge: %d items → %d events, zones=%s",
                round_num,
                len(group),
                len(result.events),
                result.zones_covered,
            )
            self._save_merge_report(round_num, group_idx, group, result)
            return result
        except ContextOverflowError:
            if len(group) <= 2:
                # Сжимаем по одному элементу и пробуем merge после каждого.
                # Программный fallback только если даже после сжатия всех не влезло.
                logger.warning(
                    "ContextOverflow on pair — trying compression before programmatic fallback"
                )
                return await self._compress_and_merge(group)
            logger.warning(
                "ContextOverflow in merge group of %d — splitting pair-wise",
                len(group),
            )
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

    async def _compress_group_inputs(self, group: list[_Item]) -> list[MergedAnalysis]:
        """Последовательно сжимает все элементы группы до LLM-вызова merge.

        Вызывается только при превышении pre_compress_threshold — превентивная мера
        перед отправкой в LLM, не замена реактивному _compress_and_merge при 400.
        """
        items = [
            item if isinstance(item, MergedAnalysis) else self._batch_to_merged(item)
            for item in group
        ]
        compressed: list[MergedAnalysis] = []
        for i, item in enumerate(items):
            before = len(item.to_json_str())
            c = await self._compress(item)
            after = len(c.to_json_str())
            logger.debug(
                "Pre-compress item %d/%d: %d → %d chars (%.0f%%)",
                i + 1, len(items), before, after, 100 * after / before if before else 0,
            )
            compressed.append(c)
        return compressed

    async def _compress_and_merge(self, group: list[_Item]) -> MergedAnalysis:
        """Сжимаем элементы по одному и пробуем merge после каждого сжатия.

        Алгоритм:
          1. Конвертируем всё в MergedAnalysis (BatchAnalysis → via _batch_to_merged).
          2. Сжимаем items[0], пробуем merge.
          3. Если overflow — сжимаем items[1], пробуем ещё раз.
          4. Если и после сжатия обоих overflow — программный fallback.
        """
        items: list[MergedAnalysis] = [
            item if isinstance(item, MergedAnalysis) else self._batch_to_merged(item)
            for item in group
        ]

        for i in range(len(items)):
            items[i] = await self._compress(items[i])
            try:
                result: MergedAnalysis = await self.llm.call_json(
                    system=self._build_merge_system(),
                    user=self._build_merge_user(items),
                    response_model=MergedAnalysis,
                    temperature=self.config.temperature_reduce,
                )
                logger.debug("Merge succeeded after compressing %d/%d items", i + 1, len(items))
                return result
            except ContextOverflowError:
                if i < len(items) - 1:
                    logger.warning(
                        "Still overflow after compressing item %d/%d — compressing next",
                        i + 1, len(items),
                    )

        logger.error("ContextOverflow even after compressing all items — falling back to programmatic merge")
        self.programmatic_merge_count += 1
        return self._programmatic_merge(items)

    # ── Evidence bank ─────────────────────────────────────────────────

    @staticmethod
    def _collect_evidence(batch_results: list[BatchAnalysis]) -> list[Evidence]:
        """Собираем все Evidence из MAP-результатов."""
        all_evidence: list[Evidence] = []
        for batch in batch_results:
            all_evidence.extend(batch.evidence)
        return all_evidence

    @staticmethod
    def _merge_alert_refs(batch_results: list[BatchAnalysis]) -> list[AlertRef]:
        """Программный merge alert_refs из всех MAP-батчей.

        Для каждого alert_id берём лучший статус (EXPLAINED > PARTIAL >
        NOT_EXPLAINED > NOT_SEEN) и комментарий от него же.
        """
        best: dict[str, AlertRef] = {}
        for batch in batch_results:
            for ref in batch.alert_refs:
                existing = best.get(ref.alert_id)
                if existing is None or (
                    AlertStatus.priority(ref.status)
                    < AlertStatus.priority(existing.status)
                ):
                    best[ref.alert_id] = ref
        return list(best.values())

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
        inc_start = self.config.incident_start_iso() or "unknown"
        inc_end = self.config.incident_end_iso() or "unknown"

        if self.config.has_context_window():
            ctx_start = self.config.context_start_iso() or inc_start
            ctx_end = self.config.context_end_iso() or inc_end
            context_note = (
                f"Context window (logs loaded): {ctx_start} → {ctx_end}\n"
                "Events may come from context_before / incident / context_after zones.\n"
            )
        else:
            context_note = ""

        return REDUCE_MERGE_SYSTEM_TEMPLATE.format(
            incident_context=self.config.incident_context or "No additional context provided.",
            incident_start=inc_start,
            incident_end=inc_end,
            context_note=context_note,
        )

    @staticmethod
    def _build_merge_user(group: list[_Item]) -> str:
        """Форматируем user-промпт: список JSON-объектов для merge.

        Включает зоны текущей группы и предупреждение о кросс-зональном merge.
        """
        zones = TreeReducer._merge_zones(group)
        is_cross_zone = "context_before" in zones and "incident" in zones

        parts = [f"Merge the following {len(group)} analyses into one MergedAnalysis.\n"]

        parts.append(f"Zones covered by this group: {', '.join(zones)}")
        if is_cross_zone:
            parts.append(
                "⚠ This group SPANS zone boundaries — actively search for cross-zone "
                "causal_chains where a context_before event caused or preceded an incident "
                "event. These are the most valuable links: build them explicitly."
            )
        parts.append("")

        for i, item in enumerate(group, 1):
            item_json = TreeReducer._item_to_json(item)
            item_zones = TreeReducer._zones_of(item)
            parts.append(f"### Analysis {i}  [zones: {', '.join(item_zones)}]")
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
        # Раскрываем batch_zone в список зон
        bz = batch.batch_zone
        if bz == "mixed":
            zones: list[str] = ["context_before", "incident", "context_after"]
        else:
            zones = [bz]

        return MergedAnalysis(
            time_range=batch.time_range,
            narrative=batch.narrative,
            events=batch.events,
            causal_chains=[],
            hypotheses=batch.hypotheses,
            anomalies=batch.anomalies,
            gaps=[],
            impact_summary=batch.metrics_context or "",
            preliminary_recommendations=batch.preliminary_recommendations,
            zones_covered=zones,
            evidence_bank=batch.evidence,
            alert_refs=[],  # управляются отдельно
        )

    @staticmethod
    def _zones_of(item: _Item) -> list[str]:
        """Возвращает список зон из BatchAnalysis или MergedAnalysis."""
        if isinstance(item, BatchAnalysis):
            bz = item.batch_zone
            return ["context_before", "incident", "context_after"] if bz == "mixed" else [bz]
        return list(item.zones_covered)

    @staticmethod
    def _merge_zones(group: list[_Item]) -> list[str]:
        """Union всех зон группы в уникальный отсортированный список."""
        order = {"context_before": 0, "incident": 1, "context_after": 2}
        seen: set[str] = set()
        for item in group:
            seen.update(TreeReducer._zones_of(item))
        return sorted(seen, key=lambda z: order.get(z, 9))

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

        # Уникальные рекомендации из всех батчей
        all_recs: list[str] = []
        seen_recs: set[str] = set()
        for item in group:
            recs = (
                item.preliminary_recommendations
                if isinstance(item, (BatchAnalysis, MergedAnalysis))
                else []
            )
            for r in recs:
                if r not in seen_recs:
                    seen_recs.add(r)
                    all_recs.append(r)

        return MergedAnalysis(
            time_range=(min(all_ts), max(all_ts)),
            narrative=" ".join(narratives),
            events=list(all_events.values()),
            causal_chains=all_causal,
            hypotheses=list(all_hyp.values()),
            anomalies=list(all_anomalies.values()),
            gaps=all_gaps,
            impact_summary="; ".join(impacts),
            preliminary_recommendations=all_recs,
            zones_covered=TreeReducer._merge_zones(group),
            alert_refs=[],  # управляются отдельно
        )
