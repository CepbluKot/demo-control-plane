"""
debug_pipeline_inspector.py — статический анализатор артефактов пайплайна.

Не делает LLM-вызовов. Читает stage_map_reduce.json из artifacts/,
моделирует REDUCE-дерево и показывает где будет overflow контекста.

Запуск:
    ./venv/bin/python debug_pipeline_inspector.py
    ./venv/bin/python debug_pipeline_inspector.py artifacts/debug_logs_summarizer/stage_map_reduce.json
"""
from __future__ import annotations

import json
import math
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────── конфиг инспектора ───────────────────────────────

# Путь к артефакту. Если пусто — берётся первый аргумент командной строки
# или дефолтный artifacts/debug_logs_summarizer_simple/stage_map_reduce.json.
ARTIFACT_PATH: str = ""

# Параметры должны совпадать с теми что вы используете в RUN_CONFIG.
REDUCE_GROUP_SIZE: int = 2
REDUCE_INPUT_MAX_CHARS: int = 40_000     # per-item лимит (для L2+)
CONTEXT_WINDOW_CHARS: int = 120_000     # ~30K токенов × 4 chars/tok — меняй под свою модель
SYSTEM_PROMPT_OVERHEAD: int = 4_000     # примерный размер системного промпта

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MapSummaryStats:
    index: int
    chars: int
    lines: int
    preview: str


@dataclass
class ReduceGroupStats:
    level: int
    group_index: int
    item_indices: List[int]
    total_input_chars: int
    estimated_prompt_chars: int
    overflows: bool
    overflow_margin: int  # отрицательный = overflow, позитивный = запас


@dataclass
class InspectorReport:
    artifact_path: str
    map_summaries_count: int
    map_stats: List[MapSummaryStats]
    reduce_tree: List[List[ReduceGroupStats]]   # reduce_tree[level][group]
    total_reduce_calls: int
    overflow_count: int
    worst_overflow_chars: int
    final_sections_prompt_chars: int
    final_sections_overflows: bool


def _load_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Артефакт не найден: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate_prompt_chars(
    *,
    input_chars: int,
    system_overhead: int,
) -> int:
    # Промпт = system overhead + JSON wrapper (~200 chars) + сами данные
    return system_overhead + 200 + input_chars


def _simulate_reduce_tree(
    summaries: List[MapSummaryStats],
    *,
    group_size: int,
    reduce_input_max_chars: int,
    context_window_chars: int,
    system_overhead: int,
) -> Tuple[List[List[ReduceGroupStats]], int, int]:
    """
    Моделирует REDUCE-дерево так как его делает _reduce_summaries_structured.
    На L1 нет pre-compression (баг). На L2+ применяется per-item лимит.
    Возвращает (дерево, кол-во overflow, максимальный overflow в chars).
    """
    tree: List[List[ReduceGroupStats]] = []
    overflow_count = 0
    worst_overflow = 0

    # Текущий уровень — индексы саммари и их размеры
    current: List[Tuple[int, int]] = [(s.index, s.chars) for s in summaries]
    level = 0

    while len(current) > 1:
        level += 1
        level_groups: List[ReduceGroupStats] = []
        next_level: List[Tuple[int, int]] = []

        # На L2+ применяем per-item лимит (как _prepare_summary_for_reduce_level)
        if level >= 2:
            current = [
                (idx, min(chars, reduce_input_max_chars))
                for idx, chars in current
            ]

        groups = [current[i : i + group_size] for i in range(0, len(current), group_size)]

        for g_idx, group in enumerate(groups):
            item_indices = [idx for idx, _ in group]
            total_chars = sum(chars for _, chars in group)
            prompt_chars = _estimate_prompt_chars(
                input_chars=total_chars,
                system_overhead=system_overhead,
            )
            overflows = prompt_chars > context_window_chars
            margin = context_window_chars - prompt_chars

            if overflows:
                overflow_count += 1
                worst_overflow = max(worst_overflow, -margin)

            grp_stat = ReduceGroupStats(
                level=level,
                group_index=g_idx,
                item_indices=item_indices,
                total_input_chars=total_chars,
                estimated_prompt_chars=prompt_chars,
                overflows=overflows,
                overflow_margin=margin,
            )
            level_groups.append(grp_stat)

            # Результат группы ≈ 50% от input (typical compression)
            merged_chars = max(total_chars // 2, 500)
            next_level.append((g_idx, merged_chars))

        tree.append(level_groups)
        current = next_level

    return tree, overflow_count, worst_overflow


def _analyze(artifact: Dict[str, Any], config: Dict[str, Any]) -> InspectorReport:
    group_size = int(config["group_size"])
    reduce_input_max_chars = int(config["reduce_input_max_chars"])
    context_window_chars = int(config["context_window_chars"])
    system_overhead = int(config["system_overhead"])

    raw_summaries: List[Any] = list(artifact.get("map_summaries") or [])

    map_stats: List[MapSummaryStats] = []
    for idx, item in enumerate(raw_summaries):
        text = str(item or "")
        chars = len(text)
        lines = text.count("\n") + 1
        preview = text[:120].replace("\n", " ")
        map_stats.append(MapSummaryStats(
            index=idx,
            chars=chars,
            lines=lines,
            preview=preview,
        ))

    tree, overflow_count, worst_overflow = _simulate_reduce_tree(
        summaries=map_stats,
        group_size=group_size,
        reduce_input_max_chars=reduce_input_max_chars,
        context_window_chars=context_window_chars,
        system_overhead=system_overhead,
    )

    total_reduce_calls = sum(len(lvl) for lvl in tree)

    # Финальный секции: ALL map summaries в один промпт (баг в pipeline)
    all_map_chars = sum(s.chars for s in map_stats)
    # Заголовки "[MAP SUMMARY #N]\n" ~ 20 chars * N
    headers_chars = 20 * len(map_stats)
    final_prompt_chars = _estimate_prompt_chars(
        input_chars=all_map_chars + headers_chars,
        system_overhead=system_overhead * 3,   # final sections system prompt крупнее
    )
    final_overflows = final_prompt_chars > context_window_chars

    stats = artifact.get("stats") or {}

    return InspectorReport(
        artifact_path=config.get("artifact_path", ""),
        map_summaries_count=len(map_stats),
        map_stats=map_stats,
        reduce_tree=tree,
        total_reduce_calls=total_reduce_calls,
        overflow_count=overflow_count,
        worst_overflow_chars=worst_overflow,
        final_sections_prompt_chars=final_prompt_chars,
        final_sections_overflows=final_overflows,
    )


def _fmt_chars(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _bar(value: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return " " * width
    filled = int(round(width * min(value, total) / total))
    return "█" * filled + "░" * (width - filled)


def _print_report(report: InspectorReport, context_window: int) -> None:
    sep = "═" * 72
    thin = "─" * 72

    print(sep)
    print("  PIPELINE INSPECTOR REPORT")
    print(sep)
    print(f"  Артефакт : {report.artifact_path}")
    print(f"  MAP саммари: {report.map_summaries_count}")
    print(f"  REDUCE вызовов LLM: {report.total_reduce_calls}")
    overflow_marker = " ← ПРОБЛЕМА!" if report.overflow_count > 0 else " ✓ ok"
    print(f"  Переполнений контекста: {report.overflow_count}{overflow_marker}")
    if report.worst_overflow_chars > 0:
        print(f"  Худший overflow: +{_fmt_chars(report.worst_overflow_chars)} chars сверх лимита")
    print()

    # ── MAP summaries ──────────────────────────────────────────────────────
    print("MAP SUMMARIES")
    print(thin)
    if not report.map_stats:
        print("  (нет MAP-саммари в артефакте)")
    else:
        max_chars = max(s.chars for s in report.map_stats) or 1
        total_chars = sum(s.chars for s in report.map_stats)
        print(f"  {'#':>4}  {'Chars':>8}  {'Bar (rel)':30}  Preview")
        print(f"  {'─'*4}  {'─'*8}  {'─'*30}  {'─'*30}")
        for s in report.map_stats:
            bar = _bar(s.chars, max_chars)
            overflow = " ← large!" if s.chars > REDUCE_INPUT_MAX_CHARS else ""
            print(f"  {s.index:>4}  {_fmt_chars(s.chars):>8}  {bar}  {s.preview[:35]!r}{overflow}")
        print(f"\n  Итого: {_fmt_chars(total_chars)} chars, avg {_fmt_chars(total_chars // max(len(report.map_stats), 1))}/summary")
    print()

    # ── REDUCE tree ────────────────────────────────────────────────────────
    print("REDUCE TREE (моделирование)")
    print(thin)
    if not report.reduce_tree:
        print("  Нет REDUCE-раундов (только 1 MAP или 0 суммарий).")
    for level_groups in report.reduce_tree:
        level = level_groups[0].level
        overflows_in_level = sum(1 for g in level_groups if g.overflows)
        level_note = f"  ← {overflows_in_level} overflow(s)!" if overflows_in_level else ""
        print(f"\n  L{level} — {len(level_groups)} групп{level_note}")
        for g in level_groups:
            items_str = ",".join(str(i) for i in g.item_indices)
            marker = " ← OVERFLOW" if g.overflows else ""
            margin_str = (
                f"-{_fmt_chars(-g.overflow_margin)} OVERFLOW"
                if g.overflow_margin < 0
                else f"+{_fmt_chars(g.overflow_margin)} ok"
            )
            bar = _bar(
                min(g.estimated_prompt_chars, context_window),
                context_window,
            )
            print(
                f"    G{g.group_index}  items=[{items_str}]  "
                f"input={_fmt_chars(g.total_input_chars)}  "
                f"prompt={_fmt_chars(g.estimated_prompt_chars)}  "
                f"{bar}  {margin_str}{marker}"
            )
    print()

    # ── Final sections ─────────────────────────────────────────────────────
    print("FINAL SECTIONS STAGE")
    print(thin)
    fs_bar = _bar(min(report.final_sections_prompt_chars, context_window), context_window)
    fs_marker = " ← OVERFLOW!" if report.final_sections_overflows else " ✓ ok"
    print(
        f"  Все MAP в одном промпте: {_fmt_chars(report.final_sections_prompt_chars)} chars  "
        f"{fs_bar}  {fs_marker}"
    )
    if report.final_sections_overflows:
        print(
            "  ПРИЧИНА: debug_logs_summarizer_pipeline.py:517 "
            "конкатенирует ВСЕ map_summaries в один str перед LLM-вызовом."
        )
        print(
            f"  ФИКС: передавать только base_summary (reduce-результат), "
            f"а не raw MAP-саммари в final sections."
        )
    print()

    # ── Ключевые рекомендации ──────────────────────────────────────────────
    print("РЕКОМЕНДАЦИИ")
    print(thin)

    issues = []

    # L1 no pre-compression
    if report.reduce_tree:
        l1_overflows = sum(1 for g in report.reduce_tree[0] if g.overflows)
        if l1_overflows > 0:
            issues.append((
                "L1 REDUCE: нет pre-compression MAP-саммари",
                f"{l1_overflows} групп(ы) на L1 переполняют контекст.",
                "Добавить вызов _prepare_summary_for_reduce_level на L1 тоже,\n"
                "    или добавить max_map_summary_chars в SummarizerConfig и обрезать\n"
                "    MAP-саммари до REDUCE_INPUT_MAX_CHARS перед L1."
            ))
        large_map = [s for s in report.map_stats if s.chars > REDUCE_INPUT_MAX_CHARS]
        if large_map:
            issues.append((
                "Большие MAP-саммари",
                f"{len(large_map)} MAP-саммари превышают reduce_input_max_chars={_fmt_chars(REDUCE_INPUT_MAX_CHARS)}.",
                "Уменьшить llm_batch_size или снизить max_cell_chars в RUN_CONFIG."
            ))

    if report.final_sections_overflows:
        issues.append((
            "Final sections: слишком много данных",
            f"Промпт финальных секций ≈{_fmt_chars(report.final_sections_prompt_chars)} — не влезет.",
            "Передавать только base_summary (REDUCE-результат) вместо всех MAP-саммари.\n"
            "    Или добавить truncation: map_summaries[:10] + '... (truncated)'"
        ))

    if not issues:
        print("  Видимых проблем не обнаружено для заданных параметров контекстного окна.")
    else:
        for i, (title, detail, fix) in enumerate(issues, 1):
            print(f"  [{i}] {title}")
            print(f"      Проблема: {detail}")
            print(f"      Фикс    : {fix}")
            print()

    print(sep)


def _make_synthetic_artifact(
    *,
    n_batches: int = 20,
    chars_per_summary: int = 8_000,
    jitter: float = 0.5,
) -> Dict[str, Any]:
    """
    Генерирует синтетический артефакт без реального DB/LLM.
    Используется для теста пайплайна при отсутствии реальных данных.
    """
    import random
    rng = random.Random(42)
    word_pool = (
        "error exception timeout connection refused null pointer stacktrace "
        "kubernetes pod container cluster namespace deployment restart crash "
        "OOMKilled evicted node memory cpu limit throttled failed health "
        "readiness liveness probe slow response latency p99 p95 alert warn"
    ).split()
    summaries = []
    for i in range(n_batches):
        size = int(chars_per_summary * (1 + rng.uniform(-jitter, jitter)))
        size = max(size, 200)
        # Строим "JSON-подобный" текст чтобы размер был близким к реальному
        lines = []
        while sum(len(l) for l in lines) < size:
            w = " ".join(rng.choices(word_pool, k=rng.randint(5, 20)))
            lines.append(f'  "event_{rng.randint(1000, 9999)}": "{w}",')
        text = "{\n" + "\n".join(lines) + "\n}"
        summaries.append(text[:size])
    return {
        "map_summaries": summaries,
        "stats": {
            "pages_fetched": n_batches * 5,
            "rows_processed": n_batches * 200,
            "llm_calls": n_batches,
            "reduce_rounds": 0,
            "rows_total_estimate": n_batches * 200,
        },
    }


def main(artifact_path_override: Optional[str] = None) -> None:
    # Режим: --synthetic [n_batches] [chars_per_summary]
    args = sys.argv[1:]
    if args and args[0] == "--synthetic":
        n_batches = int(args[1]) if len(args) > 1 else 20
        chars_per = int(args[2]) if len(args) > 2 else 8_000
        print(f"Синтетический режим: {n_batches} батчей × ~{chars_per} chars/summary")
        artifact = _make_synthetic_artifact(n_batches=n_batches, chars_per_summary=chars_per)
        config = {
            "artifact_path": f"<synthetic n={n_batches} chars={chars_per}>",
            "group_size": REDUCE_GROUP_SIZE,
            "reduce_input_max_chars": REDUCE_INPUT_MAX_CHARS,
            "context_window_chars": CONTEXT_WINDOW_CHARS,
            "system_overhead": SYSTEM_PROMPT_OVERHEAD,
        }
        report = _analyze(artifact, config)
        _print_report(report, CONTEXT_WINDOW_CHARS)
        return

    # Определяем путь к артефакту
    raw_path = (
        artifact_path_override
        or ARTIFACT_PATH
        or (args[0] if args else "")
        or "artifacts/debug_logs_summarizer_simple/stage_map_reduce.json"
    )
    path = Path(raw_path)

    print(f"Загружаю артефакт: {path.resolve()}")
    artifact = _load_artifact(path)

    config = {
        "artifact_path": str(path.resolve()),
        "group_size": REDUCE_GROUP_SIZE,
        "reduce_input_max_chars": REDUCE_INPUT_MAX_CHARS,
        "context_window_chars": CONTEXT_WINDOW_CHARS,
        "system_overhead": SYSTEM_PROMPT_OVERHEAD,
    }

    report = _analyze(artifact, config)
    _print_report(report, CONTEXT_WINDOW_CHARS)

    # Дополнительно: распечатать размеры каждого MAP-саммари
    if report.map_stats:
        print("\nDETAIL: Размеры всех MAP-саммари")
        print("─" * 40)
        for s in report.map_stats:
            flag = " ← LARGE" if s.chars > REDUCE_INPUT_MAX_CHARS else ""
            print(f"  MAP #{s.index:03d}: {_fmt_chars(s.chars):>7} chars{flag}")


if __name__ == "__main__":
    main()
