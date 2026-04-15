"""
debug_reduce_merge.py — тест REDUCE-фазы (TreeReducer) на данных из файлов.

Загружает BatchAnalysis из BATCHES_DIR, прогоняет через TreeReducer,
наглядно показывает как саммари меняются на каждом уровне дерева.

Промпты и ответы LLM сохраняются в файлы (не в лог).
Промежуточные результаты каждого раунда сохраняются отдельно.

Запуск:
    python debug_reduce_merge.py
    python debug_reduce_merge.py --batches-dir test_batches
    python debug_reduce_merge.py --print-prompt
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:8000"
API_KEY   = "sk-placeholder"
MODEL     = "PNX.QWEN3 235b a22b instruct"

BATCHES_DIR            = pathlib.Path("test_batches")
ARTIFACTS_DIR          = pathlib.Path("artifacts/debug_reduce")

MAX_CONTEXT_TOKENS     = 100_000
MAX_GROUP_SIZE         = 3
MAX_EVENTS_PER_MERGE   = 30
COMPRESSION_TARGET_PCT = 50
# ─────────────────────────────────────────────────────────────────────

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import LLMClient
from log_summarizer.models import BatchAnalysis, MergedAnalysis
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import setup_pipeline_logging

setup_pipeline_logging("INFO")
log = logging.getLogger("debug_reduce_merge")

SEP  = "═" * 72
SEP2 = "─" * 72


class _Tee:
    """Пишет одновременно в несколько потоков (stdout + файл)."""
    def __init__(self, *streams: io.IOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            s.write(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


# ══════════════════════════════════════════════════════════════════════
#  Загрузка батчей
# ══════════════════════════════════════════════════════════════════════

def load_batches(batches_dir: pathlib.Path) -> list[BatchAnalysis]:
    if not batches_dir.exists():
        log.error("Директория не найдена: %s", batches_dir.resolve())
        sys.exit(1)
    files = sorted(batches_dir.glob("batch_*.json"))
    if not files:
        log.error("В %s нет файлов batch_*.json", batches_dir.resolve())
        sys.exit(1)
    batches: list[BatchAnalysis] = []
    for f in files:
        try:
            batch = BatchAnalysis.model_validate_json(f.read_text(encoding="utf-8"))
            batches.append(batch)
        except Exception as exc:
            log.error("Ошибка при загрузке %s: %s", f.name, exc)
            sys.exit(1)
    return batches


# ══════════════════════════════════════════════════════════════════════
#  Визуализация
# ══════════════════════════════════════════════════════════════════════

def _fmt_alerts(alert_refs: list) -> str:
    if not alert_refs:
        return "—"
    return "  ".join(f"{r.alert_id}={r.status.value}" for r in alert_refs)


def _fmt_confidences(hypotheses: list) -> str:
    if not hypotheses:
        return "—"
    counts: dict[str, int] = {}
    for h in hypotheses:
        counts[h.confidence] = counts.get(h.confidence, 0) + 1
    order = ["high", "medium", "low"]
    return " ".join(f"{c}×{counts[c]}" for c in order if c in counts)


def _fmt_importance(events: list) -> str:
    if not events:
        return "—"
    high   = sum(1 for e in events if e.importance >= 0.8)
    medium = sum(1 for e in events if 0.5 <= e.importance < 0.8)
    low    = sum(1 for e in events if e.importance < 0.5)
    parts = []
    if high:   parts.append(f"imp≥0.8: {high}")
    if medium: parts.append(f"imp≥0.5: {medium}")
    if low:    parts.append(f"imp<0.5: {low}")
    return "  ".join(parts)


def print_input(batches: list[BatchAnalysis]) -> None:
    print(f"\n{SEP}")
    print(f"  INPUT: {len(batches)} батчей")
    print(SEP)
    total_ev = sum(len(b.events) for b in batches)
    total_ev_text = sum(len(b.evidence) for b in batches)
    total_hyp = sum(len(b.hypotheses) for b in batches)
    print(f"  {'файл':<16} {'период':<28} {'events':>6} {'evidence':>8} {'hyp':>4}  {'confidence':<20}  {'alerts'}")
    print(f"  {SEP2}")
    for i, b in enumerate(batches):
        t0 = b.time_range[0].strftime("%H:%M")
        t1 = b.time_range[1].strftime("%H:%M")
        conf = _fmt_confidences(b.hypotheses)
        alerts = _fmt_alerts(b.alert_refs)
        print(
            f"  batch_{i:03d}         {t0}→{t1}                     "
            f"{len(b.events):>6} {len(b.evidence):>8} {len(b.hypotheses):>4}  {conf:<20}  {alerts}"
        )
    print(f"  {SEP2}")
    print(f"  {'ИТОГО':<16} {'':28} {total_ev:>6} {total_ev_text:>8} {total_hyp:>4}")


def print_rounds(reduce_dir: pathlib.Path) -> None:
    """Читает сохранённые JSON-файлы и печатает прогрессию по раундам."""
    if not reduce_dir.exists():
        return

    # Собираем round_NN_group_MM.json, группируем по раунду
    round_files: dict[int, list[pathlib.Path]] = {}
    for f in sorted(reduce_dir.glob("round_*_group_*.json")):
        parts = f.stem.split("_")  # ['round', 'NN', 'group', 'MM']
        try:
            rnum = int(parts[1])
            round_files.setdefault(rnum, []).append(f)
        except (IndexError, ValueError):
            pass

    for rnum in sorted(round_files):
        files = sorted(round_files[rnum])
        print(f"\n{SEP}")
        print(f"  ROUND {rnum}: {len(files)} элементов после merge")
        print(SEP)
        print(f"  {'группа':<12} {'период':<28} {'events':>6} {'causal':>6} {'hyp':>4}  {'confidence':<20}  {'importance':<28}  narrative")
        print(f"  {SEP2}")
        for f in files:
            try:
                m = MergedAnalysis.model_validate_json(f.read_text(encoding="utf-8"))
                t0 = m.time_range[0].strftime("%H:%M")
                t1 = m.time_range[1].strftime("%H:%M")
                conf    = _fmt_confidences(m.hypotheses)
                imp     = _fmt_importance(m.events)
                narr    = m.narrative[:60].replace("\n", " ")
                print(
                    f"  {f.stem:<12} {t0}→{t1}                     "
                    f"{len(m.events):>6} {len(m.causal_chains):>6} {len(m.hypotheses):>4}  "
                    f"{conf:<20}  {imp:<28}  {narr}…"
                )
                log.info("Round %d | %s → %s", rnum, f.stem, f)
            except Exception as exc:
                log.warning("Не удалось прочитать %s: %s", f, exc)


def print_final(result: MergedAnalysis, run_dir: pathlib.Path) -> None:
    print(f"\n{SEP}")
    print("  ИТОГОВЫЙ MergedAnalysis")
    print(SEP)
    t0 = result.time_range[0].isoformat()
    t1 = result.time_range[1].isoformat()
    print(f"  period        : {t0} → {t1}")
    print(f"  events        : {len(result.events)}  ({_fmt_importance(result.events)})")
    print(f"  causal_chains : {len(result.causal_chains)}")
    print(f"  hypotheses    : {len(result.hypotheses)}  confidence: {_fmt_confidences(result.hypotheses)}")
    print(f"  anomalies     : {len(result.anomalies)}")
    print(f"  gaps          : {len(result.gaps)}")
    print(f"  evidence_bank : {len(result.evidence_bank)}")
    print(f"  recs          : {len(result.preliminary_recommendations)}")

    if result.alert_refs:
        print(f"\n  alert coverage:")
        for ref in result.alert_refs:
            comment = f"  — {ref.comment}" if ref.comment else ""
            print(f"    {ref.alert_id}: {ref.status.value}{comment}")

    print(f"\n  narrative ({len(result.narrative)} chars):")
    for line in result.narrative[:600].split(". "):
        print(f"    {line.strip()}.")

    if result.hypotheses:
        print(f"\n  hypotheses:")
        for h in result.hypotheses:
            alerts = f"  alerts={h.related_alert_ids}" if h.related_alert_ids else ""
            print(f"    [{h.confidence}] {h.title}{alerts}")
            print(f"           {h.description[:120]}")

    if result.causal_chains:
        print(f"\n  causal chains:")
        for cc in result.causal_chains:
            print(f"    {cc.from_event_id} → {cc.to_event_id}  [{cc.confidence}]  {cc.description[:80]}")

    if result.preliminary_recommendations:
        print(f"\n  recommendations:")
        for r in result.preliminary_recommendations:
            print(f"    • {r}")

    print(f"\n  artifacts → {run_dir.resolve()}")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

async def run(batches_dir: pathlib.Path) -> None:
    batches = load_batches(batches_dir)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir    = ARTIFACTS_DIR / ts
    reduce_dir = run_dir / "reduce"
    llm_dir    = run_dir / "llm"
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Артефакты прогона → %s", run_dir.resolve())
    log.info(
        "Загружено батчей: %d | events: %d | evidence: %d | hyp: %d",
        len(batches),
        sum(len(b.events) for b in batches),
        sum(len(b.evidence) for b in batches),
        sum(len(b.hypotheses) for b in batches),
    )

    config = PipelineConfig(
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        incident_context="payments-service OOM cascade incident",
        incident_start=batches[0].time_range[0],
        incident_end=batches[-1].time_range[1],
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_group_size=MAX_GROUP_SIZE,
        max_events_per_merge=MAX_EVENTS_PER_MERGE,
        compression_target_pct=COMPRESSION_TARGET_PCT,
        model_supports_tool_calling=False,
    )

    # audit_dir → промпты и ответы LLM пишутся в файлы, не в лог
    llm = LLMClient(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        use_instructor=True,
        model_supports_tool_calling=False,
        audit_dir=llm_dir,
    )
    reducer = TreeReducer(llm, config, run_dir=reduce_dir)

    log.info("Запуск TreeReducer...")
    result = await reducer.reduce(batches, early_summaries=[])
    log.info("TreeReducer завершён")

    # Весь вывод print() идёт одновременно в stdout и в файл
    report_path = run_dir / "run_report.txt"
    with report_path.open("w", encoding="utf-8") as report_file:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, report_file)  # type: ignore[assignment]
        try:
            print_input(batches)
            print_rounds(reduce_dir)
            print_final(result, run_dir)
        finally:
            sys.stdout = old_stdout

    # Сохраняем финальный результат
    final_path = run_dir / "final_result.json"
    final_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Отчёт о прогоне     → %s", report_path)
    log.info("Финальный результат → %s", final_path)
    log.info("LLM промпты/ответы  → %s", llm_dir)
    log.info("Промежуточные merge → %s", reduce_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches-dir", default=str(BATCHES_DIR))
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Напечатать промпт для генерации тестовых батчей и выйти",
    )
    args = parser.parse_args()

    if args.print_prompt:
        print(GENERATE_BATCHES_PROMPT)
        return

    asyncio.run(run(pathlib.Path(args.batches_dir)))


GENERATE_BATCHES_PROMPT = """см. последний промпт который ты отправил нейронке"""

if __name__ == "__main__":
    main()
