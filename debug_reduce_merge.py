"""
debug_reduce_merge.py — тест REDUCE-фазы (TreeReducer) на синтетических данных.

Не требует ClickHouse. Создаёт N синтетических BatchAnalysis-объектов
(разные временные окна инцидента), прогоняет через TreeReducer,
проверяет что дерево сжатия сходится и не теряет ключевые события.

Запуск:
    python debug_reduce_merge.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from datetime import datetime, timedelta, timezone

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
API_KEY    = "sk-placeholder"
MODEL      = "PNX.QWEN3 235b a22b instruct"

# Сколько синтетических батчей подать на вход (≥2, интереснее с 6-10)
N_BATCHES = 8

MAX_CONTEXT_TOKENS      = 100_000
MAX_GROUP_SIZE          = 3   # сколько item объединяем за раз
MAX_EVENTS_PER_MERGE    = 30  # обрезка events после каждого merge
COMPRESSION_TARGET_PCT  = 50  # сжимать если > 50% контекста
# ─────────────────────────────────────────────────────────────────────

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import LLMClient
from log_summarizer.models import (
    Anomaly, BatchAnalysis, Event, Evidence, Hypothesis, Severity,
)
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import setup_pipeline_logging

setup_pipeline_logging("INFO")
log = logging.getLogger("debug_reduce_merge")


def _dt(offset_min: int) -> datetime:
    base = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_min)


def make_batch(idx: int) -> BatchAnalysis:
    """Один синтетический батч — временное окно из ~5 минут инцидента."""
    t0 = idx * 5   # каждый батч = 5 минут
    t1 = t0 + 5

    window_labels = [
        ("pre-incident: normal traffic",      "INFO",     "api-gateway",    []),
        ("GC pressure rising in payments",    "WARN",     "payments-svc",   ["oom", "gc"]),
        ("OOM: payments-svc pod killed",      "CRITICAL", "kubelet",        ["oom", "crash"]),
        ("downstream cascade: order-svc CB",  "HIGH",     "order-svc",      ["circuit-breaker"]),
        ("second OOM: CrashLoopBackOff",      "HIGH",     "kubelet",        ["oom", "crash"]),
        ("memory limit patched by oncall",    "MEDIUM",   "kubectl",        ["fix"]),
        ("pod recovery: payments-svc up",     "MEDIUM",   "payments-svc",   ["recovery"]),
        ("full recovery confirmed by prom",   "LOW",      "prometheus",     ["recovery"]),
    ]
    label, sev_str, source, tags = window_labels[idx % len(window_labels)]
    sev = Severity(sev_str.lower()) if sev_str.lower() in Severity._value2member_map_ else Severity.MEDIUM

    ev = Event(
        id=f"evt-{idx:03d}-001",
        timestamp=_dt(t0 + 2),
        source=source,
        description=label,
        severity=sev,
        tags=tags,
    )
    evidence = Evidence(
        id=f"ev-{idx:03d}-001",
        timestamp=_dt(t0 + 2),
        source=source,
        raw_line=f"{_dt(t0+2).isoformat()} {sev_str} {source}: {label}",
        severity=sev,
        linked_event_id=ev.id,
    )
    hyp = Hypothesis(
        id=f"hyp-{idx:03d}-001",
        title=f"Hypothesis from window {idx}",
        description=f"Window {idx} suggests {label} as contributing factor",
        confidence="medium",
        supporting_event_ids=[ev.id],
    )
    return BatchAnalysis(
        time_range=(_dt(t0), _dt(t1)),
        narrative=f"Window {idx} ({_dt(t0).isoformat()} → {_dt(t1).isoformat()}): {label}.",
        events=[ev],
        evidence=[evidence],
        hypotheses=[hyp],
        anomalies=[Anomaly(description=f"anomaly in window {idx}", related_event_ids=[ev.id])],
    )


async def main() -> None:
    config = PipelineConfig(
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        incident_context="payments-service OOM cascade incident",
        incident_start=_dt(0),
        incident_end=_dt(N_BATCHES * 5),
        max_context_tokens=MAX_CONTEXT_TOKENS,
        max_group_size=MAX_GROUP_SIZE,
        max_events_per_merge=MAX_EVENTS_PER_MERGE,
        compression_target_pct=COMPRESSION_TARGET_PCT,
        model_supports_tool_calling=False,
    )

    batches = [make_batch(i) for i in range(N_BATCHES)]
    log.info("Input batches: %d", len(batches))
    log.info(
        "Events total: %d | Evidence total: %d",
        sum(len(b.events) for b in batches),
        sum(len(b.evidence) for b in batches),
    )

    llm = LLMClient(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        use_instructor=True,
        model_supports_tool_calling=False,
    )
    reducer = TreeReducer(llm, config)

    result = await reducer.reduce(batches, early_summaries=[])

    sep = "=" * 70
    print(f"\n{sep}")
    print("REDUCE RESULT")
    print(sep)
    print(f"  time_range   : {result.time_range[0].isoformat()} → {result.time_range[1].isoformat()}")
    print(f"  events       : {len(result.events)}")
    print(f"  evidence_bank: {len(result.evidence_bank)}")
    print(f"  hypotheses   : {len(result.hypotheses)}")
    print(f"  causal_chains: {len(result.causal_chains)}")
    print(f"  anomalies    : {len(result.anomalies)}")
    print(f"  gaps         : {len(result.gaps)}")
    print(f"\n  narrative:\n    {result.narrative[:400]}")
    print(f"\n  impact_summary:\n    {result.impact_summary[:200]}")
    if result.hypotheses:
        print(f"\n  top hypothesis: [{result.hypotheses[0].confidence}] {result.hypotheses[0].title}")
    print(sep)

    pathlib.Path("artifacts").mkdir(exist_ok=True)
    out_path = pathlib.Path("artifacts/debug_reduce_result.json")
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Saved → %s", out_path.resolve())


if __name__ == "__main__":
    asyncio.run(main())
