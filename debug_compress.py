"""
debug_compress.py — тест сжатия MergedAnalysis через LLM.

Создаёт намеренно раздутый MergedAnalysis (многословные описания),
прогоняет через TreeReducer._compress, печатает до/после.

Не требует ClickHouse. Не требует test_batches/.

Запуск:
    python debug_compress.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
API_KEY  = "sk-placeholder"
MODEL    = "PNX.QWEN3 235b a22b instruct"
INPUT_ANALYSIS_PATH = pathlib.Path("test_merged/merged_analysis.json")

MAX_CONTEXT_TOKENS     = 100_000
COMPRESSION_TARGET_PCT = 50
# ─────────────────────────────────────────────────────────────────────

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import LLMClient
from log_summarizer.models import (
    Anomaly,
    CausalChain,
    Evidence,
    Event,
    Hypothesis,
    MergedAnalysis,
    Severity,
)
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import setup_pipeline_logging
from log_summarizer.utils.tokens import estimate_tokens

setup_pipeline_logging("INFO")
log = logging.getLogger("debug_compress")


def _dt(h: int, m: int) -> datetime:
    return datetime(2024, 1, 15, h, m, 0, tzinfo=timezone.utc)


def make_bloated_analysis() -> MergedAnalysis:
    """MergedAnalysis с намеренно многословными описаниями."""
    return MergedAnalysis(
        time_range=(_dt(14, 0), _dt(15, 10)),
        narrative=(
            "At approximately 14:30 UTC on January 15, 2024, the payments-service "
            "began exhibiting severe memory pressure due to an insufficiently configured "
            "Java heap size limit. The JVM garbage collector was unable to reclaim "
            "sufficient memory to sustain normal operation, causing extended GC pause "
            "times that eventually escalated from 1.2 seconds at 14:25 UTC to over 5 "
            "seconds by 14:30 UTC, at which point the JVM threw OutOfMemoryError and "
            "the Kubernetes kubelet OOM-killed the pod. "
            "Following the OOM kill, the pod was automatically restarted by Kubernetes, "
            "but because the underlying root cause — insufficient heap memory — had not "
            "been addressed, the newly started pod quickly filled its heap again and "
            "was killed a second time at 14:44 UTC, then a third time at 14:56 UTC, "
            "leading the pod into CrashLoopBackOff state. "
            "The downstream cascade manifested as follows: api-gateway began receiving "
            "HTTP 503 errors from payments-svc, exhausted its retry budget, and started "
            "returning errors to end clients. Meanwhile, order-svc detected sustained "
            "100% error rate from payments-svc and opened its circuit breaker at 14:41 UTC, "
            "blocking all payment-dependent order flows for approximately 29 minutes. "
            "The on-call engineer identified the issue at 15:00 UTC via PagerDuty alert "
            "and increased the memory limit from 2Gi to 4Gi using kubectl patch. "
            "A new pod with the updated limit was scheduled and started successfully at "
            "15:02 UTC. The circuit breaker in order-svc transitioned to HALF_OPEN at "
            "15:05 UTC and fully CLOSED at 15:08 UTC after successful probe requests. "
            "All downstream error rates returned to normal by 15:10 UTC."
        ),
        events=[
            Event(
                id="evt-001",
                timestamp=_dt(14, 30),
                source="payments-svc",
                description=(
                    "JVM threw OutOfMemoryError: Java heap space after sustained GC pressure "
                    "lasting approximately 5 minutes, with old gen utilization reaching 97%"
                ),
                severity=Severity.critical,
                tags=["oom", "jvm", "heap"],
            ),
            Event(
                id="evt-002",
                timestamp=_dt(14, 30),
                source="kubelet",
                description=(
                    "Pod payments-svc-7d8f9c-xk2p4 was OOM-killed by the Linux kernel OOM "
                    "killer because it exceeded its configured memory limit of 2 gigabytes"
                ),
                severity=Severity.critical,
                tags=["oom", "pod-kill", "kubernetes"],
            ),
            Event(
                id="evt-003",
                timestamp=_dt(14, 41),
                source="order-svc",
                description=(
                    "CircuitBreaker for payments-svc dependency transitioned to OPEN state "
                    "after detecting 100% error rate sustained for 60 consecutive seconds, "
                    "blocking all payment-dependent order processing flows"
                ),
                severity=Severity.high,
                tags=["circuit-breaker", "cascade", "order-svc"],
            ),
            Event(
                id="evt-004",
                timestamp=_dt(14, 56),
                source="kubelet",
                description=(
                    "Pod payments-svc entered CrashLoopBackOff state after third consecutive "
                    "OOM kill, with back-off delay of 20 seconds between restart attempts; "
                    "root cause of repeated crashes was unresolved heap memory exhaustion"
                ),
                severity=Severity.high,
                tags=["crashloop", "kubernetes", "oom"],
            ),
            Event(
                id="evt-005",
                timestamp=_dt(15, 0),
                source="kubectl",
                description=(
                    "On-call engineer applied remediation by patching deployment resource "
                    "limits, increasing memory limit from 2Gi to 4Gi and memory request "
                    "from 1Gi to 2Gi to prevent immediate re-occurrence of OOM condition"
                ),
                severity=Severity.medium,
                tags=["remediation", "memory-limit", "kubectl"],
            ),
            Event(
                id="evt-006",
                timestamp=_dt(15, 8),
                source="order-svc",
                description=(
                    "CircuitBreaker for payments-svc transitioned from HALF_OPEN back to "
                    "CLOSED state after 3 consecutive successful probe requests confirmed "
                    "payments-svc was fully operational with new memory configuration"
                ),
                severity=Severity.low,
                tags=["circuit-breaker", "recovery", "order-svc"],
            ),
        ],
        hypotheses=[
            Hypothesis(
                id="hyp-001",
                title="Memory leak or workload spike caused heap exhaustion",
                description=(
                    "The payments-service heap limit of 2Gi was either always insufficient "
                    "for the current workload volume, or a memory leak gradually consumed "
                    "available heap over the hours preceding the incident. The absence of "
                    "heap dump artifacts makes it difficult to distinguish between these "
                    "two root causes without further profiling. GC logs showing sustained "
                    "old gen pressure starting ~5 minutes before OOM support the gradual "
                    "exhaustion hypothesis rather than a sudden spike."
                ),
                confidence="high",
                supporting_event_ids=["evt-001", "evt-002", "evt-004"],
                contradicting_event_ids=[],
            ),
            Hypothesis(
                id="hyp-002",
                title="Insufficient alerting on heap utilization allowed OOM to occur",
                description=(
                    "No alert fired on heap utilization reaching 78% or 89%, allowing the "
                    "condition to progress undetected to OOM. The first alert was the "
                    "KubePodCrashLooping alert, which fires only after multiple restarts — "
                    "by which point significant user impact had already occurred. Adding "
                    "a JVM heap utilization alert at 80% threshold would have given the "
                    "team ~5 minutes to intervene before the first OOM."
                ),
                confidence="medium",
                supporting_event_ids=["evt-001"],
                contradicting_event_ids=[],
            ),
        ],
        causal_chains=[
            CausalChain(
                root_cause_event_id="evt-001",
                chain=["evt-001", "evt-002", "evt-003", "evt-004"],
                description=(
                    "JVM OOM → pod killed by kubelet → payments-svc unavailable → "
                    "order-svc circuit breaker opened → downstream order failures"
                ),
            )
        ],
        anomalies=[
            Anomaly(
                description=(
                    "GC pause duration increased monotonically from 0.1s baseline to 5.1s "
                    "over a 5-minute window without any intermediate alerting, indicating "
                    "a gap in observability for JVM internals"
                ),
                related_event_ids=["evt-001"],
            )
        ],
        gaps=[
            "No heap dumps were captured at time of OOM — profiling not possible post-incident",
            "Exact timeline of memory growth before 14:25 UTC unavailable — logs not retained",
        ],
        impact_summary=(
            "Payment processing was fully unavailable for approximately 32 minutes "
            "(14:30–15:02 UTC). Order placement was blocked for ~29 minutes "
            "(14:41–15:10 UTC) due to circuit breaker. Estimated transaction volume "
            "impact: ~3,200 payment attempts failed or were not attempted."
        ),
        evidence_bank=[
            Evidence(
                id="ev-001",
                timestamp=_dt(14, 30),
                source="payments-svc",
                raw_line="2024-01-15T14:30:12Z ERROR payments-svc: java.lang.OutOfMemoryError: Java heap space",
                severity=Severity.critical,
                linked_event_id="evt-001",
            ),
            Evidence(
                id="ev-002",
                timestamp=_dt(14, 30),
                source="kubelet",
                raw_line="2024-01-15T14:30:13Z INFO kubelet: OOMKilled: payments-svc-7d8f9c-xk2p4 (limit 2Gi)",
                severity=Severity.critical,
                linked_event_id="evt-002",
            ),
        ],
    )


def load_analysis_from_file(path: pathlib.Path) -> MergedAnalysis:
    """Загружает MergedAnalysis из JSON-файла."""
    raw = path.read_text(encoding="utf-8")
    return MergedAnalysis.model_validate_json(raw)


async def main() -> None:
    if INPUT_ANALYSIS_PATH.exists():
        analysis = load_analysis_from_file(INPUT_ANALYSIS_PATH)
        log.info("Loaded analysis from %s", INPUT_ANALYSIS_PATH)
    else:
        analysis = make_bloated_analysis()
        log.warning(
            "Input file %s not found, using built-in bloated fixture",
            INPUT_ANALYSIS_PATH,
        )

    original_json = analysis.to_json_str()
    original_tokens = estimate_tokens(original_json)
    log.info("Original size: %d chars / ~%d tokens", len(original_json), original_tokens)

    config = PipelineConfig(
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        incident_context="payments-service OOM cascade",
        incident_start=_dt(14, 0),
        incident_end=_dt(15, 10),
        max_context_tokens=MAX_CONTEXT_TOKENS,
        compression_target_pct=COMPRESSION_TARGET_PCT,
        model_supports_tool_calling=False,
    )
    llm = LLMClient(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        use_instructor=True,
        model_supports_tool_calling=False,
    )
    reducer = TreeReducer(llm, config)

    log.info("Running compression...")
    compressed = await reducer._compress(analysis)

    compressed_json = compressed.to_json_str()
    compressed_tokens = estimate_tokens(compressed_json)

    sep = "=" * 70
    print(f"\n{sep}")
    print("COMPRESSION RESULT")
    print(sep)
    print(f"  original : {len(original_json):>7} chars  /  ~{original_tokens:>6} tokens")
    print(f"  compressed: {len(compressed_json):>7} chars  /  ~{compressed_tokens:>6} tokens")
    print(f"  ratio    : {len(compressed_json) / len(original_json):.0%} of original")
    print(f"\n  events     : {len(analysis.events)} → {len(compressed.events)}")
    print(f"  hypotheses : {len(analysis.hypotheses)} → {len(compressed.hypotheses)}")
    print(f"  evidence   : {len(analysis.evidence_bank)} → {len(compressed.evidence_bank)}")

    print(f"\n── narrative BEFORE ────────────────────────────────────────────────")
    print(f"  {analysis.narrative[:400]}")
    print(f"\n── narrative AFTER  ────────────────────────────────────────────────")
    print(f"  {compressed.narrative[:400]}")
    print(sep)

    out_dir = pathlib.Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "debug_compress_before.json").write_text(
        original_json, encoding="utf-8"
    )
    (out_dir / "debug_compress_after.json").write_text(
        compressed_json, encoding="utf-8"
    )
    log.info("Saved → artifacts/debug_compress_before.json")
    log.info("Saved → artifacts/debug_compress_after.json")


if __name__ == "__main__":
    asyncio.run(main())
