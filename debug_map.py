"""
debug_map.py — тест MAP-фазы на синтетических логах.

Не требует ClickHouse. Создаёт ~60 синтетических строк лога
(OOM + рестарты + connection errors), прогоняет через MapProcessor,
печатает BatchAnalysis.

Запуск:
    python debug_map.py
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
API_KEY    = "sk-placeholder"
MODEL      = "PNX.QWEN3 235b a22b instruct"

INCIDENT_CONTEXT = "payments-service OOM, pod restarts after 14:30 UTC"
INCIDENT_START   = "2024-01-15T14:00:00"
INCIDENT_END     = "2024-01-15T15:00:00"

MAX_CONTEXT_TOKENS = 100_000
# ─────────────────────────────────────────────────────────────────────

from log_summarizer.chunker import Chunker
from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import LLMClient
from log_summarizer.map_processor import MapProcessor
from log_summarizer.models import LogRow
from log_summarizer.utils.logging import setup_pipeline_logging

setup_pipeline_logging("INFO")
log = logging.getLogger("debug_map")


def _dt(offset_min: int) -> datetime:
    base = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_min)


def make_synthetic_rows() -> list[LogRow]:
    """60 строк, имитирующих Kubernetes инцидент."""
    rows: list[LogRow] = []

    def add(offset_min: int, level: str, source: str, msg: str) -> None:
        ts = _dt(offset_min)
        raw = f"{ts.isoformat()} {level} {source}: {msg}"
        rows.append(LogRow(timestamp=ts, level=level, source=source, message=msg, raw_line=raw))

    # t+0: нормальный трафик
    for i in range(5):
        add(i, "INFO", "api-gateway", "POST /api/v1/payment 200 OK 45ms")

    # t+5: первые признаки
    add(5,  "WARN",  "payments-svc", "GC pause 1.2s (old gen 78%)")
    add(6,  "WARN",  "payments-svc", "GC pause 2.4s (old gen 89%)")
    add(7,  "ERROR", "payments-svc", "GC pause 5.1s (old gen 97%) — potential OOM")
    add(8,  "ERROR", "payments-svc", "java.lang.OutOfMemoryError: Java heap space")
    add(8,  "ERROR", "payments-svc", "\tat com.bank.payments.ProcessorService.process(ProcessorService.java:142)")
    add(9,  "ERROR", "kubelet",      "OOMKilled: payments-svc-7d8f9c-xk2p4 (limit 2Gi)")
    add(9,  "INFO",  "kubelet",      "Pulling image payments-svc:v3.2.1 for restart")

    # t+10: лавина ошибок downstream
    for i in range(3):
        add(10 + i, "ERROR", "api-gateway", f"upstream connect error: payments-svc UNAVAILABLE (attempt {i+1})")
    add(10, "ERROR", "order-svc",    "payment dependency timeout after 30s")
    add(11, "ERROR", "order-svc",    "CircuitBreaker OPEN: payments-svc error_rate=100% for 60s")
    add(11, "WARN",  "api-gateway",  "retry budget exhausted for /api/v1/payment")

    # t+12: pod поднимается
    add(12, "INFO",  "kubelet",      "payments-svc-7d8f9c-xk2p4: Started container, phase=Running")
    add(13, "INFO",  "payments-svc", "Application started in 8.3s (JVM running for 9.1s)")
    add(13, "WARN",  "payments-svc", "GC pause 0.3s (old gen 45%)")  # ещё не стабильно

    # t+14: второй OOM
    add(14, "ERROR", "payments-svc", "java.lang.OutOfMemoryError: Java heap space")
    add(14, "ERROR", "kubelet",      "OOMKilled: payments-svc-7d8f9c-xk2p4 (restart #2)")

    # t+16: третий перезапуск, pod входит в CrashLoopBackOff
    add(16, "WARN",  "kubelet",      "payments-svc-7d8f9c-xk2p4: back-off 20s restarting failed container")
    add(17, "WARN",  "kubelet",      "payments-svc-7d8f9c-xk2p4: CrashLoopBackOff")

    # t+18: аномалии в метриках
    add(18, "WARN",  "prometheus",   "FIRING: PaymentsLatencyHigh p99=45000ms threshold=5000ms for 3m")
    add(18, "WARN",  "prometheus",   "FIRING: PaymentsErrorRate 100% threshold=5% for 3m")
    add(19, "WARN",  "prometheus",   "FIRING: KubePodCrashLooping payments-svc-7d8f9c-xk2p4 restarts=3 for 5m")

    # t+20: дежурный применяет fix (увеличение heap)
    add(20, "INFO",  "kubectl",      "deployment/payments-svc patched: resources.limits.memory=4Gi")
    add(21, "INFO",  "kubelet",      "payments-svc-7d8f9c-new99: Scheduled on node payments-node-3")
    add(22, "INFO",  "kubelet",      "payments-svc-7d8f9c-new99: Started container, phase=Running")
    add(23, "INFO",  "payments-svc", "Application started in 7.8s")

    # t+25: стабилизация
    for i in range(5):
        add(25 + i, "INFO", "api-gateway", "POST /api/v1/payment 200 OK 67ms")
    add(28, "INFO",  "prometheus",   "RESOLVED: PaymentsErrorRate normal (0.2%)")
    add(29, "INFO",  "prometheus",   "RESOLVED: PaymentsLatencyHigh p99=350ms")
    add(30, "INFO",  "order-svc",    "CircuitBreaker HALF_OPEN: testing payments-svc")
    add(31, "INFO",  "order-svc",    "CircuitBreaker CLOSED: payments-svc recovered")

    return rows


async def main() -> None:
    config = PipelineConfig(
        model=MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        incident_context=INCIDENT_CONTEXT,
        incident_start=datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
        incident_end=datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc),
        max_context_tokens=MAX_CONTEXT_TOKENS,
        model_supports_tool_calling=False,
    )

    rows = make_synthetic_rows()
    log.info("Synthetic rows: %d", len(rows))

    chunker = Chunker(
        max_batch_tokens=int(MAX_CONTEXT_TOKENS * 0.55),
        min_batch_lines=config.min_batch_lines,
    )
    chunks = chunker.chunk(rows)
    log.info("Chunks: %d", len(chunks))

    llm = LLMClient(
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        use_instructor=True,
        model_supports_tool_calling=False,
    )
    processor = MapProcessor(llm, chunker, config)

    results = await processor.process_all(chunks)

    print("\n" + "=" * 70)
    for i, r in enumerate(results):
        print(f"\n── BatchAnalysis #{i+1} ─────────────────────────────────────────────")
        print(f"  time_range : {r.time_range[0].isoformat()} → {r.time_range[1].isoformat()}")
        print(f"  events     : {len(r.events)}")
        print(f"  evidence   : {len(r.evidence)}")
        print(f"  hypotheses : {len(r.hypotheses)}")
        print(f"  anomalies  : {len(r.anomalies)}")
        print(f"  narrative  :\n    {r.narrative[:300]}")
        if r.hypotheses:
            print(f"  hypotheses[0]: [{r.hypotheses[0].confidence}] {r.hypotheses[0].title}")
    print("=" * 70)

    # Сохранить результат
    out = [r.model_dump(mode="json") for r in results]
    with open("artifacts/debug_map_result.json", "w", encoding="utf-8") as f:
        import pathlib; pathlib.Path("artifacts").mkdir(exist_ok=True)
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    log.info("Saved → artifacts/debug_map_result.json")


if __name__ == "__main__":
    import pathlib; pathlib.Path("artifacts").mkdir(exist_ok=True)
    asyncio.run(main())
