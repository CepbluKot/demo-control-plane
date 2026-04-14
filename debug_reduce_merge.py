"""
debug_reduce_merge.py — тест слияния двух больших батчей через _reduce_structured_group.

Сценарий: каждый батч влезает в контекст модели, но вместе — нет.
Проверяем что split-fallback → compress → retry работает корректно.

Запуск:
    python debug_reduce_merge.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

LLM_API_BASE = "http://localhost:8000/v1"
LLM_API_KEY  = "sk-placeholder"
LLM_MODEL    = "PNX.QWEN3 235b a22b instruct"

# Файлы с батчами — кладём рядом с этим скриптом
BATCH_A = Path("batch_a.txt")
BATCH_B = Path("batch_b.txt")

OUTPUT_DIR = Path("artifacts/debug_reduce_merge")

# Контекстное окно модели в символах (токены × 4).
# Qwen3 235B, 100K ctx → 400_000
CONTEXT_WINDOW_CHARS = 400_000

# ═══════════════════════════════════════════════════════════════
#  КОНЕЦ CONFIG
# ═══════════════════════════════════════════════════════════════

from settings import settings
settings.OPENAI_API_BASE_DB = LLM_API_BASE
settings.OPENAI_API_KEY_DB  = LLM_API_KEY
settings.LLM_MODEL_ID       = LLM_MODEL

from schemas import IncidentSummary
from my_summarizer import PeriodLogSummarizer, SummarizerConfig, _make_llm_call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
for noisy in ("httpcore", "httpx", "openai", "instructor", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("debug_reduce_merge")


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", "_")


def load_batch(path: Path) -> IncidentSummary:
    log.info("Загружаю %s ...", path.resolve())
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    batch = IncidentSummary.model_validate(data)
    log.info("  OK: %d events, %d hypotheses, %d causal_links",
             len(batch.timeline), len(batch.hypotheses), len(batch.causal_links))
    return batch


def build_summarizer(reduce_input_max_chars: int) -> PeriodLogSummarizer:
    def _on_retry(attempt: int, total: int, exc: Exception) -> None:
        log.warning("LLM retry | attempt=%s | error=%s", attempt, exc)

    def _on_attempt(attempt: int, total: int, timeout_seconds: float) -> None:
        log.info("LLM attempt=%s | timeout=%.0fs", attempt, timeout_seconds)

    def _on_result(attempt: int, total: int, success: bool, elapsed: float, err: Any) -> None:
        log.info("LLM result | success=%s | elapsed=%.2fs | err=%s", success, elapsed, err or "")

    llm_call = _make_llm_call(
        max_retries=3,
        on_retry=_on_retry,
        on_attempt=_on_attempt,
        on_result=_on_result,
        llm_timeout=600.0,
        fail_open_return_empty=False,
    )

    progress_events: List[Dict[str, Any]] = []

    def on_progress(event: str, payload: Dict[str, Any]) -> None:
        progress_events.append({"event": event, **payload})
        log.info("progress | %s | %s", event,
                 {k: v for k, v in payload.items() if k not in ("data",)})

    return PeriodLogSummarizer(
        db_fetch_page=lambda **_: [],
        llm_call=llm_call,
        config=SummarizerConfig(
            reduce_input_max_chars=reduce_input_max_chars,
            reduce_group_size=2,
            use_instructor=True,
            model_supports_tool_calling=True,
            compression_target_pct=50,
            compression_importance_threshold=0.7,
        ),
        on_progress=on_progress,
        prompt_context={},
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Загрузка и валидация батчей ───────────────────────────────
    batch_a = load_batch(BATCH_A)
    batch_b = load_batch(BATCH_B)

    # Размерный анализ (без LLM)
    _sizer = PeriodLogSummarizer(
        db_fetch_page=lambda **_: [],
        llm_call=lambda _: "",
        config=SummarizerConfig(),
        on_progress=lambda e, p: None,
        prompt_context={},
    )
    size_a = _sizer._summary_chars(batch_a)
    size_b = _sizer._summary_chars(batch_b)
    size_combined = size_a + size_b

    sep = "═" * 70
    log.info(sep)
    log.info("РАЗМЕРЫ БАТЧЕЙ")
    log.info("  batch_a  : %s chars | %d events", _fmt(size_a), len(batch_a.timeline))
    log.info("  batch_b  : %s chars | %d events", _fmt(size_b), len(batch_b.timeline))
    log.info("  combined : %s chars", _fmt(size_combined))
    log.info("  context  : %s chars", _fmt(CONTEXT_WINDOW_CHARS))
    log.info("  batch_a alone : %s", "✓ fits" if size_a < CONTEXT_WINDOW_CHARS else "✗ too large")
    log.info("  batch_b alone : %s", "✓ fits" if size_b < CONTEXT_WINDOW_CHARS else "✗ too large")
    log.info("  combined      : %s", "✗ OVERFLOW (ожидаемо)" if size_combined > CONTEXT_WINDOW_CHARS else "✓ fits — увеличь батчи")
    log.info(sep)

    # ── Запуск слияния ────────────────────────────────────────────
    # reduce_input_max_chars = размер большего батча (это лимит per-item перед L1)
    reduce_input_max_chars = max(size_a, size_b)

    period_start = batch_a.context.time_range_start
    period_end   = batch_b.context.time_range_end

    summarizer = build_summarizer(reduce_input_max_chars)

    log.info("ЗАПУСКАЮ _reduce_structured_group([batch_a, batch_b])")
    log.info("Ожидаемое поведение:")
    log.info("  1. LLM вызов с обоими → context overflow (400 / error)")
    log.info("  2. split: batch_a и batch_b обрабатываются по отдельности (passthrough, 1 item)")
    log.info("  3. _compress_summary_on_overflow для каждого")
    log.info("  4. retry merge сжатых → success (или ещё один split+compress)")
    log.info(sep)

    t0 = time.time()
    try:
        result, llm_calls = summarizer._reduce_structured_group(
            summaries=[batch_a, batch_b],
            period_start=period_start,
            period_end=period_end,
            reduce_round=1,
            group_index=1,
            group_total=1,
            depth=0,
        )
        elapsed = time.time() - t0
        size_result = summarizer._summary_chars(result)
        compression = (1.0 - size_result / size_combined) * 100

        log.info(sep)
        log.info("РЕЗУЛЬТАТ: SUCCESS")
        log.info("  LLM вызовов     : %d", llm_calls)
        log.info("  Время           : %.1fs", elapsed)
        log.info("  Входной размер  : %s chars", _fmt(size_combined))
        log.info("  Выходной размер : %s chars", _fmt(size_result))
        log.info("  Сжатие          : %.1f%%", compression)
        log.info("  Timeline events : %d", len(result.timeline))
        log.info("  Hypotheses      : %d", len(result.hypotheses))
        log.info(sep)

        out = OUTPUT_DIR / "merge_result.json"
        out.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Результат: %s", out.resolve())

    except Exception as exc:
        log.error(sep)
        log.error("РЕЗУЛЬТАТ: ОШИБКА после %.1fs", time.time() - t0)
        log.error("  %s: %s", type(exc).__name__, exc)
        log.error(sep)
        raise


if __name__ == "__main__":
    main()
