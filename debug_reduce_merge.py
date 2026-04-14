"""
debug_reduce_merge.py — тест REDUCE-фазы (TreeReducer) на данных из файлов.

Загружает BatchAnalysis-объекты из директории BATCHES_DIR (JSON-файлы),
прогоняет через TreeReducer, печатает итоговый MergedAnalysis.

Как подготовить данные — см. промпт в GENERATE_BATCHES_PROMPT.md
(или запусти: python debug_reduce_merge.py --print-prompt).

Запуск:
    python debug_reduce_merge.py
    python debug_reduce_merge.py --batches-dir artifacts/my_batches
    python debug_reduce_merge.py --print-prompt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone

# ── CONFIG ────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000"
API_KEY    = "sk-placeholder"
MODEL      = "PNX.QWEN3 235b a22b instruct"

# Директория с файлами batch_000.json, batch_001.json, ...
BATCHES_DIR = pathlib.Path("artifacts/test_batches")

MAX_CONTEXT_TOKENS      = 100_000
MAX_GROUP_SIZE          = 3
MAX_EVENTS_PER_MERGE    = 30
COMPRESSION_TARGET_PCT  = 50
# ─────────────────────────────────────────────────────────────────────

from log_summarizer.config import PipelineConfig
from log_summarizer.llm_client import LLMClient
from log_summarizer.models import BatchAnalysis
from log_summarizer.tree_reducer import TreeReducer
from log_summarizer.utils.logging import setup_pipeline_logging

setup_pipeline_logging("INFO")
log = logging.getLogger("debug_reduce_merge")


# ── Промпт для генерации тестовых батчей ──────────────────────────────

GENERATE_BATCHES_PROMPT = """\
Сгенерируй N файлов с тестовыми данными для отладки REDUCE-фазы LLM-пайплайна анализа инцидентов.
Каждый файл — JSON-сериализация одного BatchAnalysis (результат MAP-фазы).
Все файлы описывают один и тот же реальный инцидент, но разные временные окна (~5 минут каждое).

=== СЦЕНАРИЙ ИНЦИДЕНТА ===
payments-service (Java, k8s) начал падать из-за OOM в 14:30 UTC.
Каскад: api-gateway получил ошибки → order-service открыл circuit breaker →
oncall вмешался, увеличил memory limit → сервис восстановился к 15:10.

=== ФОРМАТ КАЖДОГО ФАЙЛА ===
Имена файлов: batch_000.json, batch_001.json, ..., batch_00N.json
Каждый файл — валидный JSON следующей схемы:

{
  "time_range": ["<ISO8601 UTC>", "<ISO8601 UTC>"],
  "narrative": "<3-5 предложений что происходило в этом окне>",
  "events": [
    {
      "id": "<evt-NNN-MMM>",
      "timestamp": "<ISO8601 UTC>",
      "source": "<service/pod/component>",
      "description": "<краткое описание ≤80 символов>",
      "severity": "critical|high|medium|low|info",
      "tags": ["oom", "timeout", "connection", "crash", "recovery", ...]
    }
  ],
  "evidence": [
    {
      "id": "<ev-NNN-MMM>",
      "timestamp": "<ISO8601 UTC>",
      "source": "<service>",
      "raw_line": "<дословная строка лога>",
      "severity": "critical|high|medium|low|info",
      "linked_event_id": "<evt-id или null>"
    }
  ],
  "hypotheses": [
    {
      "id": "<hyp-NNN-MMM>",
      "title": "<заголовок ≤60 символов>",
      "description": "<что утверждает гипотеза и почему>",
      "confidence": "low|medium|high",
      "supporting_event_ids": ["<evt-id>"],
      "contradicting_event_ids": []
    }
  ],
  "anomalies": [
    {
      "description": "<что необычного>",
      "related_event_ids": ["<evt-id>"]
    }
  ],
  "metrics_context": "<что показывали метрики в этом окне, или null>",
  "data_quality": null
}

=== ТРЕБОВАНИЯ ===
- Сгенерируй ровно 8 файлов (batch_000.json … batch_007.json)
- Временные окна: 2024-01-15T14:00Z → 14:05, 14:05 → 14:10, ..., 14:35 → 14:40
- В каждом файле 2-5 events и 1-3 evidence
- IDs должны быть уникальными ГЛОБАЛЬНО (evt-000-001, evt-001-001, ...)
- supporting_event_ids в гипотезах должны ссылаться на реальные events.id из того же файла
- raw_line в evidence должны выглядеть как настоящие строки из k8s/java логов
- Гипотезы должны ЭВОЛЮЦИОНИРОВАТЬ: в первых файлах — неопределённость, в последних — уверенность
- В batch_006 и batch_007 должна быть хотя бы одна гипотеза с confidence=high, подтверждающая OOM как root cause
- metrics_context заполни для batch_002 и batch_005 (остальные null)
- Не добавляй markdown-обёртку вокруг JSON — только сам JSON

Выдай каждый файл отдельным блоком, начиная со строки: === FILE: batch_NNN.json ===
"""


# ── Загрузка батчей ───────────────────────────────────────────────────

def load_batches(batches_dir: pathlib.Path) -> list[BatchAnalysis]:
    if not batches_dir.exists():
        log.error("Директория не найдена: %s", batches_dir.resolve())
        log.error("Запусти: python debug_reduce_merge.py --print-prompt")
        log.error("Скопируй промпт → нейронка → сохрани файлы в %s/", batches_dir)
        sys.exit(1)

    files = sorted(batches_dir.glob("batch_*.json"))
    if not files:
        log.error("В %s нет файлов batch_*.json", batches_dir.resolve())
        sys.exit(1)

    batches: list[BatchAnalysis] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            batch = BatchAnalysis.model_validate(data)
            batches.append(batch)
            log.info(
                "Loaded %-25s  events=%d  evidence=%d  hypotheses=%d",
                f.name,
                len(batch.events),
                len(batch.evidence),
                len(batch.hypotheses),
            )
        except Exception as exc:
            log.error("Ошибка при загрузке %s: %s", f.name, exc)
            sys.exit(1)

    return batches


# ── Main ──────────────────────────────────────────────────────────────

async def run(batches_dir: pathlib.Path) -> None:
    batches = load_batches(batches_dir)

    log.info(
        "Итого батчей: %d | events: %d | evidence: %d",
        len(batches),
        sum(len(b.events) for b in batches),
        sum(len(b.evidence) for b in batches),
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

    out_path = pathlib.Path("artifacts/debug_reduce_result.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    log.info("Saved → %s", out_path.resolve())


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


if __name__ == "__main__":
    main()
