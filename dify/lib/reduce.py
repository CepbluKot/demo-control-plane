"""Логика REDUCE-цикла. tree_reduce() принимает list[dict] и сворачивает до одного."""
import json
import time
from typing import Callable


def _group_items(items: list, group_size: int) -> list[list]:
    return [items[i:i + group_size] for i in range(0, len(items), group_size)]


def _is_server_down(err: str) -> bool:
    t = err.lower()
    return "502" in t or "503" in t or "bad gateway" in t or "service unavailable" in t


def _is_timeout(err: str) -> bool:
    t = err.lower()
    return "timeout" in t or "timed out" in t


def _programmatic_merge(group: list[dict]) -> dict:
    return {
        "time_range": [
            min(item.get("time_range", [""])[0] for item in group),
            max(item.get("time_range", ["", ""])[1] for item in group),
        ],
        "narrative": " | ".join(item.get("narrative", "") for item in group),
        "narrative_ru": " | ".join(item.get("narrative_ru", "") for item in group),
        "events": [e for item in group for e in item.get("events", [])],
        "causal_chains": [c for item in group for c in item.get("causal_chains", [])],
        "hypotheses": [h for item in group for h in item.get("hypotheses", [])],
        "evidence_bank": [e for item in group for e in item.get("evidence_bank", item.get("evidence", []))],
        "gaps": list({g for item in group for g in item.get("gaps", [])}),
        "alert_refs": [],
        "zones_covered": [],
    }


def _build_user(items: list[dict]) -> str:
    parts = [f"=== Analysis {i+1} ===\n{json.dumps(item, ensure_ascii=False)}"
             for i, item in enumerate(items)]
    return "\n\n".join(parts)


def _merge_one_group(
    group: list[dict],
    call_llm_fn: Callable,
    system: str,
    max_retries: int = 5,
) -> dict:
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm_fn(system=system, user=_build_user(group))
            return json.loads(raw)
        except Exception as exc:
            err = str(exc)
            if _is_timeout(err):
                pass  # вызывающая сторона управляет таймаутом
            if _is_server_down(err):
                time.sleep(30)
            if attempt == max_retries:
                return _programmatic_merge(group)
    return _programmatic_merge(group)


def tree_reduce(
    items: list[dict],
    call_llm_fn: Callable,
    incident_info: str,
    period_start: str,
    period_end: str,
    group_size: int = 4,
    max_retries: int = 5,
) -> dict:
    """Итеративно мержит items до одного MergedAnalysis."""
    if len(items) == 1:
        return items[0]

    system = (
        "You are a senior SRE synthesizing partial incident analyses into a unified view.\n\n"
        f"=== Incident context ===\n{incident_info}\n\n"
        f"Incident window: {period_start} → {period_end}\n\n"
        "Merge the given JSON analyses into one unified MergedAnalysis. "
        "Deduplicate events, merge hypotheses, preserve all evidence. "
        "Output a single JSON object — no prose before or after.\n\n"
        "Output schema:\n"
        '{"time_range":["<ISO>","<ISO>"],"narrative":"<str>","narrative_ru":"<str>",'
        '"events":[...],"causal_chains":[...],"hypotheses":[...],'
        '"evidence_bank":[...],"gaps":[...],"alert_refs":[],"zones_covered":[]}'
    )

    while len(items) > 1:
        groups = _group_items(items, group_size)
        next_items = []
        for group in groups:
            if len(group) == 1:
                next_items.append(group[0])
            else:
                next_items.append(_merge_one_group(group, call_llm_fn, system, max_retries))
        items = next_items

    return items[0]
