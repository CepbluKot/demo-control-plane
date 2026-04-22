"""Dify Code Node: REDUCE

Copy-paste в Dify Code Node (Python).
Inputs:
  map_results   (Array[Object]) — вывод MAP Iteration ноды
  incident_info (str)
  period_start  (str)
  period_end    (str)
  llm_api_base  (str) — endpoint LLM (без /v1)
  llm_api_key   (str) — API ключ
  llm_model     (str) — название модели
  llm_timeout   (str) — таймаут в секундах (default "2400")
  group_size    (str) — размер группы REDUCE (default "4")
Outputs: merged_analysis (Object)
"""
import json
import time
import urllib.request


# ── LLM HTTP ──────────────────────────────────────────────────────────

def call_llm_http(api_base, api_key, model, system, user, timeout=2400):
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


# ── Helpers ───────────────────────────────────────────────────────────

def _is_server_down(err):
    t = str(err).lower()
    return "502" in t or "503" in t or "bad gateway" in t or "service unavailable" in t


def _is_timeout(err):
    t = str(err).lower()
    return "timeout" in t or "timed out" in t


def _group_items(items, group_size):
    return [items[i:i + group_size] for i in range(0, len(items), group_size)]


def _programmatic_merge(group):
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


def _build_user(items):
    parts = [f"=== Analysis {i+1} ===\n{json.dumps(item, ensure_ascii=False)}"
             for i, item in enumerate(items)]
    return "\n\n".join(parts)


def _build_system(incident_info, period_start, period_end):
    return (
        "You are a senior SRE synthesizing partial incident analyses into a unified view.\n\n"
        f"=== Incident context ===\n{incident_info}\n\n"
        f"Incident window: {period_start} \u2192 {period_end}\n\n"
        "=== Language ===\n"
        "Think and write all English fields in English. Keep technical terms as-is.\n"
        "For every field that has a _ru counterpart, also write the Russian translation.\n\n"
        "=== Task ===\n"
        "Merge the given JSON analyses into one unified MergedAnalysis. "
        "Deduplicate events, merge hypotheses, preserve all evidence. "
        "Output a single JSON object — no prose before or after.\n\n"
        "=== Output schema ===\n"
        '{"time_range":["<ISO>","<ISO>"],"narrative":"<str>","narrative_ru":"<str>",'
        '"events":[...],"causal_chains":[...],"hypotheses":[...],'
        '"evidence_bank":[...],"gaps":[...],"alert_refs":[],"zones_covered":[]}'
    )


def _merge_group(group, system, api_base, api_key, model, timeout, max_retries=5):
    current_timeout = timeout
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm_http(api_base, api_key, model, system, _build_user(group), current_timeout)
            return json.loads(raw)
        except Exception as exc:
            err = str(exc)
            if _is_timeout(err):
                current_timeout *= 2
            if _is_server_down(err):
                time.sleep(30)
            if attempt == max_retries:
                return _programmatic_merge(group)
    return _programmatic_merge(group)


# ── Main ──────────────────────────────────────────────────────────────

def main(
    map_results: list,
    incident_info: str,
    period_start: str,
    period_end: str,
    llm_api_base: str = "http://localhost:8000",
    llm_api_key: str = "none",
    llm_model: str = "default",
    llm_timeout: str = "2400",
    group_size: str = "4",
) -> dict:
    api_base   = llm_api_base
    api_key    = llm_api_key
    model      = llm_model
    timeout    = float(llm_timeout)
    group_size = int(group_size)

    items = list(map_results)

    if len(items) == 1:
        return {"merged_analysis": items[0]}

    system = _build_system(incident_info, period_start, period_end)

    while len(items) > 1:
        groups = _group_items(items, group_size)
        next_items = []
        for group in groups:
            if len(group) == 1:
                next_items.append(group[0])
            else:
                next_items.append(_merge_group(group, system, api_base, api_key, model, timeout))
        items = next_items

    return {"merged_analysis": items[0]}
