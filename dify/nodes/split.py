"""Dify Code Node: Split

Берёт плоский список логов и возвращает N батчей за раз (для параллельного
запуска N LLM-нод). Каждый батч ограничен токенным бюджетом и макс. числом строк.

Inputs:
  rows          (Array[Object]) — полный список логов (global переменная)
  offset        (Number)        — с какой строки начинать (global, начало = 0)
  token_budget  (str)           — токенов на один батч (default "6000")
  max_batch     (str)           — макс. строк в одном батче (default "29")
  n_parallel    (str)           — сколько батчей вернуть за раз (default "3")
Outputs:
  batches     (Array[Object]) — N объектов вида {rows, start, end}
  next_offset (Number)
  has_more    (Number)        — 1 если ещё есть данные, 0 если конец
"""
import json


def _take_batch(rows, idx, budget, max_rows):
    """Берёт один батч начиная с idx. Возвращает (batch_strings, new_idx)."""
    batch  = []
    tokens = 0
    while idx < len(rows) and len(batch) < max_rows:
        row_str = json.dumps(rows[idx], ensure_ascii=False)
        t = len(row_str) // 3
        if batch and tokens + t > budget:
            break
        batch.append(row_str)
        tokens += t
        idx += 1
    return batch, idx


def main(rows: list, offset: int = 0, token_budget: str = "6000",
         max_batch: str = "29", n_parallel: str = "3") -> dict:
    budget     = int(token_budget)
    max_rows   = int(max_batch)
    n          = int(n_parallel)
    idx        = int(offset)

    batches = []

    for _ in range(n):
        if idx >= len(rows):
            batches.append({"rows": [], "start": "", "end": ""})
            continue
        batch, idx = _take_batch(rows, idx, budget, max_rows)
        if not batch:
            batches.append({"rows": [], "start": "", "end": ""})
            continue
        parsed = [json.loads(s) for s in batch]
        batches.append({
            "rows":  batch,
            "start": min(r.get("timestamp", "") for r in parsed),
            "end":   max(r.get("end_time") or r.get("timestamp", "") for r in parsed),
        })

    return {
        "batches":     batches,
        "next_offset": idx,
        "has_more":    1 if idx < len(rows) else 0,
    }
