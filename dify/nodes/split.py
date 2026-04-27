"""Dify Code Node: Split

Берёт плоский список логов и возвращает один батч,
ограниченный токенным бюджетом и макс. числом строк.

Inputs:
  rows          (Array[Object]) — полный список логов (global переменная)
  offset        (Number)        — с какой строки начинать (global, начало = 0)
  token_budget  (str)           — токенов на батч (default "6000")
  max_batch     (str)           — макс. строк в батче (default "29")
Outputs:
  batch        (Array[String]) — строки батча (json-сериализованные)
  next_offset  (Number)
  has_more     (Number)        — 1 если ещё есть данные, 0 если конец
  batch_start  (String)
  batch_end    (String)
"""
import json


def main(rows: list, offset: int = 0, token_budget: str = "6000", max_batch: str = "29") -> dict:
    budget   = int(token_budget)
    max_rows = int(max_batch)
    idx      = int(offset)

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

    parsed      = [json.loads(s) for s in batch]
    batch_start = min((r.get("timestamp", "") for r in parsed), default="")
    batch_end   = max((r.get("end_time") or r.get("timestamp", "") for r in parsed), default="")

    return {
        "batch":       batch,
        "next_offset": idx,
        "has_more":    1 if idx < len(rows) else 0,
        "batch_start": batch_start,
        "batch_end":   batch_end,
    }
