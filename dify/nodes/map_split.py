"""Dify Code Node: Split (multi-batch)

Берёт плоский список логов и возвращает N батчей за один вызов —
по одному на каждый параллельный LLM-вызов в Dify.

Inputs:
  rows          (Array[Object]) — полный список логов (global переменная)
  offset        (Number)        — с какой строки начинать (global, начало = 0)
  token_budget  (str)           — токенов на один батч (default "6000")
  max_batch     (str)           — макс. строк в одном батче (default "29")
  n_parallel    (str)           — сколько батчей за раз (default "3")

Outputs:
  batch_1 .. batch_N  (Array[String]) — строки батча (json-сериализованные)
  next_offset         (Number)
  has_more            (Number)        — 1 если ещё есть данные, 0 если конец
  batch_start         (String)        — min timestamp по всем батчам итерации
  batch_end           (String)        — max timestamp по всем батчам итерации
"""

import json


def main(
    rows: list,
    offset: int = 0,
    token_budget: str = "6000",
    max_batch: str = "29",
    n_parallel: str = "3",
) -> dict:
    budget   = int(token_budget)
    max_rows = int(max_batch)
    n        = int(n_parallel)
    idx      = int(offset)

    result     = {}
    all_parsed = []

    for i in range(1, n + 1):
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

        result[f"batch_{i}"] = batch
        all_parsed.extend(json.loads(s) for s in batch)

    batch_start = min((r.get("timestamp", "") for r in all_parsed), default="")
    batch_end   = max((r.get("end_time") or r.get("timestamp", "") for r in all_parsed), default="")

    result["next_offset"] = idx
    result["has_more"]    = 1 if idx < len(rows) else 0
    result["batch_start"] = batch_start
    result["batch_end"]   = batch_end

    return result
