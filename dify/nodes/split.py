"""Dify Code Node: Split

Берёт плоский список логов и возвращает один батч по токенному бюджету,
сдвигая offset для следующей итерации Loop.

Inputs:
  rows          (Array[Object]) — полный список логов (global переменная)
  offset        (Number)        — с какой строки начинать (global, начало = 0)
  token_budget  (str)           — токенов на батч (default "6000")
Outputs:
  batch        (Array[Object]) — текущий батч
  next_offset  (Number)        — offset для следующей итерации
  has_more     (Number)        — 1 если ещё есть данные, 0 если конец
"""
import json


def main(rows: list, offset: int = 0, token_budget: str = "6000") -> dict:
    budget = int(token_budget)
    idx = int(offset)

    batch = []
    tokens = 0

    while idx < len(rows):
        row = rows[idx]
        t = len(json.dumps(row, ensure_ascii=False)) // 3
        if batch and tokens + t > budget:
            break
        batch.append(row)
        tokens += t
        idx += 1

    return {
        "batch":       batch,
        "next_offset": idx,
        "has_more":    1 if idx < len(rows) else 0,
    }
