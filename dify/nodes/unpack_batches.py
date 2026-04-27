"""Dify Code Node: Unpack Batches

Dify не поддерживает индексы в переменных (batches[0]),
поэтому распаковываем массив батчей в именованные выходы.

Inputs:
  batches (Array[Object]) — выход split ноды [{rows, start, end}, ...]
Outputs:
  batch_1, batch_2, batch_3 (Object) — {rows, start, end}
"""
import json


_EMPTY = {"rows": [], "start": "", "end": ""}


def main(batches: list) -> dict:
    while len(batches) < 3:
        batches.append(_EMPTY)

    return {
        "batch_1": batches[0],
        "batch_2": batches[1],
        "batch_3": batches[2],
    }
