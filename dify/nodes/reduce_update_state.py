"""Dify Code Node: Reduce — Update State

Дописывает merged в конец remaining, обновляет global items.

Inputs:
  remaining (Array[Object]) — из take_group
  merged    (Object)        — распаршенный результат мержа
Outputs:
  items (Array[Object]) — новый global items
  done  (Number)        — 1 если остался 1 элемент (финал)
"""


def main(remaining: list, merged: dict) -> dict:
    items = remaining + [merged]
    return {
        "items": items,
        "done":  1 if len(items) == 1 else 0,
    }
