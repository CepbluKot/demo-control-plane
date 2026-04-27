"""Dify Code Node: Reduce — Take Group

Inputs:
  items        (Array[Object]) — global
  offset       (Number)        — global: текущая позиция в items
  token_budget (str)           — токенов на группу (default "6000")
  max_group    (str)           — макс. элементов в группе (default "29")
Outputs:
  group       (Array[Object]) — группа для мержа
  new_offset  (Number)        — следующий offset
  has_more    (Number)        — 1 если после группы ещё есть элементы
"""
import json


def main(items: list, offset: int = 0,
         token_budget: str = "6000", max_group: str = "29") -> dict:
    budget  = int(token_budget)
    max_els = int(max_group)
    idx     = int(offset)

    group  = []
    tokens = 0

    for item in items[idx:]:
        item_str = json.dumps(item, ensure_ascii=False)
        t = len(item_str) // 3
        if group and tokens + t > budget:
            break
        group.append(item)
        tokens += t
        if len(group) >= max_els:
            break

    new_offset = idx + len(group)
    return {
        "group":      group,
        "new_offset": new_offset,
        "has_more":   1 if new_offset < len(items) else 0,
    }
