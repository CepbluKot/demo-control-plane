"""Dify Code Node: Reduce — Take Group

Берёт первую группу из items (по токенному бюджету + макс. элементов).

Inputs:
  items        (Array[Object]) — global: текущий список саммари
  token_budget (str)           — токенов на группу (default "6000")
  max_group    (str)           — макс. элементов в группе (default "29")
Outputs:
  group     (Array[Object]) — группа для мержа
  remaining (Array[Object]) — остаток items после группы
"""
import json


def main(items: list, token_budget: str = "6000", max_group: str = "29") -> dict:
    budget  = int(token_budget)
    max_els = int(max_group)

    group  = []
    tokens = 0

    for item in items:
        item_str = json.dumps(item, ensure_ascii=False)
        t = len(item_str) // 3
        if group and tokens + t > budget:
            break
        group.append(item)
        tokens += t
        if len(group) >= max_els:
            break

    return {
        "group":     group,
        "remaining": items[len(group):],
    }
