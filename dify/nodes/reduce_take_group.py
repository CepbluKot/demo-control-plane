"""Dify Code Node: Reduce — Take Group

Берёт следующую группу элементов для мержа,
ограниченную токенным бюджетом и макс. числом элементов.

Inputs:
  items        (Array[Object]) — global: текущий список саммари
  group_idx    (Number)        — global: с какого элемента берём группу
  token_budget (str)           — токенов на группу (default "6000")
  max_group    (str)           — макс. элементов в группе (default "29")
Outputs:
  group         (Array[Object]) — группа для мержа
  new_group_idx (Number)        — следующий group_idx
  pass_done     (Number)        — 1 если дошли до конца текущего прохода
"""
import json


def main(items: list, group_idx: int = 0,
         token_budget: str = "6000", max_group: str = "29") -> dict:
    budget   = int(token_budget)
    max_els  = int(max_group)
    idx      = int(group_idx)

    group  = []
    tokens = 0

    while idx < len(items) and len(group) < max_els:
        item_str = json.dumps(items[idx], ensure_ascii=False)
        t = len(item_str) // 3
        if group and tokens + t > budget:
            break
        group.append(items[idx])
        tokens += t
        idx += 1

    return {
        "group":          group,
        "new_group_idx":  idx,
        "pass_done":      1 if idx >= len(items) else 0,
    }
