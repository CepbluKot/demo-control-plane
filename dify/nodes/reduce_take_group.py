"""Dify Code Node: Reduce — Take Group

Берёт следующую группу элементов для мержа.

Inputs:
  items      (Array[Object]) — global: текущий список саммари
  group_idx  (Number)        — global: с какого элемента берём группу
  group_size (str)           — размер группы (default "4")
Outputs:
  group        (Array[Object]) — группа для мержа
  new_group_idx (Number)       — следующий group_idx
  pass_done    (Number)        — 1 если дошли до конца текущего прохода
"""


def main(items: list, group_idx: int = 0, group_size: str = "4") -> dict:
    gs    = int(group_size)
    idx   = int(group_idx)
    group = items[idx:idx + gs]

    return {
        "group":          group,
        "new_group_idx":  idx + len(group),
        "pass_done":      1 if idx + len(group) >= len(items) else 0,
    }
