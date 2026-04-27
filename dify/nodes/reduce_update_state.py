"""Dify Code Node: Reduce — Update State

Добавляет смёрженный результат в next_items.
Если проход завершён — переключает на следующий проход или выставляет done.

Inputs:
  items         (Array[Object]) — global: текущие items
  next_items    (Array[Object]) — global: аккумулятор текущего прохода
  merged        (Object)        — результат мержа текущей группы
  new_group_idx (Number)        — из take_group
  pass_done     (Number)        — из take_group
Outputs:
  items      (Array[Object]) — новое значение global items
  next_items (Array[Object]) — новое значение global next_items
  group_idx  (Number)        — новое значение global group_idx
  done       (Number)        — 1 если финальный результат готов
"""


def main(items: list, next_items: list, merged: dict,
         new_group_idx: int, pass_done: int) -> dict:
    next_items = next_items + [merged]

    if pass_done:
        if len(next_items) == 1:
            # всё смёржено — готово
            return {
                "items":      next_items,
                "next_items": [],
                "group_idx":  0,
                "done":       1,
            }
        else:
            # начинаем следующий проход
            return {
                "items":      next_items,
                "next_items": [],
                "group_idx":  0,
                "done":       0,
            }
    else:
        return {
            "items":      items,
            "next_items": next_items,
            "group_idx":  new_group_idx,
            "done":       0,
        }
