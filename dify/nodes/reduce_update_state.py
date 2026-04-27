"""Dify Code Node: Reduce — Update State

Inputs:
  items      (Array[Object]) — global
  next_items (Array[Object]) — global: аккумулятор текущего прохода
  merged     (Object)        — распаршенный результат мержа
  new_offset (Number)        — из take_group
  has_more   (Number)        — из take_group
Outputs:
  items      (Array[Object]) — новый global items
  next_items (Array[Object]) — новый global next_items
  offset     (Number)        — новый global offset
  done       (Number)        — 1 если финальный результат готов
"""


def main(items: list, next_items: list, merged: dict,
         new_offset: int, has_more: int) -> dict:
    next_items = next_items + [merged]

    if has_more:
        # продолжаем текущий проход
        return {
            "items":      items,
            "next_items": next_items,
            "offset":     new_offset,
            "done":       0,
        }
    else:
        # проход завершён
        if len(next_items) == 1:
            return {
                "items":      next_items,
                "next_items": [],
                "offset":     0,
                "done":       1,
            }
        else:
            # начинаем следующий проход
            return {
                "items":      next_items,
                "next_items": [],
                "offset":     0,
                "done":       0,
            }
