"""Dify Code Node: Unpack Batches

Dify не поддерживает индексы в переменных (batches[0]),
поэтому распаковываем массив батчей в именованные плоские выходы.

Inputs:
  batches (Array[Object]) — выход split ноды [{rows, start, end}, ...]
Outputs:
  batch_1 (Array[String]), batch_1_start (String), batch_1_end (String)
  batch_2 (Array[String]), batch_2_start (String), batch_2_end (String)
  batch_3 (Array[String]), batch_3_start (String), batch_3_end (String)
"""


_EMPTY = {"rows": [], "start": "", "end": ""}


def main(batches: list) -> dict:
    while len(batches) < 3:
        batches.append(_EMPTY)

    return {
        "batch_1":       batches[0]["rows"],
        "batch_1_start": batches[0]["start"],
        "batch_1_end":   batches[0]["end"],
        "batch_2":       batches[1]["rows"],
        "batch_2_start": batches[1]["start"],
        "batch_2_end":   batches[1]["end"],
        "batch_3":       batches[2]["rows"],
        "batch_3_start": batches[2]["start"],
        "batch_3_end":   batches[2]["end"],
    }
