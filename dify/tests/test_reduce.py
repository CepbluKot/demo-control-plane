import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.reduce import tree_reduce, _group_items


def _batch(idx):
    return {"time_range": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            "narrative": f"batch {idx}", "events": [], "hypotheses": [],
            "evidence": [], "gaps": [], "alert_refs": []}


def _merged(idx):
    return {"time_range": ["2024-01-01T00:00:00", "2024-01-01T02:00:00"],
            "narrative": f"merged {idx}", "narrative_ru": "",
            "events": [], "causal_chains": [], "hypotheses": [],
            "evidence_bank": [], "gaps": [], "alert_refs": [], "zones_covered": []}


def test_group_items():
    assert _group_items(list(range(9)), 4) == [[0,1,2,3],[4,5,6,7],[8]]


def test_single_item_no_llm():
    calls = {"n": 0}
    def fake(**_): calls["n"] += 1; return json.dumps(_merged(0))
    result = tree_reduce([_batch(0)], fake, "inc", "s", "e", group_size=4)
    assert calls["n"] == 0
    assert result == _batch(0)


def test_two_items_one_call():
    calls = {"n": 0}
    merged = _merged(0)
    def fake(**_): calls["n"] += 1; return json.dumps(merged)
    result = tree_reduce([_batch(0), _batch(1)], fake, "inc", "s", "e", group_size=4)
    assert calls["n"] == 1
    assert result == merged


def test_eight_items_three_calls():
    # 8 items, group_size=4 → round1: 2 calls → 2 items → round2: 1 call
    calls = {"n": 0}
    merged = _merged(0)
    def fake(**_): calls["n"] += 1; return json.dumps(merged)
    tree_reduce([_batch(i) for i in range(8)], fake, "inc", "s", "e", group_size=4)
    assert calls["n"] == 3
