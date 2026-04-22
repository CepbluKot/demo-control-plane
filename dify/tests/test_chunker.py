import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.chunker import chunk_rows, estimate_tokens


def test_estimate_tokens():
    assert estimate_tokens("a" * 400) == 100


def test_single_chunk_when_small():
    chunks = chunk_rows(["line1", "line2", "line3"], token_budget=1000)
    assert len(chunks) == 1
    assert "line1" in chunks[0] and "line3" in chunks[0]


def test_splits_when_over_budget():
    big = "x" * 1000  # ~250 tokens
    chunks = chunk_rows([big, big, big], token_budget=300)
    assert len(chunks) == 3


def test_empty():
    assert chunk_rows([], token_budget=1000) == []


def test_oversized_single_row():
    big = "x" * 10000
    chunks = chunk_rows([big], token_budget=100)
    assert len(chunks) == 1
