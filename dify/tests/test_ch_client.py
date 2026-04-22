import json
import sys
import os
import urllib.request
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.ch_client import ch_query


def _mock_resp(rows):
    body = "\n".join(json.dumps(r) for r in rows).encode()
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = body
    return m


def test_parses_jsoneachrow():
    rows = [{"timestamp": "2024-01-01", "raw_line": "foo"}]
    with patch("urllib.request.urlopen", return_value=_mock_resp(rows)):
        assert ch_query("localhost", 8123, "u", "p", "SELECT 1") == rows


def test_skips_empty_lines():
    body = b'{"a": 1}\n\n{"a": 2}\n'
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = body
    with patch("urllib.request.urlopen", return_value=m):
        assert len(ch_query("localhost", 8123, "u", "p", "SELECT 1")) == 2


def test_raises_runtime_error_on_404():
    err = urllib.request.HTTPError(
        url="http://localhost:8123/",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )
    err.read = lambda: b"Table not found"
    with patch("urllib.request.urlopen", side_effect=err):
        try:
            ch_query("localhost", 8123, "u", "p", "SELECT 1")
            assert False, "должно бросить RuntimeError"
        except RuntimeError as e:
            assert "404" in str(e)
            assert "Table not found" in str(e)
