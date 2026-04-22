import json
import urllib.request


def ch_query(
    host: str,
    port: int,
    user: str,
    password: str,
    sql: str,
    timeout: int = 60,
) -> list[dict]:
    """POST-запрос к ClickHouse HTTP interface, возвращает list[dict]."""
    url = f"http://{host}:{port}/"
    body = (sql + " FORMAT JSONEachRow").encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("X-ClickHouse-User", user)
    req.add_header("X-ClickHouse-Key", password)
    req.add_header("Content-Type", "text/plain; charset=utf-8")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            lines = resp.read().decode("utf-8").splitlines()
            return [json.loads(line) for line in lines if line.strip()]
    except urllib.request.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ClickHouse HTTP {e.code} at {url!r}: {body_text[:500]}") from e
