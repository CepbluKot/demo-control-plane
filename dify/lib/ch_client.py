import json
import urllib.parse
import urllib.request


def ch_query(
    host: str,
    port: int,
    user: str,
    password: str,
    sql: str,
    timeout: int = 60,
) -> list[dict]:
    """Выполняет SQL в ClickHouse через HTTP-интерфейс, возвращает list[dict]."""
    encoded = urllib.parse.quote(sql + " FORMAT JSONEachRow")
    url = f"http://{host}:{port}/?query={encoded}"
    req = urllib.request.Request(url)
    req.add_header("X-ClickHouse-User", user)
    req.add_header("X-ClickHouse-Key", password)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        lines = resp.read().decode("utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
