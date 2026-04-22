def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов: ~4 символа на токен."""
    return max(1, len(text) // 4)


def chunk_rows(rows: list[str], token_budget: int = 6000) -> list[str]:
    """Нарезает строки логов на батчи по токеновому бюджету."""
    if not rows:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for row in rows:
        t = estimate_tokens(row)
        if current and current_tokens + t > token_budget:
            chunks.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(row)
        current_tokens += t
    if current:
        chunks.append("\n".join(current))
    return chunks
