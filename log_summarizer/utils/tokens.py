"""Оценка токенов и токеновые бюджеты.

Намеренно грубая оценка: точная токенизация потребовала бы загрузки
tiktoken / sentencepiece, что не нужно для бюджетирования.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов.

    len(text) // 3 — консервативная оценка для логов с JSON,
    трейсами, нелатинскими символами. Для чистого английского текста
    правильнее // 4, но логи плотнее.
    """
    return max(len(text) // 3, 0)


def chars_to_tokens(chars: int) -> int:
    """Конвертация символов в токены (та же формула)."""
    return max(chars // 3, 0)


def tokens_to_chars(tokens: int) -> int:
    """Обратная конвертация токенов в символы."""
    return tokens * 3


def fits_in_budget(text: str, budget_tokens: int) -> bool:
    """Проверяет, влезает ли текст в токеновый бюджет."""
    return estimate_tokens(text) <= budget_tokens


def trim_to_budget(text: str, budget_tokens: int) -> str:
    """Обрезает текст до бюджета (по символам, не по токенам)."""
    max_chars = tokens_to_chars(budget_tokens)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def trim_lines_to_budget(text: str, budget_tokens: int) -> str:
    """Обрезает многострочный текст до бюджета, не разрезая строки.

    В отличие от trim_to_budget, гарантирует целые строки по '\\n'.
    Если первая строка сама превышает бюджет — возвращает пустую строку.
    """
    max_chars = tokens_to_chars(budget_tokens)
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut <= 0:
        return ""
    return text[:cut]


def trim_rows_to_budget(rows: list[str], budget_tokens: int) -> list[str]:
    """Берёт максимальный префикс строк, влезающий в бюджет токенов.

    Никогда не обрезает строку на полуслове — берёт целые элементы.
    Строки, которые сами по себе превышают бюджет, пропускаются.
    """
    result: list[str] = []
    used = 0
    for row in rows:
        row_tokens = estimate_tokens(row)
        if used + row_tokens > budget_tokens:
            break
        result.append(row)
        used += row_tokens
    return result
