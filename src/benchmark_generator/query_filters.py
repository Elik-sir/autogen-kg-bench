from __future__ import annotations

import re


def _is_zero_like_value(value) -> bool:
    return isinstance(value, (int, float)) and value == 0


def _row_is_empty_like(row: dict) -> bool:
    if not isinstance(row, dict) or not row:
        return True
    values = list(row.values())
    return all(value is None or _is_zero_like_value(value) for value in values)


def should_skip_low_signal_result(cypher_query: str, result: list[dict]) -> bool:
    """Отбрасывает low-signal агрегатные результаты вида None/0."""
    if not result:
        return True
    if not all(isinstance(row, dict) for row in result):
        return False

    has_aggregate = bool(
        re.search(r"\b(count|sum|avg|min|max)\s*\(", cypher_query or "", flags=re.IGNORECASE)
    )
    if has_aggregate and all(_row_is_empty_like(row) for row in result):
        return True
    return False


def append_limit_if_missing(cypher_query: str, row_limit: int) -> str:
    """Добавляет LIMIT к Cypher, если его нет, чтобы не раздувать ground_truth."""
    q = str(cypher_query or "").strip()
    if not q:
        return q
    if re.search(r"\blimit\s+\d+\b", q, flags=re.IGNORECASE):
        return q
    # UNION обычно требует отдельного LIMIT по подзапросам; не трогаем автоматически.
    if re.search(r"\bunion\b", q, flags=re.IGNORECASE):
        return q
    q = q.rstrip(";")
    return f"{q} LIMIT {row_limit}"
