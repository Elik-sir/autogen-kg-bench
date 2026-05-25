import re


_INSUFFICIENT_ANSWER_PATTERNS = (
    r"^no data for this query\.?$",
    r"^no (relevant )?data\.?$",
    r"^no information\.?$",
    r"^cannot determine\.?$",
    r"^can'?t determine\.?$",
    r"^insufficient data\.?$",
    r"^not enough (information|data)\.?$",
    r"^unable to (determine|answer)\.?$",
    r"^no matching records\.?$",
    r"^unknown\.?$",
    r"^n/?a\.?$",
    r"^нет данных\.?$",
    r"^недостаточно данных\.?$",
    r"^не удалось определить\.?$",
)

_INSUFFICIENT_ANSWER_PREFIXES = (
    "no data",
    "no information",
    "no relevant data",
    "cannot determine",
    "can't determine",
    "insufficient data",
    "unable to answer",
    "unable to determine",
    "not enough information",
    "not enough data",
    "there is no data",
    "there is no information",
    "i cannot determine",
    "i can't determine",
    "нет данных",
    "недостаточно данных",
)


def is_insufficient_answer(answer: str) -> bool:
    """True, если answer — отказ/заглушка без полезного эталонного ответа."""
    text = str(answer or "").strip()
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", text.lower()).strip().rstrip(".")
    for pattern in _INSUFFICIENT_ANSWER_PATTERNS:
        if re.match(pattern, normalized, flags=re.IGNORECASE):
            return True
    for prefix in _INSUFFICIENT_ANSWER_PREFIXES:
        if normalized.startswith(prefix):
            return True
    return False


def is_trivial_self_return(cypher_query):
    """
    Отсекает тривиальные запросы вида:
    WHERE x.prop = ... RETURN x.prop
    """
    normalized = " ".join(cypher_query.strip().split())
    where_match = re.search(r"WHERE\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=", normalized, re.IGNORECASE)
    return_match = re.search(r"RETURN\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", normalized, re.IGNORECASE)
    if not where_match or not return_match:
        return False
    return (
        where_match.group(1).lower() == return_match.group(1).lower()
        and where_match.group(2).lower() == return_match.group(2).lower()
    )


def value_to_text(value):
    if isinstance(value, dict):
        return ", ".join(f"{key}: {value_to_text(inner_value)}" for key, inner_value in value.items())
    if isinstance(value, list):
        return ", ".join(value_to_text(inner_value) for inner_value in value)
    return str(value)


def result_to_ground_truth(question, result_rows):
    lowered_question = question.strip().lower()
    if lowered_question.startswith("есть ли"):
        first_row = result_rows[0] if result_rows else {}
        first_value = next(iter(first_row.values()), None)
        if isinstance(first_value, bool):
            return "Да, есть." if first_value else "Нет, не найдено."
        if isinstance(first_value, (int, float)):
            return "Да, есть." if first_value > 0 else "Нет, не найдено."
        return "Да, есть."
    if lowered_question.startswith(
        ("is there ", "are there ", "does ", "do ", "did ", "was there ", "were there ")
    ):
        first_row = result_rows[0] if result_rows else {}
        first_value = next(iter(first_row.values()), None)
        if isinstance(first_value, bool):
            return "Yes." if first_value else "No matching records."
        if isinstance(first_value, (int, float)):
            return "Yes." if first_value > 0 else "No matching records."
        return "Yes."

    row_texts = []
    for row in result_rows:
        row_text = ", ".join(value_to_text(value) for value in row.values())
        if row_text:
            row_texts.append(row_text)
    return "; ".join(row_texts)
