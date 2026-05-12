import re


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


def _clean_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_opaque_id(value):
    text = _clean_text(value)
    if not text:
        return False
    if re.fullmatch(r"\d{4,}", text):
        return True
    if re.fullmatch(r"[a-f0-9-]{24,}", text, flags=re.IGNORECASE):
        return True
    return False


def _row_to_context_text(row):
    if not isinstance(row, dict):
        return ""

    title = _clean_text(row.get("target_title") or row.get("title") or row.get("name") or row.get("target_value"))
    context = _clean_text(
        row.get("target_context")
        or row.get("summary")
        or row.get("description")
        or row.get("content")
        or row.get("text")
    )
    if context and title:
        return f"title: {title}; context: {context}"
    if context:
        return f"context: {context}"
    if title and not _looks_like_opaque_id(title):
        return title

    row_text = ", ".join(value_to_text(value) for value in row.values())
    return _clean_text(row_text)


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
        row_text = _row_to_context_text(row)
        if row_text:
            row_texts.append(row_text)
    return "; ".join(row_texts)
