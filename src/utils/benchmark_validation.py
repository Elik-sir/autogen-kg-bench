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

    row_texts = []
    for row in result_rows:
        row_text = ", ".join(value_to_text(value) for value in row.values())
        if row_text:
            row_texts.append(row_text)
    return "; ".join(row_texts)
