from __future__ import annotations

import re
from difflib import SequenceMatcher


def normalize_question_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Убираем пунктуацию, чтобы "?" и "," не мешали дедупликации
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def is_near_duplicate_question(
    question: str, seen_normalized_questions: list[str], threshold: float = 0.92
) -> bool:
    normalized = normalize_question_text(question)
    if not normalized:
        return True
    return any(
        SequenceMatcher(None, normalized, seen).ratio() >= threshold
        for seen in seen_normalized_questions
    )
