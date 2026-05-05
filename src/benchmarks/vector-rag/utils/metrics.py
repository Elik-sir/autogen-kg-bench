from __future__ import annotations

import re


def _tokenize(s: str) -> set[str]:
    s = s.lower()
    return set(re.findall(r"[\w\.\-]+", s, re.UNICODE)) - {""}


def recall_overlap(ground_truth: str, answer: str) -> float:
    if not ground_truth.strip():
        return 1.0
    g = _tokenize(ground_truth)
    if not g:
        return 0.0
    a = _tokenize(answer) if answer else set()
    return len(g & a) / len(g)
