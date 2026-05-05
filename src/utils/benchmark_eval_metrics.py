"""
Метрики retrieval / faithfulness по строкам бенчмарка (без LLM).

Используются скриптами eval у vector-rag, light-rag и др.: поля ``question``,
``answer``, ``contexts``; для ``context_recall`` — тот же эталон, что и в прогоне
(``ground_truth`` или ``ideal_for_scoring`` при ``scoring_reference`` = ``answer``).
"""

from __future__ import annotations

import re
from statistics import mean

WORD_RE = re.compile(r"[\w\.\-]+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")

STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "и",
    "в",
    "во",
    "на",
    "по",
    "с",
    "со",
    "что",
    "это",
    "как",
    "к",
    "из",
    "у",
    "за",
    "для",
    "не",
    "но",
}

REFUSAL_PHRASES: tuple[str, ...] = (
    "по контексту ответ дать нельзя",
    "cannot be answered",
    "cannot answer",
    "insufficient context",
    "not enough context",
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = {
        token
        for token in WORD_RE.findall(text.lower())
        if token and token not in STOPWORDS and any(ch.isalnum() for ch in token)
    }
    return tokens


def _f1(overlap: int, left_size: int, right_size: int) -> float:
    precision = _safe_ratio(overlap, left_size)
    recall = _safe_ratio(overlap, right_size)
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _contexts_text(contexts: list[str]) -> str:
    return "\n".join(str(x) for x in contexts if x is not None)


def reference_text_for_metrics(item: dict) -> str:
    """Эталон для context_recall в духе recall в main (subgraph-deep → ideal_for_scoring)."""
    if item.get("scoring_reference") == "answer":
        ideal = item.get("ideal_for_scoring")
        return str(ideal).strip() if ideal is not None else ""
    gt = item.get("ground_truth")
    return str(gt).strip() if gt is not None else ""


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    gt_tokens = tokenize(ground_truth)
    if not ground_truth.strip():
        return 1.0
    if not gt_tokens:
        return 0.0
    ctx_tokens = tokenize(_contexts_text(contexts))
    return _safe_ratio(len(gt_tokens & ctx_tokens), len(gt_tokens))


def context_relevance(question: str, contexts: list[str]) -> float:
    q_tokens = tokenize(question)
    if not q_tokens or not contexts:
        return 0.0

    chunk_scores: list[float] = []
    for chunk in contexts:
        chunk_tokens = tokenize(str(chunk))
        if not chunk_tokens:
            continue
        overlap = len(q_tokens & chunk_tokens)
        chunk_scores.append(_f1(overlap, len(chunk_tokens), len(q_tokens)))

    if not chunk_scores:
        return 0.0
    top_k = sorted(chunk_scores, reverse=True)[:3]
    return mean(top_k)


def faithfulness(answer: str, contexts: list[str]) -> float:
    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0
    ctx_tokens = tokenize(_contexts_text(contexts))
    return _safe_ratio(len(answer_tokens & ctx_tokens), len(answer_tokens))


def answer_relevancy(question: str, answer: str) -> float:
    if not answer.strip():
        return 0.0
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in REFUSAL_PHRASES):
        return 0.0

    q_tokens = tokenize(question)
    a_tokens = tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens)
    return _f1(overlap, len(a_tokens), len(q_tokens))


def evidence_coverage(answer: str, contexts: list[str]) -> float:
    ctx_tokens = tokenize(_contexts_text(contexts))
    if not ctx_tokens:
        return 0.0

    raw_sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(answer) if s.strip()]
    sentence_tokens = [tokenize(sentence) for sentence in raw_sentences]
    sentence_tokens = [tokens for tokens in sentence_tokens if tokens]
    if not sentence_tokens:
        return 0.0

    supported = 0
    for tokens in sentence_tokens:
        overlap = len(tokens & ctx_tokens)
        ratio = _safe_ratio(overlap, len(tokens))
        if overlap >= 2 and ratio >= 0.3:
            supported += 1
    return _safe_ratio(supported, len(sentence_tokens))


def compute_item_metrics(item: dict) -> dict[str, float]:
    question = str(item.get("question", "") or "")
    reference = reference_text_for_metrics(item)
    answer = str(item.get("answer", "") or "")
    contexts = item.get("contexts") or []
    contexts_list = [str(x) for x in contexts]

    return {
        "context_recall": round(context_recall(reference, contexts_list), 4),
        "context_relevance": round(context_relevance(question, contexts_list), 4),
        "faithfulness": round(faithfulness(answer, contexts_list), 4),
        "answer_relevancy": round(answer_relevancy(question, answer), 4),
        "evidence_coverage": round(evidence_coverage(answer, contexts_list), 4),
    }


def aggregate_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {
            "context_recall": 0.0,
            "context_relevance": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "evidence_coverage": 0.0,
        }
    keys = metric_rows[0].keys()
    return {k: round(mean(row[k] for row in metric_rows), 4) for k in keys}
