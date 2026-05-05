from __future__ import annotations

import json
from typing import Any


def extract_contexts_from_rag_qa(raw: Any, qa_top_k: int) -> list[str]:
    """
    Чанки, которые HippoRAG реально подставляет в QA-промпт: ``docs[:qa_top_k]``
    (см. ``HippoRAG.qa``). То же поле удобно передавать в RAGAS как ``contexts``
    для Faithfulness.
    """
    if raw is None or not isinstance(raw, tuple) or not raw:
        return []
    solutions = raw[0]
    if not isinstance(solutions, list) or not solutions:
        return []
    sol0 = solutions[0]
    docs = getattr(sol0, "docs", None)
    if not isinstance(docs, list):
        return []
    k = max(0, int(qa_top_k))
    return [str(d) for d in docs[:k]]


def extract_answer_from_result(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, tuple) and raw:
        solutions = raw[0]
        messages = raw[1] if len(raw) > 1 else None
        if isinstance(solutions, list) and solutions:
            sol0 = solutions[0]
            if hasattr(sol0, "answer"):
                ans = getattr(sol0, "answer", None)
                if ans is not None and str(ans).strip():
                    return str(ans).strip()
        if isinstance(messages, list) and messages:
            msg = messages[0]
            if isinstance(msg, str) and msg.strip():
                parts = msg.split("Answer:", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return msg.strip()
        return ""
    if isinstance(raw, list):
        if not raw:
            return ""
        first = raw[0]
        if hasattr(first, "answer"):
            ans = getattr(first, "answer", None)
            if ans is not None and str(ans).strip():
                return str(ans).strip()
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("answer", "response", "generated_text", "text", "prediction"):
                val = first.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
            return json.dumps(first, ensure_ascii=False)
        return str(first).strip()
    if isinstance(raw, dict):
        for key in ("answer", "response", "generated_text", "text", "prediction"):
            val = raw.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return json.dumps(raw, ensure_ascii=False)
    return str(raw).strip()
