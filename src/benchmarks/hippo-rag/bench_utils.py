"""Утилиты для прогона HippoRAG-бенчмарка (пути, метрики, вывод)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import settings

REPO_ROOT = settings.HIPPO_RAG_DIR.parent.parent.parent


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


def _resolve_output_path() -> Path:
    s = settings.OUTPUT_FILE
    if s and str(s).strip():
        p = Path(s).expanduser()
        return p if p.is_absolute() else (settings.HIPPO_RAG_DIR / p).resolve()
    return (REPO_ROOT / "hipporag_benchmark_results.json").resolve()


def _resolve_working_dir() -> Path:
    p = Path(settings.WORKING_DIR).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (settings.HIPPO_RAG_DIR / p).resolve()


def _write_results(path: Path, summary: dict, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        lines = [json.dumps({"kind": "summary", **summary}, ensure_ascii=False)]
        lines += [json.dumps({"kind": "item", **it}, ensure_ascii=False) for it in items]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(
            json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _prepare_hipporag_env() -> None:
    """Ключ и заголовки для OpenAI-совместимых клиентов (OpenRouter) внутри hipporag."""
    key = (settings.OPENAI_API_KEY or "").strip()
    if not key:
        key = (os.environ.get("OPENROUTER_API_KEY", "") or "").strip()
    os.environ["OPENAI_API_KEY"] = key
    if not (os.environ.get("OPENROUTER_API_KEY", "") or "").strip():
        os.environ["OPENROUTER_API_KEY"] = key
    os.environ.setdefault("OPENAI_BASE_URL", settings.OPENAI_API_BASE)
    if settings.OPENROUTER_HTTP_REFERER.strip():
        os.environ["OPENROUTER_HTTP_REFERER"] = settings.OPENROUTER_HTTP_REFERER.strip()
    if settings.OPENROUTER_APP_TITLE.strip():
        os.environ["OPENROUTER_APP_TITLE"] = settings.OPENROUTER_APP_TITLE.strip()


def _extract_contexts_from_rag_qa(raw: Any, qa_top_k: int) -> list[str]:
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


def _extract_answer_from_result(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    # HippoRAG.rag_qa возвращает (solutions, llm_messages, metadata[, ...]) — не str(dict).
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
