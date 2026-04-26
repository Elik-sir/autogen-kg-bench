"""
LLM-as-judge: бинарная метрика accuracy (ответ эквивалентен эталону для бенчмарка).

Использует те же ключи OpenRouter, что и vector_rag (settings).
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

import settings  # noqa: E402

ACCURACY_SYSTEM = """Ты строгий судья бенчмарка RAG. Нужно решить: **верен ли ответ системы относительно эталона** для данного вопроса.

ПРАВИЛА:
1) **Эталон (ground_truth)** — ожидаемый факт. Если эталон конкретен (число, имя, список, да/нет), ответ должен **передать тот же смысл и ключевые факты**. Допустимы перефраз и другой порядок, если суть совпадает.
2) Если эталон непустой, а ответ — отказ («нельзя ответить», «нет в контексте», «недостаточно данных») или **игнорирует** ожидаемый факт → **correct: false**.
3) Эталоны вроде «0», «1», «7.7» — полноценны: ответ должен содержать то же значение/эквивалент.
4) Не ставь correct: true за вежливость или длину, если суть эталона не донесена.
5) Если эталон — составной (несколько пар «сущность; число»), оценивай, покрывает ли ответ **существенную часть** ожидаемого; мелкие расхождения в формулировках допустимы, пропуск целых блоков — нет.

Верни только JSON одной строкой: {"correct": true или false}"""

ACCURACY_USER_TEMPLATE = """Вопрос:
{question}

Эталон:
{ideal}

Ответ системы:
{answer}"""


def judge_model() -> str:
    m = (getattr(settings, "METRICS_JUDGE_MODEL", None) or "").strip()
    return m or settings.LLM_MODEL


def openai_client() -> OpenAI:
    key = settings.OPENROUTER_API_KEY or ""
    if not key.strip():
        raise RuntimeError("Нужен OPENROUTER_API_KEY в settings или .env")
    kwargs: dict[str, Any] = {
        "base_url": settings.OPENAI_API_BASE,
        "api_key": key,
    }
    if settings.OPENROUTER_HTTP_REFERER:
        kwargs["default_headers"] = {
            "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
            "X-Title": settings.OPENROUTER_APP_TITLE,
        }
    return OpenAI(**kwargs)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2 and not text.startswith("{"):
        text = m2.group(0)
    return json.loads(text)


def judge_correct(
    client: OpenAI,
    model: str,
    question: str,
    ground_truth: str,
    answer: str,
) -> bool:
    user = ACCURACY_USER_TEMPLATE.format(
        question=question or "",
        ideal=ground_truth or "",
        answer=answer or "",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ACCURACY_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_json_object(content)
    if "correct" not in data:
        raise ValueError(f"judge JSON missing 'correct': {content!r}")
    v = data["correct"]
    if not isinstance(v, bool):
        raise ValueError(f"'correct' must be bool, got {type(v).__name__}")
    return v
