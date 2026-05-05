"""
Единый LLM-as-judge для вспомогательных скриптов и шкал CDE (основные бенчмарки accuracy не пишут).

Промпты и парсинг JSON здесь же — модули hippo-rag / light-rag / vector-rag
подключают только клиент и модель из своих settings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

# --- Бинарный LLM-as-judge (legacy / ручные скрипты; не используется в main бенчмарков) ---

ACCURACY_SYSTEM = """Ты строгий судья бенчмарка RAG. Нужно решить: **верен ли ответ системы относительно эталона** для данного вопроса.

ПРАВИЛА:
1) **Эталон (ground_truth)** — ожидаемый факт. Если эталон конкретен (число, имя, список, да/нет), ответ должен **передать тот же смысл и ключевые факты**. Допустимы перефраз и другой порядок, если суть совпадает.
2) Если эталон непустой, а ответ — отказ («нельзя ответить», «нет в контексте», «недостаточно данных») или **игнорирует** ожидаемый факт → **correct: false**.
3) Эталоны вроде «0», «1», «7.7» — полноценны: ответ должен содержать то же значение/эквивалент.
4) Не ставь correct: true за вежливость или длину, если суть эталона не донесена.
5) Если эталон — составной (несколько пар «сущность; число»), оценивай, покрывает ли ответ **существенную часть** ожидаемого; мелкие расхождения в формулировках допустимы, пропуск целых блоков — нет.
6) Если эталон — развёрнутый ответ на открытый вопрос (аналитика), оценивай совпадение **ключевых утверждений и фактов** с эталоном; полное слово в слово не требуется.

Верни только JSON одной строкой: {"correct": true или false}"""

ACCURACY_USER_TEMPLATE = """Вопрос:
{question}

Эталон:
{ideal}

Ответ системы:
{answer}"""


# --- CDE (comprehensiveness, diversity, empowerment), шкала 1–5 ---

CDE_JUDGE_SYSTEM = """Ты оценщик ответов RAG в бенчмарке. Твоя задача — судить по соответствию **эталону** и вопросу, а не по «вежливости» или гладкости формулировки.

КРИТИЧЕСКИЕ ПРАВИЛА (имеют приоритет над общими формулировками):
1) **Эталон (ground truth)** — ожидаемый факт для этой строки бенчмарка. Если эталон НЕ пустой и содержит конкретику (число, имя, перечень, факт «да/нет» в виде числа и т.п.), а ответ системы **не передаёт этот факт** (в т.ч. говорит «нет данных», «невозможно ответить», «в контексте не упоминается», «недостаточно информации») — это **провал извлечения/ответа**, а не успех. В таких случаях:
   - comprehensiveness: 1 (иногда 2, если передан хотя бы частично релевантный факт, но не эталон).
   - empowerment: 1 (иногда 2 по той же логике).
   - diversity: 1–2, если ответ — по сути один тезис об отсутствии информации; не завышай за объём текста.
2) Эталон вроде **«0»**, **«1»**, **«7.7»** — это полноценные ответы. Если система **не** даёт то же значение/смысл (а отказывается) — п.1, оценки низкие. Не путай «честный отказ» с хорошим ответом: для бенчмарка отказ при заданном эталоне = ошибка.
3) **Не** завышай баллы за структуру, список ссылок, вежливость, развёрнутость, если суть из эталона не донесена.
4) Высокие 4–5 по comprehensiveness/empowerment только если ожидаемое содержание эталона **видно** в ответе (с учётом перефраза и чисел/имён).
5) diversity: 4–5 только если в ответе **реально** несколько различимых, уместных аспектов; короткий или длинный повтор «мы не можем» — низкая diversity.

Шкалы 1–5 — целые. Верни только JSON."""

CDE_JUDGE_USER_TEMPLATE = """{recall_block}Вопрос:
{question}

Эталон (ожидаемый ответ в бенчмарке; может быть одним числом или краткой строкой):
{ideal}

Ответ системы:
{answer}

Оцени (целые 1–5):
- **comprehensiveness**: насколько ответ **по содержанию** покрывает ожидаемое эталоном и вопросом (сопоставь с эталоном явно). Отказ/«нет в контексте» при непустом эталоне — низкий балл (см. правила system).
- **diversity**: разнообразие **уместных** независимых смыслов; не путай с длиной отказа.
- **empowerment**: насколько пользователь получил **ожидаемый факт** (или прямой эквивалент), а не уход от ответа.

JSON одной строкой, без пояснений:
{{"comprehensiveness": <int 1-5>, "diversity": <int 1-5>, "empowerment": <int 1-5>}}"""


@dataclass
class CdeScores:
    comprehensiveness: float
    diversity: float
    empowerment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "comprehensiveness": self.comprehensiveness,
            "diversity": self.diversity,
            "empowerment": self.empowerment,
        }


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2 and not text.startswith("{"):
        text = m2.group(0)
    return json.loads(text)


def resolve_judge_api_key(settings: Any) -> str:
    for attr in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        v = (getattr(settings, attr, None) or "").strip()
        if v:
            return v
    return ""


def resolve_judge_model(settings: Any) -> str:
    m = (getattr(settings, "METRICS_JUDGE_MODEL", None) or "").strip()
    return m or getattr(settings, "LLM_MODEL", "")


def openai_client_for_judge(settings: Any) -> OpenAI:
    key = resolve_judge_api_key(settings)
    if not key:
        raise RuntimeError("Нужен OPENROUTER_API_KEY или OPENAI_API_KEY в settings или .env")
    base_url = (getattr(settings, "OPENAI_API_BASE", None) or "").strip()
    if not base_url:
        raise RuntimeError("Нужен OPENAI_API_BASE в settings")
    kwargs: dict[str, Any] = {"base_url": base_url, "api_key": key}
    referer = (getattr(settings, "OPENROUTER_HTTP_REFERER", None) or "").strip()
    if referer:
        kwargs["default_headers"] = {
            "HTTP-Referer": referer,
            "X-Title": getattr(settings, "OPENROUTER_APP_TITLE", "") or "",
        }
    return OpenAI(**kwargs)


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
    data = extract_json_object(content)
    if "correct" not in data:
        raise ValueError(f"judge JSON missing 'correct': {content!r}")
    v = data["correct"]
    if not isinstance(v, bool):
        raise ValueError(f"'correct' must be bool, got {type(v).__name__}")
    return v


def cde_recall_block(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    v = row.get("recall_on_ground_truth_tokens")
    if v is None:
        return ""
    try:
        r = float(v)
    except (TypeError, ValueError):
        return ""
    return (
        f"Служебно: recall относительно эталона (токенное пересечение, 0..1) = {r:.4f}. "
        f"Если 0.0 и эталон непустой, ответ с высокой вероятностью **не** воспроизводит ожидаемый факт; "
        f"оцени строго, особенно если ответ = отказ/«нет данных».\n\n"
    )


def score_cde_answer(
    client: OpenAI,
    model: str,
    question: str,
    ideal: str,
    answer: str,
    row: dict[str, Any] | None = None,
) -> CdeScores:
    rb = cde_recall_block(row)
    user = CDE_JUDGE_USER_TEMPLATE.format(
        recall_block=rb,
        question=question or "",
        ideal=ideal or "",
        answer=answer or "",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CDE_JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = extract_json_object(content)
    c = int(data["comprehensiveness"])
    d = int(data["diversity"])
    e = int(data["empowerment"])
    for name, val in (("comprehensiveness", c), ("diversity", d), ("empowerment", e)):
        if not 1 <= val <= 5:
            raise ValueError(f"{name} out of range: {val}")
    return CdeScores(
        comprehensiveness=float(c),
        diversity=float(d),
        empowerment=float(e),
    )
