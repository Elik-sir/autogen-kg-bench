"""
Production-ready LLM-as-a-Judge for RAG answers (DRAGOn-style criteria).

- Async batch evaluation via :class:`LLMJudgeEvaluator` and ``asyncio``.
- OpenAI-compatible API (:class:`openai.AsyncOpenAI`) for GPT, OpenRouter, vLLM, etc.
- Pydantic schemas for inputs and structured judge outputs (score + reasoning per axis).
- Retries (tenacity) for rate limits, transient API errors, and invalid JSON.

Example::

    import asyncio
    from utils.rag_llm_judge import LLMJudgeEvaluator, RagEvalInput

    async def main():
        ev = LLMJudgeEvaluator(
            api_key="...",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )
        data = [
            RagEvalInput(
                question="What is the capital of France?",
                answer="Paris is the capital.",
                context="France is a country in Europe. Its capital is Paris.",
                ground_truth="Paris",
                item_id="0",
            )
        ]
        out = await ev.evaluate_batch(data)
        ev.save_results("judge_out.json", out)

    asyncio.run(main())
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError, field_validator
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic: вход одной строки датасета
# ---------------------------------------------------------------------------


class RagEvalInput(BaseModel):
    """Один пример для оценки: вопрос, ответ модели, контекст RAG и опционально эталон."""

    question: str = Field(..., description="Пользовательский запрос.")
    answer: str = Field(..., description="Ответ тестируемой RAG-системы.")
    context: str = Field(
        default="",
        description="Текст контекста (чанки, подграф KG и т.д.), на котором строился ответ.",
    )
    ground_truth: str | None = Field(
        default=None,
        description="Эталонный ответ для проверки фактической корректности; может отсутствовать.",
    )
    item_id: str | None = Field(default=None, description="Внешний идентификатор строки датасета.")

    @field_validator("question", "answer", mode="before")
    @classmethod
    def _strip_required(cls, v: Any) -> str:
        if v is None:
            raise ValueError("must not be null")
        s = str(v).strip()
        if not s:
            raise ValueError("must not be empty")
        return s

    @field_validator("context", "ground_truth", mode="before")
    @classmethod
    def _strip_optional(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None


# ---------------------------------------------------------------------------
# Pydantic: структурированный ответ судьи (по оси — score + reasoning)
# ---------------------------------------------------------------------------


class CriterionScore(BaseModel):
    """Оценка по одному критерию: числовой балл и краткое обоснование."""

    score: int = Field(..., ge=1, le=5, description="Целая оценка от 1 (плохо) до 5 (отлично).")
    reasoning: str = Field(
        ...,
        min_length=1,
        description="1–3 предложения: почему выставлен такой балл.",
    )


class DragonJudgeVerdict(BaseModel):
    """
    Итоговая разметка судьи по методологии, близкой к DRAGOn (RAG-качество).

    - **answer_relevance**: соответствие ответа исходному запросу.
    - **fluff**: насколько ответ без «воды» и лишних отступлений (5 — почти нет лишнего).
    - **faithfulness**: насколько утверждения опираются на контекст (без выдумок сверх него).
    - **correctness**: фактическая согласованность с ground truth, если он дан; иначе — с контекстом.
    - **context_sufficiency**: достаточно ли контекста, чтобы обоснованно ответить на вопрос.
    """

    answer_relevance: CriterionScore
    fluff: CriterionScore
    faithfulness: CriterionScore
    correctness: CriterionScore
    context_sufficiency: CriterionScore


class EvaluatedSample(BaseModel):
    """Результат оценки одной строки: вход, вердикт или ошибка."""

    item_id: str | None = None
    input: RagEvalInput
    verdict: DragonJudgeVerdict | None = None
    error: str | None = None
    raw_model_output: str | None = Field(default=None, repr=False)

    def model_dump_for_json(self) -> dict[str, Any]:
        """Сериализация без огромных повторов при необходимости — здесь полный дамп."""
        return self.model_dump(mode="json")


class JudgeBatchReport(BaseModel):
    """Обёртка для сохранения на диск (метаданные + список оценок)."""

    created_at: str
    model: str
    base_url: str
    n_input: int
    n_success: int
    n_failed: int
    samples: list[EvaluatedSample]


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------

DRAGON_JUDGE_SYSTEM: Final[str] = """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
Your task is to score a single assistant answer using five criteria inspired by the DRAGOn-style RAG evaluation methodology.

General rules:
- Use integer scores from 1 (very poor) to 5 (excellent) for every criterion.
- Each score MUST be accompanied by a short reasoning (1–3 sentences), grounded in the provided texts.
- Do not invent facts that are not supported by the question, context, or ground truth.
- Output MUST be a single JSON object only, no markdown fences, no extra keys beyond those in the schema described in the user message.

Criterion definitions:
1) answer_relevance — How well the answer addresses what the user actually asked (focus, scope, missing parts).
2) fluff — Absence of unnecessary padding, hedging without substance, generic filler, or off-topic material (5 = concise and on-point).
3) faithfulness — Whether claims in the answer are supported by the provided CONTEXT (including KG/text snippets). Penalize hallucinations and unsupported specifics.
4) correctness — Factual alignment: if GROUND TRUTH is provided, compare to it; if not, judge internal factual consistency with CONTEXT only (do not guess external facts).
5) context_sufficiency — Whether the CONTEXT is sufficient to answer the QUESTION well; if the context is empty or irrelevant, scores should reflect that honestly.
"""


def _build_user_message(inp: RagEvalInput) -> str:
    gt_block = (
        f"GROUND TRUTH (reference answer for factual check):\n{inp.ground_truth}\n"
        if inp.ground_truth
        else "GROUND TRUTH: not provided — for correctness, judge factual consistency with CONTEXT only.\n"
    )
    ctx = inp.context.strip() if inp.context else "(empty — no retrieved context was supplied)"
    return f"""QUESTION:
{inp.question}

CONTEXT (retrieved / KG material the system was allowed to use):
{ctx}

MODEL ANSWER:
{inp.answer}

{gt_block}
Return JSON with exactly these keys and shapes (integers 1–5 for each score):
{{
  "answer_relevance": {{"score": <int>, "reasoning": "<str>"}},
  "fluff": {{"score": <int>, "reasoning": "<str>"}},
  "faithfulness": {{"score": <int>, "reasoning": "<str>"}},
  "correctness": {{"score": <int>, "reasoning": "<str>"}},
  "context_sufficiency": {{"score": <int>, "reasoning": "<str>"}}
}}
"""


# ---------------------------------------------------------------------------
# JSON extraction (совместимость с моделями, оборачивающими JSON в ```)
# ---------------------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    """Извлекает первый JSON-объект из ответа модели."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2 and not text.lstrip().startswith("{"):
        text = m2.group(0)
    return json.loads(text)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code is None:
            return True
        return code >= 500 or code == 429
    if isinstance(exc, (json.JSONDecodeError, ValueError, ValidationError)):
        return True
    return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class LLMJudgeEvaluator:
    """
    Асинхронный LLM-судья с батчевой оценкой и ограничением параллелизма.

    Parameters
    ----------
    api_key:
        API-ключ (или placeholder для локального шлюза).
    base_url:
        OpenAI-совместимый ``base_url`` (OpenAI, OpenRouter, vLLM, LiteLLM proxy и т.д.).
    model:
        Имя модели на стороне провайдера.
    max_concurrency:
        Максимум одновременных запросов к API.
    max_attempts:
        Число попыток на один пример (ретраи tenacity + повтор при невалидном JSON).
    request_timeout_sec:
        Таймаут HTTP для одного вызова completions.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_concurrency: int = 8,
        max_attempts: int = 5,
        request_timeout_sec: float = 120.0,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_concurrency = max_concurrency
        self._max_attempts = max_attempts
        self._timeout = request_timeout_sec
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self._base_url,
            "timeout": self._timeout,
        }
        if default_headers:
            kwargs["default_headers"] = default_headers
        self._client = AsyncOpenAI(**kwargs)

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    async def _call_model_once(self, inp: RagEvalInput) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": DRAGON_JUDGE_SYSTEM},
                {"role": "user", "content": _build_user_message(inp)},
            ],
            temperature=0.0,
        )
        choice = resp.choices[0].message
        return (choice.content or "").strip()

    async def _evaluate_one_with_retries(self, inp: RagEvalInput) -> EvaluatedSample:
        """Повторы при сетевых ошибках и невалидном JSON/схеме."""

        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential_jitter(initial=1, max=60),
            stop=stop_after_attempt(self._max_attempts),
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "retry judge item_id=%s attempt=%s exc=%s",
                inp.item_id,
                rs.attempt_number,
                rs.outcome.exception(),
            ),
        )
        async def _attempt() -> tuple[DragonJudgeVerdict, str]:
            raw = await self._call_model_once(inp)
            try:
                data = extract_json_object(raw)
                verdict = DragonJudgeVerdict.model_validate(data)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"invalid judge JSON: {e}: {raw[:500]!r}") from e
            return verdict, raw

        try:
            verdict, raw = await _attempt()
            return EvaluatedSample(
                item_id=inp.item_id,
                input=inp,
                verdict=verdict,
                error=None,
                raw_model_output=raw,
            )
        except Exception as e:  # noqa: BLE001 — финальный отказ после ретраев
            logger.exception("judge failed after retries item_id=%s", inp.item_id)
            return EvaluatedSample(
                item_id=inp.item_id,
                input=inp,
                verdict=None,
                error=f"{type(e).__name__}: {e}",
                raw_model_output=None,
            )

    async def evaluate_batch(self, dataset: Sequence[RagEvalInput]) -> list[EvaluatedSample]:
        """
        Асинхронно оценить батч примеров с ограничением параллелизма.

        Порядок результатов соответствует порядку ``dataset``.
        """
        if not dataset:
            return []

        sem = asyncio.Semaphore(self._max_concurrency)

        async def _run(inp: RagEvalInput) -> EvaluatedSample:
            async with sem:
                return await self._evaluate_one_with_retries(inp)

        tasks = [asyncio.create_task(_run(item)) for item in dataset]
        return list(await asyncio.gather(*tasks))

    def save_results(self, path: str | Path, samples: Sequence[EvaluatedSample]) -> None:
        """
        Сохранить результаты в JSON-файл (UTF-8, человекочитаемый indent).

        В файл пишется :class:`JudgeBatchReport` с метаданными прогона и списком
        :class:`EvaluatedSample`.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        report = JudgeBatchReport(
            created_at=datetime.now(UTC).isoformat(),
            model=self._model,
            base_url=self._base_url,
            n_input=len(samples),
            n_success=sum(1 for s in samples if s.verdict is not None),
            n_failed=sum(1 for s in samples if s.error is not None),
            samples=list(samples),
        )
        p.write_text(report.model_dump_json(indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Утилиты загрузки датасета из JSON
# ---------------------------------------------------------------------------


def load_rag_eval_inputs_from_json_file(path: str | Path) -> list[RagEvalInput]:
    """
    Загрузить список примеров из JSON-файла.

    Ожидается либо ``[ {...}, ... ]``, либо ``{"items": [...]}``.
    Поля каждого объекта маппятся на :class:`RagEvalInput`` (лишние ключи игнорируются).
    """
    raw_text = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list or an object with key 'items'")
    out: list[RagEvalInput] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"item {i} must be an object")
        row = dict(row)
        if row.get("item_id") is None and "index" in row:
            row["item_id"] = str(row["index"])
        out.append(RagEvalInput.model_validate(row))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_headers(raw: str | None) -> dict[str, str] | None:
    if not raw or not raw.strip():
        return None
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k.strip()] = v.strip()
    return out or None


async def _async_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Async DRAGOn-style LLM judge for RAG answers")
    p.add_argument("--input", type=Path, required=True, help="JSON file: list of RagEvalInput objects")
    p.add_argument("--output", type=Path, required=True, help="Output .json path")
    p.add_argument("--api-key", default="", help="API key (else OPENAI_API_KEY env)")
    p.add_argument("--base-url", default="https://api.openai.com/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument(
        "--headers",
        default="",
        help='Optional comma-separated headers, e.g. "HTTP-Referer: https://x,X-Title: bench"',
    )
    args = p.parse_args(argv)

    import os

    api_key = args.api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Need --api-key or OPENAI_API_KEY", file=sys.stderr)
        return 1

    dataset = load_rag_eval_inputs_from_json_file(args.input)
    ev = LLMJudgeEvaluator(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        max_concurrency=args.concurrency,
        max_attempts=args.max_attempts,
        default_headers=_parse_headers(args.headers),
    )
    results = await ev.evaluate_batch(dataset)
    ev.save_results(args.output, results)
    print(f"Wrote {args.output} ({len(results)} rows, errors={sum(1 for r in results if r.error)})")
    return 0


def main() -> None:
    """Точка входа для ``python -m utils.rag_llm_judge`` при настроенном PYTHONPATH."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
