"""Запуск RAGAS ``evaluate`` и сохранение результатов в JSON."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset
from openai import OpenAI
from ragas.utils import safe_nanmean

from .config import RagasJudgeSettings

Preset = Literal["default", "faithfulness_only"]


def _patch_openai_chat_completions_extra_body(
    client: OpenAI, extra: dict[str, Any] | None
) -> OpenAI:
    """
    Подмешивать ``extra_body`` в ``chat.completions.create`` без обёртки клиента.

    ``instructor.from_openai`` требует именно ``openai.OpenAI``; кастомный прокси даёт
    ``NoneType`` у внутреннего ``.chat`` и предупреждение Instructor.
    """
    if not extra:
        return client
    completions = client.chat.completions
    orig_create = completions.create

    def create_with_extra_body(*args: Any, **kwargs: Any) -> Any:
        merged = {**extra, **(kwargs.pop("extra_body", None) or {})}
        kwargs["extra_body"] = merged
        return orig_create(*args, **kwargs)

    setattr(completions, "create", create_with_extra_body)
    return client


def _make_answer_relevancy_embeddings(client: Any, model: str) -> Any:
    """
    LangChain-интерфейс для ``AnswerRelevancy``: ``embed_query`` / ``embed_documents``.

    Современный Ragas ``OpenAIEmbeddings`` даёт только ``embed_text`` / ``embed_texts``.
    """
    from langchain_core.embeddings import Embeddings as LCEmbeddings
    from ragas.embeddings.openai_provider import OpenAIEmbeddings as RagasOpenAIEmb

    class _E(LCEmbeddings):
        def __init__(self) -> None:
            super().__init__()
            self._impl = RagasOpenAIEmb(client=client, model=model)

        def embed_query(self, text: str) -> list[float]:
            return self._impl.embed_text(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._impl.embed_texts(texts)

    return _E()


def _make_text_generation_ragas_llm(settings: RagasJudgeSettings) -> Any:
    """
    LLM с ``agenerate_text`` для метрик RAGAS, которые не используют Instructor (например
    ``ContextRelevance`` / ``nv_context_relevance``). ``llm_factory`` даёт ``InstructorLLM``
    без ``agenerate_text``, из‑за чего эти метрики падают с AttributeError.
    """
    import warnings

    from langchain_openai import ChatOpenAI
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    kw: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.api_key,
        "max_tokens": settings.llm_max_tokens,
        "temperature": 0.1,
    }
    if settings.base_url:
        kw["base_url"] = settings.base_url.rstrip("/")
    if settings.default_headers:
        kw["default_headers"] = dict(settings.default_headers)
    if settings.openai_chat_extra_body:
        kw["extra_body"] = dict(settings.openai_chat_extra_body)

    rc = RunConfig(max_retries=settings.ragas_run_max_retries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(ChatOpenAI(**kw), run_config=rc)


def records_to_hf_dataset(records: list[dict[str, Any]]) -> Dataset:
    """
    Собрать HuggingFace ``Dataset`` в «v1»-колонках RAGAS: question, answer, contexts, ground_truth.

    ``contexts`` — список строк на строку (после :func:`normalize_eval_record`).
    """
    if not records:
        return Dataset.from_dict(
            {"question": [], "answer": [], "contexts": [], "ground_truth": []}
        )
    return Dataset.from_dict(
        {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r.get("ground_truth") for r in records],
        }
    )


def _build_eval_clients(
    settings: RagasJudgeSettings, preset: Preset
) -> tuple[list[Any], Any, Any]:
    """
    Собрать метрики RAGAS 0.4.x (классы ``Metric``), LLM и эмбеддинги для ``evaluate(..., llm=, embeddings=)``.

    ``ragas.metrics.collections.*`` здесь не подходят: они другого базового класса и ломают
    ``isinstance(..., Metric)`` внутри ``aevaluate``.
    """
    from ragas.llms import llm_factory
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._context_recall import ContextRecall
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._nv_metrics import ContextRelevance

    client = OpenAI(**settings.openai_client_kwargs())
    _patch_openai_chat_completions_extra_body(client, settings.openai_chat_extra_body)
    llm = llm_factory(
        settings.llm_model,
        client=client,
        max_tokens=settings.llm_max_tokens,
    )
    embeddings = None
    if preset == "default":
        embeddings = _make_answer_relevancy_embeddings(
            client, settings.embedding_model
        )
    # Контекстные метрики RAGAS (нужны реальные ``contexts``; см. judge.io):
    # - context_recall: насколько эталон опирается на извлечённый контекст;
    # - context_precision: насколько каждый чанк полезен для эталонного ответа (AP по вердиктам);
    # - nv_context_relevance: релевантность контекста вопросу (NVIDIA / LLM).
    cr = ContextRecall()
    cp = ContextPrecision()
    crel = ContextRelevance()
    # nv_context_relevance требует agenerate_text; InstructorLLM от llm_factory его не даёт.
    crel.llm = _make_text_generation_ragas_llm(settings)
    context_bundle = [cr, cp, crel]

    # if preset == "faithfulness_only":
    #     return [Faithfulness(), *context_bundle], llm, embeddings
    # ar = AnswerRelevancy(strictness=settings.answer_relevancy_strictness)
    # return [Faithfulness(), ar, *context_bundle], llm, embeddings
    return context_bundle, llm, embeddings


def evaluate_with_ragas(
    records: list[dict[str, Any]],
    settings: RagasJudgeSettings,
    *,
    metrics_preset: Preset = "default",
    show_progress: bool = True,
    raise_exceptions: bool = False,
) -> Any:
    """
    Прогнать записи через ``ragas.evaluate``.

    Parameters
    ----------
    records:
        Список словарей после :func:`normalize_eval_record`.
    settings:
        Ключ, URL и имена моделей; ``ragas_max_workers`` — параллельность ``evaluate`` (по умолчанию 4).
    metrics_preset:
        ``default`` — faithfulness + answer_relevancy + контекстный блок RAGAS:
        ``context_recall``, ``context_precision``, ``nv_context_relevance`` (Context Relevancy).
        ``faithfulness_only`` — то же без answer_relevancy и без эмбеддингов.
        Контекстные метрики опираются на **реально извлечённые** ``contexts``; при заглушке из
        :func:`judge.io.normalize_eval_record` значения условные. ``context_precision`` и
        ``context_recall`` требуют непустой ``ground_truth``. Answer relevancy требует валидного JSON в теле ответа;
        у reasoning-моделей (например ``gpt-oss-*``) поле ``content`` часто битое — см. ``--metrics faithfulness_only``
        или смените ``--llm-model``. Число генераций answer_relevancy — поле
        ``answer_relevancy_strictness`` в ``settings`` (по умолчанию 1: без предупреждения,
        если API отдаёт только одну генерацию). Лимит длины ответа LLM — ``llm_max_tokens``
        (по умолчанию 4096; RAGAS иначе часто режет structured output на 1024).
    show_progress:
        Прогресс-бар RAGAS.
    raise_exceptions:
        Пробрасывать исключения из метрик; иначе в результате будут NaN по строкам (поведение RAGAS).

    Returns
    -------
    ragas.dataset_schema.EvaluationResult
        Результат с полями ``scores``, ``dataset`` и агрегатами в ``__repr__``.
    """
    if not records:
        raise ValueError("records must be non-empty")

    from ragas import evaluate
    from ragas.run_config import RunConfig

    metrics, llm, embeddings = _build_eval_clients(settings, metrics_preset)
    ds = records_to_hf_dataset(records)
    run_config = RunConfig(
        max_workers=settings.ragas_max_workers,
        max_retries=settings.ragas_run_max_retries,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return evaluate(
            ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            show_progress=show_progress,
            raise_exceptions=raise_exceptions,
        )


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def result_summary_mean(result: Any) -> dict[str, float | None]:
    """Средние по метрикам (как у RAGAS ``EvaluationResult.__repr__``)."""
    if not result.scores:
        return {}
    keys = result.scores[0].keys()
    return {str(k): float(safe_nanmean([row[k] for row in result.scores])) for k in keys}


def aggregate_summaries_weighted(
    parts: list[tuple[str, dict[str, float | None], int]],
) -> dict[str, float | None]:
    """
    Взвешенное среднее по числу строк для каждой метрики.

    ``parts`` — список ``(question_type, summary_mean, n_rows)``.
    """
    total_n = sum(n for _, _, n in parts if n > 0)
    if total_n <= 0:
        return {}
    keys: set[str] = set()
    for _, sm, _ in parts:
        keys.update(sm.keys())
    agg: dict[str, float | None] = {}
    for k in sorted(keys):
        num = 0.0
        den = 0
        for _, sm, n in parts:
            if n <= 0:
                continue
            v = sm.get(k)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue
            num += float(v) * n
            den += n
        agg[k] = (num / den) if den > 0 else None
    return agg


def save_evaluation_json(
    result: Any,
    path: str | Path,
    *,
    source_records: list[dict[str, Any]] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """
    Сохранить сводку и построчные скоры в UTF-8 JSON.

    Если передан ``source_records`` (тот же порядок, что при оценке), в каждую строку
    добавляются ``item_id`` / поля из исходной нормализации.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    summary = result_summary_mean(result)

    per_row: list[dict[str, Any]] = []
    for i, score_row in enumerate(result.scores):
        row: dict[str, Any] = {"index": i, "scores": _json_safe(dict(score_row))}
        if source_records is not None and i < len(source_records):
            src = source_records[i]
            if "item_id" in src:
                row["item_id"] = src["item_id"]
            row["question"] = src.get("question")
            row["answer"] = src.get("answer")
            row["ground_truth"] = src.get("ground_truth")
        per_row.append(row)

    payload = {
        "summary": _json_safe(summary),
        "rows": per_row,
    }
    if extra_meta:
        payload["meta"] = extra_meta

    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
