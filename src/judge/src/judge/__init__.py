"""
Оценка ответов RAG через RAGAS (OpenAI-совместимый API).

Изолированное uv-окружение: из каталога ``src/judge`` выполните::

    uv sync
    uv run python -m judge --input data/sample_ragas_input.json --output out.json

В режиме ``default`` RAGAS считает ещё контекстный блок: **context_recall**, **context_precision**,
**nv_context_relevance** (Context Relevancy); в ``faithfulness_only`` те же три без answer_relevancy.

Оценка **по типам вопросов** (файлы из ``vector-rag/results`` и т.п.)::

    uv run python -m judge --input-dir ../benchmarks/vector-rag/results --output-dir data/ragas_vector

По умолчанию при ``--input-dir`` без ``--output-dir`` результаты пишутся в ``<input-dir>/ragas/``.

Либо ``uv run judge-cli ...`` (см. ``[project.scripts]`` в pyproject).

Переменные по умолчанию подхватываются из ``.env`` в корне этого проекта (рядом с pyproject.toml).

Файл ``benchmark_data.jsonl`` из бенчмарков можно передать в ``--input`` как есть
(см. модуль ``judge.io``: строки ``kind: item``, эталон с учётом ``ideal_for_scoring``).

Публичный API: :class:`RagasJudgeSettings`, :func:`load_eval_records`,
:func:`records_to_hf_dataset`, :func:`evaluate_with_ragas`, :func:`save_evaluation_json`,
:func:`result_summary_mean`.
"""

from __future__ import annotations

from .config import RagasJudgeSettings
from .io import load_eval_records
from .ragas_eval import (
    evaluate_with_ragas,
    records_to_hf_dataset,
    result_summary_mean,
    save_evaluation_json,
)

__all__ = [
    "RagasJudgeSettings",
    "evaluate_with_ragas",
    "load_eval_records",
    "records_to_hf_dataset",
    "result_summary_mean",
    "save_evaluation_json",
]
