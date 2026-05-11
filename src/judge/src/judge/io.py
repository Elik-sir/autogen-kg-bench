"""
Загрузка примеров для RAGAS из JSON / JSONL (в т.ч. экспорты бенчмарка).

**Файлы вида ``benchmark_data.jsonl``** (``kind: item`` из HippoRAG / light-rag) уже
совместимы по полям ``question``, ``answer``, ``ground_truth``: их можно передать
в :func:`load_eval_records` без конвертации.

**Пакетный режим CLI** (``--input-dir``): каталог с результатами бенчмарка по типам
(например ``vector-rag/results``) — в нём ищутся ``simple.jsonl``, ``multi-hop.jsonl``, …
в порядке :data:`QUESTION_TYPE_ORDER`; см. :func:`discover_benchmark_result_files`.

Важно для метрик RAGAS:

- **Faithfulness**, **context_recall**, **context_precision**, **nv_context_relevance** (RAGAS)
  опираются на **реально извлечённый контекст** (список чанков). В типичном ``benchmark_data.jsonl``
  чанки **не сохраняются** — тогда подставляется техническая заглушка; эти метрики **условные**.
  Для ``context_precision`` / ``context_recall`` нужен непустой эталон ``ground_truth``.
  Чтобы метрика была осмысленной, при прогоне бенчмарка нужно **писать в каждую
  строку** поле ``contexts`` (список строк) или один ``context`` — текст,
  который реально видела модель.
- **Answer relevancy** опирается на вопрос и ответ; контекст не обязателен.
- Колонка эталона для RAGAS: при ``scoring_reference == "answer"`` (как в
  subgraph-deep) для ``ground_truth`` берётся ``ideal_for_scoring``, если оно
  есть, иначе обычный ``ground_truth``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Согласовано с ``utils.benchmark_by_type.QUESTION_TYPE_ORDER`` (прогон бенчмарка по типам).
QUESTION_TYPE_ORDER: tuple[str, ...] = (
    "simple",
    "multi-hop",
    "aggregation",
    "subgraph-deep-analytics",
)


def discover_benchmark_result_files(results_dir: str | Path) -> list[tuple[str, Path]]:
    """
    Найти в каталоге (например ``vector-rag/results``) файлы ``<тип>.jsonl`` или ``<тип>.json``
    в порядке ``QUESTION_TYPE_ORDER``. Файл ``_summary.*`` не используется.
    """
    d = Path(results_dir).expanduser().resolve()
    if not d.is_dir():
        raise NotADirectoryError(str(d))
    out: list[tuple[str, Path]] = []
    for t in QUESTION_TYPE_ORDER:
        for ext in (".jsonl", ".json"):
            p = d / f"{t}{ext}"
            if p.is_file():
                out.append((t, p))
                break
    return out


def try_load_per_type_metrics_json(
    output_dir: str | Path,
    question_type: str,
) -> tuple[dict[str, Any], int] | None:
    """
    Если в ``output_dir`` уже есть ``<question_type>.json`` (итог прошлого прогона),
    вернуть ``(summary, n_rows)`` для взвешенной сводки без повторного LLM.

    Ожидается JSON с ключами ``summary`` (object) и ``rows`` (array) — как у
    ``retrieval_metrics`` / ``generation_metrics`` / ``extended_metrics`` / RAGAS CLI.
    """
    d = Path(output_dir).expanduser().resolve()
    p = d / f"{question_type}.json"
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    rows = doc.get("rows")
    summary = doc.get("summary")
    if not isinstance(rows, list):
        return None
    if not isinstance(summary, dict):
        return None
    return dict(summary), len(rows)


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Загрузить список записей из файла.

    Поддерживается:

    - **.json** — корень: массив объектов или ``{\"items\": [...]}``;
    - **.jsonl** — по одному JSON-объекту на строку (строки ``kind: summary`` пропускаются).

    Ожидаемые поля (гибкие алиасы см. :func:`normalize_eval_record`):

    - вопрос: ``question``;
    - ответ: ``answer``;
    - контекст: ``contexts`` (список строк) или одиночный ``context`` / ``rag_context``;
    - эталон: см. :func:`normalize_eval_record` (учёт ``scoring_reference`` / ``ideal_for_scoring``).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    suf = p.suffix.lower()
    if suf == ".jsonl":
        return _load_jsonl(p)
    if suf == ".json" or suf == "":
        return _load_json(p)
    raise ValueError(f"Unsupported extension {p.suffix!r}; use .json or .jsonl")


def _resolve_reference_for_ragas(row: dict[str, Any]) -> str | None:
    """
    Эталон для колонки ``ground_truth`` в HF Dataset (RAGAS v1).

    Для экспорта бенчмарка с ``scoring_reference == "answer"`` используем
    ``ideal_for_scoring``, как в LLM-judge бенчмарка.
    """
    sr = (row.get("scoring_reference") or "").strip()
    if sr == "answer":
        ref = row.get("ideal_for_scoring")
        if ref is not None and str(ref).strip():
            return str(ref).strip()
    ref = row.get("ground_truth")
    if ref is None:
        ref = row.get("reference") or row.get("ideal")
    if ref is not None:
        s = str(ref).strip()
        return s or None
    return None


def _load_json(p: Path) -> list[dict[str, Any]]:
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list or an object with key 'items'")
    return [normalize_eval_record(dict(x)) for x in raw if isinstance(x, dict)]


def _load_jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL line {line_no}: {e}") from e
        if not isinstance(row, dict):
            continue
        if row.get("kind") == "summary":
            continue
        if row.get("kind") == "item" or "question" in row:
            out.append(normalize_eval_record(row))
    return out


def normalize_eval_record(row: dict[str, Any]) -> dict[str, Any]:
    """
    Привести запись к полям, которые затем попадут в HF Dataset для RAGAS v1-колонок.

    Итоговые ключи: ``question``, ``answer``, ``contexts`` (list[str]), ``ground_truth`` (str | None).
    """
    q = row.get("question")
    if q is None or not str(q).strip():
        raise ValueError("record missing non-empty 'question'")
    ans = row.get("answer")
    if ans is None or not str(ans).strip():
        raise ValueError("record missing non-empty 'answer'")

    ctxs = row.get("contexts")
    if ctxs is None:
        single = (
            row.get("context")
            or row.get("rag_context")
            or row.get("retrieved_context")
            or row.get("retrieved_contexts")
            or row.get("retrieved_chunks")
        )
        if single is None:
            ctxs = []
        elif isinstance(single, list):
            ctxs = [str(c).strip() for c in single if str(c).strip()]
        else:
            ctxs = [str(single).strip()] if str(single).strip() else []
    else:
        if not isinstance(ctxs, list):
            raise ValueError("'contexts' must be a list of strings when provided")
        ctxs = [str(c).strip() for c in ctxs if str(c).strip()]

    if not ctxs:
        # Faithfulness в RAGAS требует непустой список контекстов
        ctxs = ["(No retrieval context was provided for this example.)"]

    ref = _resolve_reference_for_ragas(row)

    out = {
        "question": str(q).strip(),
        "answer": str(ans).strip(),
        "contexts": ctxs,
        "ground_truth": ref,
    }
    if row.get("item_id") is not None:
        out["item_id"] = str(row["item_id"])
    elif row.get("index") is not None:
        out["item_id"] = str(row["index"])
    return out
