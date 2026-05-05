"""
Векторный RAG (LangChain: FAISS + top-k + LLM) и прогон вопросов бенчмарка.

Вопросы берутся из каталога `benchmark_questions_by_type/` в корне репозитория
(файлы `simple.json`, `multi-hop.json`, … — порядок прогона задаётся в
`utils.benchmark_by_type.QUESTION_TYPE_ORDER`). Результаты пишутся в `results/<тип>.<суффикс>`
и сводка в `results/_summary.json`.

  cd src/benchmarks/vector-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py

Вопросы внутри типа обрабатываются параллельно (``ThreadPoolExecutor``); лимит —
``settings.QUESTION_CONCURRENCY`` (по умолчанию 8). Поставьте ``1``, если провайдер
режет по rate limit или при странных ошибках FAISS.

Положите `corpus.txt` (или путь в settings.CORPUS_FILE) рядом или скопируйте из light-rag.
"""

from __future__ import annotations

import json
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

_SRC = _BENCH_ROOT.parent.parent
_BENCHMARK_BY_TYPE_PATH = _SRC / "utils" / "benchmark_by_type.py"
_BENCHMARK_BY_TYPE_SPEC = importlib.util.spec_from_file_location(
    "shared_benchmark_by_type",
    _BENCHMARK_BY_TYPE_PATH,
)
assert _BENCHMARK_BY_TYPE_SPEC is not None
assert _BENCHMARK_BY_TYPE_SPEC.loader is not None
_benchmark_by_type = importlib.util.module_from_spec(_BENCHMARK_BY_TYPE_SPEC)
_BENCHMARK_BY_TYPE_SPEC.loader.exec_module(_benchmark_by_type)

import settings  # noqa: E402
from utils.io import (  # noqa: E402
    resolve_working_dir,
    write_json,
    write_results,
)
from utils.metrics import recall_overlap  # noqa: E402
from utils.progress import print_question_progress  # noqa: E402

build_benchmark_plan = _benchmark_by_type.build_benchmark_plan
output_suffix_from_setting = _benchmark_by_type.output_suffix_from_setting
results_subdir = _benchmark_by_type.results_subdir

REPO_ROOT = settings.VECTOR_RAG_DIR.parent.parent.parent


def run() -> int:
    from corpus_text import load_corpus, resolved_corpus_path  # noqa: WPS433
    from vector_rag import answer_from_store_with_contexts, build_or_load_vectorstore  # noqa: WPS433

    plan = build_benchmark_plan(
        repo_root=REPO_ROOT,
        benchmark_pkg_dir=settings.VECTOR_RAG_DIR,
        benchmark_file_setting=settings.BENCHMARK_FILE,
        benchmark_questions_dir_setting=getattr(settings, "BENCHMARK_QUESTIONS_DIR", ""),
    )
    batches = [(c, p, its) for c, p, its in plan.multi_parts]
    if not batches:
        print("Нет ни одного файла вопросов по типам в benchmark_questions_by_type.", file=sys.stderr)
        return 1

    limit = int(settings.LIMIT_QUESTIONS)
    total_planned = sum(len(b[2]) for b in batches)
    if limit and limit > 0:
        total_planned = min(total_planned, limit)

    try:
        corpus = load_corpus()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not corpus.strip():
        print("Пустой корпус.", file=sys.stderr)
        return 1

    work = resolve_working_dir(
        working_dir_setting=settings.WORKING_DIR,
        vector_rag_dir=settings.VECTOR_RAG_DIR,
    )
    print(f"Индексация FAISS (langchain)… chunk={settings.CHUNK_SIZE}, k={settings.RETRIEVAL_K}")
    store = build_or_load_vectorstore(corpus, work)
    corpus_path = str(resolved_corpus_path())
    concurrency = max(1, int(getattr(settings, "QUESTION_CONCURRENCY", 8)))
    print(f"Режим бенчмарка: multi, вопросов к прогону (с учётом лимита): {total_planned}")
    print(f"Параллельность вопросов: {concurrency} (settings.QUESTION_CONCURRENCY)")

    def run_questions(question_items: list[dict], *, type_label: str | None) -> list[dict]:
        n = len(question_items)
        if n == 0:
            return []

        def one(i: int, it: dict) -> dict:
            q = it.get("question", "")
            ground_truth = it.get("ground_truth")
            complexity = type_label or str(it.get("complexity", "") or "")
            try:
                answer, contexts = answer_from_store_with_contexts(store, q)
            except Exception as e:  # noqa: BLE001
                answer, contexts = f"[error] {e}", []
            return {
                "index": i,
                "complexity": complexity,
                "recall_on_ground_truth_tokens": round(
                    recall_overlap(ground_truth, str(answer)), 4
                ),
                "question": q,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
            }

        workers = min(concurrency, n)
        if workers <= 1:
            rows = [one(i, it) for i, it in enumerate(question_items, 1)]
            for r in rows:
                print_question_progress(r, type_label, n)
            return rows

        done_lock = threading.Lock()
        done_count = 0

        by_index: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(one, i, it): i for i, it in enumerate(question_items, 1)
            }
            for fut in as_completed(future_map):
                i = future_map[fut]
                rdict = fut.result()
                by_index[i] = rdict
                with done_lock:
                    done_count += 1
                    dc = done_count
                print_question_progress(rdict, type_label, n, completed=dc)
        return [by_index[i] for i in range(1, n + 1)]

    remaining = limit if limit and limit > 0 else 0
    all_results: list[dict] = []
    per_type_meta: list[dict] = []

    sfx = output_suffix_from_setting(settings.OUTPUT_FILE)
    res_dir = results_subdir(settings.VECTOR_RAG_DIR)

    for complexity, bpath, raw_items in batches:
        chunk = raw_items
        if remaining > 0:
            chunk = raw_items[:remaining]
            remaining -= len(chunk)
        if not chunk:
            continue
        type_key = complexity or "mixed"
        batch_results = run_questions(chunk, type_label=type_key)
        all_results.extend(batch_results)
        mean_b = sum(r["recall_on_ground_truth_tokens"] for r in batch_results) / max(
            len(batch_results), 1
        )
        row_summary = {
            "question_type": type_key,
            "benchmark": str(bpath),
            "n": len(batch_results),
            "mean_recall_on_ground_truth_tokens": round(mean_b, 4),
        }
        per_type_meta.append(row_summary)

        type_summary = {
            "settings": "settings.py",
            "backend": "vector-rag-langchain-faiss",
            "corpus": corpus_path,
            "benchmark_mode": "multi",
            "question_type": type_key,
            "benchmark": str(bpath),
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "retrieval_k": settings.RETRIEVAL_K,
            "n": len(batch_results),
            "mean_recall_on_ground_truth_tokens": round(mean_b, 4),
        }
        write_results(res_dir / f"{type_key}{sfx}", type_summary, batch_results)
        print(f"Тип {type_key!r}: результаты → {res_dir / f'{type_key}{sfx}'}")

        if remaining == 0 and limit > 0:
            break

    if not all_results:
        print("Нет вопросов для прогона.", file=sys.stderr)
        return 1

    mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in all_results) / len(all_results)
    summary: dict = {
        "settings": "settings.py",
        "backend": "vector-rag-langchain-faiss",
        "corpus": corpus_path,
        "benchmark_mode": "multi",
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "retrieval_k": settings.RETRIEVAL_K,
        "n": len(all_results),
        "mean_recall_on_ground_truth_tokens": round(mean_recall, 4),
        "by_question_type": per_type_meta,
        "results_dir": str(res_dir),
    }
    write_json(res_dir / "_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Сводка по типам: {res_dir / '_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
