"""
LightRAG: индексация текста и прогон бенчмарка.

Все параметры — константы в `settings.py`. Запуск:
  cd src/benchmarks/light-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py

Уже проиндексировано — только вопросы бенчмарка:
  (PowerShell)  $env:LIGHTRAG_QUERY_ONLY = "1"; uv run python main.py

Вопросы берутся из каталога `benchmark_questions_by_type/` (см. `utils.benchmark_by_type`).
Результаты — в `results/<тип>.<суффикс>` и сводка `results/_summary.json`.

В каждую запись результатов добавляется `contexts` — тексты чанков из retrieval
(LightRAG `aquery_llm` → `data.chunks`), чтобы RAGAS Faithfulness опиралась на
реальный контекст (см. `src/judge`, `load_eval_records`). Объём retrieval задаётся
в `settings.py` (`QUERY_*` / `LIGHTRAG_QUERY_*`), по умолчанию сужен для метрик.
Параллельные `aquery_llm`: `QUERY_CONCURRENCY` / env `LIGHTRAG_QUERY_CONCURRENCY` (по умолчанию 4).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
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
from utils.indexing import require_indexing_complete  # noqa: E402
from utils.io import resolve_working_dir, write_json, write_results  # noqa: E402
from utils.query import benchmark_query_param, run_single_benchmark_query  # noqa: E402

build_benchmark_plan = _benchmark_by_type.build_benchmark_plan
output_suffix_from_setting = _benchmark_by_type.output_suffix_from_setting
results_subdir = _benchmark_by_type.results_subdir

REPO_ROOT = settings.LIGHT_RAG_DIR.parent.parent.parent


async def _run() -> int:
    from openrouter_lightrag import (  # noqa: WPS433
        apply_openrouter_env_defaults,
        build_rag,
        clear_working_dir,
        ensure_lightrag_available,
        indexing_llm_retry_scope,
    )
    from raw_corpus import load_raw_text, resolved_corpus_path  # noqa: WPS433

    ensure_lightrag_available()
    apply_openrouter_env_defaults()

    plan = build_benchmark_plan(
        repo_root=REPO_ROOT,
        benchmark_pkg_dir=settings.LIGHT_RAG_DIR,
        benchmark_file_setting=settings.BENCHMARK_FILE,
        benchmark_questions_dir_setting=getattr(settings, "BENCHMARK_QUESTIONS_DIR", ""),
    )
    batches = [(c, p, its) for c, p, its in plan.multi_parts]
    if not batches:
        print("Нет файлов в benchmark_questions_by_type.", file=sys.stderr)
        return 1
    bench_path = batches[0][1]

    limit = int(settings.LIMIT_QUESTIONS)
    remaining = limit if limit and limit > 0 else 0

    corpus_path = resolved_corpus_path()
    corpus = ""

    if settings.QUERY_ONLY and settings.RESUME_PIPELINE_ONLY:
        print(
            "Несовместимо: QUERY_ONLY и RESUME_PIPELINE_ONLY.",
            file=sys.stderr,
        )
        return 1
    if settings.QUERY_ONLY and settings.REBUILD_CACHE:
        print(
            "Несовместимо: QUERY_ONLY и REBUILD_CACHE.",
            file=sys.stderr,
        )
        return 1

    if settings.QUERY_ONLY:
        if not corpus_path.is_file():
            print(
                f"Предупреждение: корпус не найден (для QUERY_ONLY не нужен): {corpus_path}",
                file=sys.stderr,
            )
    else:
        if not corpus_path.is_file():
            print(f"Текстовый корпус не найден: {corpus_path}", file=sys.stderr)
            return 1

        if not settings.RESUME_PIPELINE_ONLY:
            try:
                corpus = load_raw_text(corpus_path)
            except FileNotFoundError as e:
                print(str(e), file=sys.stderr)
                return 1
            if not corpus.strip():
                print("Файл корпуса пустой: нечего индексировать.", file=sys.stderr)
                return 1

    work = resolve_working_dir(
        working_dir_setting=settings.WORKING_DIR,
        light_rag_dir=settings.LIGHT_RAG_DIR,
    )
    if settings.REBUILD_CACHE and settings.RESUME_PIPELINE_ONLY:
        print(
            "Несовместимо: REBUILD_CACHE и RESUME_PIPELINE_ONLY. "
            "Отключите очистку кэша для дорисовки.",
            file=sys.stderr,
        )
        return 1
    if settings.REBUILD_CACHE:
        clear_working_dir(work)
    work.mkdir(parents=True, exist_ok=True)

    rag = build_rag(work)
    await rag.initialize_storages()

    mode = str(settings.QUERY_MODE).lower()
    if mode not in ("naive", "local", "global", "hybrid"):
        mode = "hybrid"

    run_total = 0
    _rem = remaining
    for _, _, raw in batches:
        if _rem > 0:
            c = min(len(raw), _rem)
            run_total += c
            _rem -= c
        else:
            run_total += len(raw)

    try:
        print(
            f"(LLM: timeout={settings.LLM_TIMEOUT_SEC}s, воркер ~{2 * settings.LLM_TIMEOUT_SEC}s; "
            f"при индексации — повторы вызова до успеха в пределах бюджета воркера)"
        )
        if settings.QUERY_ONLY:
            print(
                "Режим QUERY_ONLY: пропуск ainsert и проверки doc_status, "
                "прогон только по вопросам из бенчмарка."
            )
        elif settings.RESUME_PIPELINE_ONLY:
            print(
                "Режим RESUME_PIPELINE_ONLY: дорисовка очереди (FAILED/PENDING/PROCESSING), "
                "без повторного enqueue текста. Уже закэшированные extract-ответы подхватятся из KV."
            )
            with indexing_llm_retry_scope():
                await rag.apipeline_process_enqueue_documents()
        else:
            print(f"Индексация (ainsert), источник: {corpus_path} …")
            with indexing_llm_retry_scope():
                await rag.ainsert(corpus)
        if not settings.QUERY_ONLY:
            indexing_err = await require_indexing_complete(rag)
            if indexing_err:
                print(indexing_err, file=sys.stderr)
                return 2
            print(f"Индексация успешна, вопросов в прогоне: {run_total}")
        else:
            print(f"Вопросов в прогоне: {run_total}")

        print(
            "QueryParam (retrieval): "
            f"top_k={settings.QUERY_TOP_K}, chunk_top_k={settings.QUERY_CHUNK_TOP_K}, "
            f"max_entity_tokens={settings.QUERY_MAX_ENTITY_TOKENS}, "
            f"max_relation_tokens={settings.QUERY_MAX_RELATION_TOKENS}, "
            f"max_total_tokens={settings.QUERY_MAX_TOTAL_TOKENS}"
            + (
                " [LIGHTRAG_QUERY_FULL_BUDGET]"
                if getattr(settings, "USE_FULL_QUERY_BUDGET", False)
                else ""
            )
        )
        _qp = benchmark_query_param(mode)
        _conc = int(getattr(settings, "QUERY_CONCURRENCY", 1))
        print(f"Параллельных запросов к RAG (aquery_llm): {_conc}", flush=True)

        _sem = asyncio.Semaphore(_conc)
        _plock = asyncio.Lock()
        results_all: list[dict] = []
        per_type_meta: list[dict] = []
        rem = remaining
        sfx = output_suffix_from_setting(settings.OUTPUT_FILE)
        res_dir = results_subdir(settings.LIGHT_RAG_DIR)

        for complexity, bpath, raw_items in batches:
            chunk = raw_items
            if rem > 0:
                chunk = raw_items[:rem]
                rem -= len(chunk)
            if not chunk:
                continue
            n_items = len(chunk)
            _tasks = [
                run_single_benchmark_query(
                    rag,
                    _qp,
                    _sem,
                    _plock,
                    index=i,
                    item=it,
                    n_items=n_items,
                )
                for i, it in enumerate(chunk, 1)
            ]
            batch_results = list(await asyncio.gather(*_tasks))
            results_all.extend(batch_results)
            mean_b = sum(r["recall_on_ground_truth_tokens"] for r in batch_results) / max(
                len(batch_results), 1
            )
            type_key = complexity or "mixed"
            per_type_meta.append(
                {
                    "question_type": type_key,
                    "benchmark": str(bpath),
                    "n": len(batch_results),
                    "mean_recall_on_ground_truth_tokens": round(mean_b, 4),
                }
            )
            type_summary = {
                "settings": "settings.py",
                "corpus": str(corpus_path),
                "benchmark_mode": "multi",
                "question_type": type_key,
                "benchmark": str(bpath),
                "mode": mode,
                "n": len(batch_results),
                "mean_recall_on_ground_truth_tokens": round(mean_b, 4),
                "query_top_k": settings.QUERY_TOP_K,
                "query_chunk_top_k": settings.QUERY_CHUNK_TOP_K,
                "query_max_entity_tokens": settings.QUERY_MAX_ENTITY_TOKENS,
                "query_max_relation_tokens": settings.QUERY_MAX_RELATION_TOKENS,
                "query_max_total_tokens": settings.QUERY_MAX_TOTAL_TOKENS,
                "query_full_budget": getattr(settings, "USE_FULL_QUERY_BUDGET", False),
                "query_concurrency": int(getattr(settings, "QUERY_CONCURRENCY", 1)),
            }
            write_results(res_dir / f"{type_key}{sfx}", type_summary, batch_results)
            print(f"Тип {type_key!r}: результаты → {res_dir / f'{type_key}{sfx}'}", flush=True)
            if rem == 0 and remaining > 0:
                break

        if not results_all:
            print("Нет вопросов для прогона.", file=sys.stderr)
            return 1

        mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in results_all) / len(results_all)
        summary = {
            "settings": "settings.py",
            "corpus": str(corpus_path),
            "benchmark_mode": "multi",
            "benchmark": str(bench_path),
            "mode": mode,
            "n": len(results_all),
            "mean_recall_on_ground_truth_tokens": round(mean_recall, 4),
            "query_top_k": settings.QUERY_TOP_K,
            "query_chunk_top_k": settings.QUERY_CHUNK_TOP_K,
            "query_max_entity_tokens": settings.QUERY_MAX_ENTITY_TOKENS,
            "query_max_relation_tokens": settings.QUERY_MAX_RELATION_TOKENS,
            "query_max_total_tokens": settings.QUERY_MAX_TOTAL_TOKENS,
            "query_full_budget": getattr(settings, "USE_FULL_QUERY_BUDGET", False),
            "query_concurrency": int(getattr(settings, "QUERY_CONCURRENCY", 1)),
            "by_question_type": per_type_meta,
            "results_dir": str(res_dir),
        }
        write_json(res_dir / "_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Сводка по типам: {res_dir / '_summary.json'}")
    finally:
        await rag.finalize_storages()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
