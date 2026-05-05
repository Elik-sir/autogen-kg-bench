"""
HippoRAG: индексация корпуса и прогон вопросов бенчмарка.

Запуск:
  cd src/benchmarks/hippo-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py

Вопросы берутся из каталога `benchmark_questions_by_type/` (см. `utils.benchmark_by_type`).
Результаты — в `results/<тип>.<суффикс>` и сводка `results/_summary.json`.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
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
from utils.env import prepare_hipporag_env  # noqa: E402
from utils.io import resolve_working_dir, write_json, write_results  # noqa: E402
from utils.run import run_benchmark_chunk  # noqa: E402

build_benchmark_plan = _benchmark_by_type.build_benchmark_plan
output_suffix_from_setting = _benchmark_by_type.output_suffix_from_setting
results_subdir = _benchmark_by_type.results_subdir

REPO_ROOT = settings.HIPPO_RAG_DIR.parent.parent.parent


def run() -> int:
    from corpus_text import load_corpus_chunks, resolved_corpus_path  # noqa: WPS433

    try:
        from hipporag import HippoRAG  # noqa: WPS433
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) == "resource":
            print(
                "Не удалось импортировать HippoRAG на Windows: пакет vLLM внутри hipporag "
                "пытается импортировать Unix-модуль 'resource'.\n"
                "Что сделать:\n"
                "1) Рекомендуемо: запускать hippo-rag в WSL/Linux.\n"
                "2) Либо попробовать обновить vllm до версии с фиксом Windows-импорта "
                "('uv add vllm --upgrade') и повторить запуск.\n"
                "3) Если не поможет, остаётся WSL/Linux, так как vLLM официально ориентирован "
                "на Linux-окружение.",
                file=sys.stderr,
            )
            return 2
        raise

    plan = build_benchmark_plan(
        repo_root=REPO_ROOT,
        benchmark_pkg_dir=settings.HIPPO_RAG_DIR,
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
        docs = load_corpus_chunks()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not docs:
        print("Пустой корпус.", file=sys.stderr)
        return 1

    work = resolve_working_dir(
        working_dir_setting=settings.WORKING_DIR,
        hippo_rag_dir=settings.HIPPO_RAG_DIR,
    )
    if settings.REBUILD_INDEX and work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    prepare_hipporag_env()
    emb = settings.EMBEDDING_MODEL
    if "text-embedding" not in emb.lower():
        print(
            "HippoRAG: в имени модели эмбеддингов должна быть подстрока "
            "'text-embedding' (так библиотека включает OpenAI-совместимый клиент). "
            f"Сейчас: {emb!r}. Задайте HIPPORAG_EMBEDDING_MODEL, например "
            "'openai/text-embedding-3-small'.",
            file=sys.stderr,
        )
        return 2
    print(f"Индексация HippoRAG… docs={len(docs)}")
    print(f"LLM_MODEL: {settings.LLM_MODEL}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"OPENAI_API_BASE: {settings.OPENAI_API_BASE}")
    rag = HippoRAG(
        save_dir=str(work),
        llm_model_name=settings.LLM_MODEL,
        embedding_model_name=settings.EMBEDDING_MODEL,
        llm_base_url=settings.OPENAI_API_BASE,
        embedding_base_url=settings.OPENAI_API_BASE,
    )
    rag.index(docs=docs)

    print(f"Вопросов в прогоне: {run_total}")
    results: list[dict] = []
    per_type_meta: list[dict] = []
    rem = remaining
    sfx = output_suffix_from_setting(settings.OUTPUT_FILE)
    res_dir = results_subdir(settings.HIPPO_RAG_DIR)

    for complexity, bpath, raw_items in batches:
        chunk = raw_items
        if rem > 0:
            chunk = raw_items[:rem]
            rem -= len(chunk)
        if not chunk:
            continue
        type_key = complexity or "mixed"
        print(f"--- тип: {type_key} ({len(chunk)} вопросов) ---")
        batch_results = run_benchmark_chunk(rag, chunk)
        results.extend(batch_results)
        mean_b = sum(r["recall_on_ground_truth_tokens"] for r in batch_results) / max(
            len(batch_results), 1
        )
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
            "backend": "hipporag",
            "corpus": str(resolved_corpus_path()),
            "benchmark_mode": "multi",
            "question_type": type_key,
            "benchmark": str(bpath),
            "retrieval_k": settings.RETRIEVAL_K,
            "n": len(batch_results),
            "mean_recall_on_ground_truth_tokens": round(mean_b, 4),
        }
        write_results(res_dir / f"{type_key}{sfx}", type_summary, batch_results)
        print(f"Тип {type_key!r}: результаты → {res_dir / f'{type_key}{sfx}'}")
        if rem == 0 and remaining > 0:
            break

    if not results:
        print("Нет вопросов для прогона.", file=sys.stderr)
        return 1

    mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in results) / len(results)
    summary: dict = {
        "settings": "settings.py",
        "backend": "hipporag",
        "corpus": str(resolved_corpus_path()),
        "benchmark_mode": "multi",
        "benchmark": str(bench_path),
        "retrieval_k": settings.RETRIEVAL_K,
        "n": len(results),
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
