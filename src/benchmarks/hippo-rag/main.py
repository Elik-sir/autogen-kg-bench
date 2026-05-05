"""
HippoRAG: индексация корпуса и прогон graphrag_benchmark.json.

Запуск:
  cd src/benchmarks/hippo-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402
from bench_utils import (  # noqa: E402
    _extract_answer_from_result,
    _extract_contexts_from_rag_qa,
    _prepare_hipporag_env,
    _resolve_output_path,
    _resolve_working_dir,
    _write_results,
    recall_overlap,
)
from utils.benchmark_by_type import (  # noqa: E402
    build_benchmark_plan,
    output_suffix_from_setting,
    results_subdir,
)

REPO_ROOT = settings.HIPPO_RAG_DIR.parent.parent.parent


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    if plan.mode == "single":
        bench_path = plan.single_path
        assert bench_path is not None
        if not bench_path.is_file():
            print(f"Файл бенчмарка не найден: {bench_path}", file=sys.stderr)
            return 1
        with open(bench_path, encoding="utf-8") as f:
            all_benchmark_items: list[dict] = json.load(f)
        batches: list[tuple[str | None, Path, list[dict]]] = [(None, bench_path, all_benchmark_items)]
    else:
        batches = [(c, p, its) for c, p, its in plan.multi_parts]
        if not batches:
            print("Нет файлов в benchmark_questions_by_type и нет graphrag_benchmark.json.", file=sys.stderr)
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

    work = _resolve_working_dir()
    if settings.REBUILD_INDEX and work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    _prepare_hipporag_env()
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

    def run_chunk(chunk: list[dict]) -> list[dict]:
        rows: list[dict] = []
        n = len(chunk)
        for i, it in enumerate(chunk, 1):
            q = str(it.get("question", ""))
            ground_truth = it.get("ground_truth")
            complexity = str(it.get("complexity", "") or "")
            is_subgraph_deep = complexity == "subgraph-deep-analytics"
            benchmark_answer = str(it.get("answer") or "").strip()
            reference = (
                benchmark_answer
                if is_subgraph_deep
                else (str(ground_truth).strip() if ground_truth is not None else "")
            )
            try:
                qa_raw = rag.rag_qa(queries=[q])
                answer = _extract_answer_from_result(qa_raw)
                contexts = _extract_contexts_from_rag_qa(
                    qa_raw, int(getattr(rag.global_config, "qa_top_k", 5))
                )
            except Exception as e:  # noqa: BLE001
                answer = f"[error] {e}"
                contexts = []

            row = {
                "index": i,
                "complexity": complexity,
                "scoring_reference": "answer" if is_subgraph_deep else "ground_truth",
                "recall_on_ground_truth_tokens": round(recall_overlap(reference, str(answer)), 4),
                "question": q,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
            }
            if is_subgraph_deep:
                row["ideal_for_scoring"] = benchmark_answer
            rows.append(row)
            sc = row["recall_on_ground_truth_tokens"]
            ref_tag = row["scoring_reference"]
            print(
                f"  [{i}/{n}] recall@{ref_tag}={sc:.3f}  {q[:70]}…"
                if len(q) > 70
                else f"  [{i}/{n}] recall@{ref_tag}={sc:.3f}  {q}"
            )
        return rows

    print(f"Вопросов в прогоне: {run_total}")
    results: list[dict] = []
    per_type_meta: list[dict] = []
    rem = remaining
    sfx = output_suffix_from_setting(settings.OUTPUT_FILE) if plan.mode == "multi" else ""
    res_dir = results_subdir(settings.HIPPO_RAG_DIR) if plan.mode == "multi" else None

    for complexity, bpath, raw_items in batches:
        chunk = raw_items
        if rem > 0:
            chunk = raw_items[:rem]
            rem -= len(chunk)
        if not chunk:
            continue
        type_key = complexity or "mixed"
        print(f"--- тип: {type_key} ({len(chunk)} вопросов) ---")
        batch_results = run_chunk(chunk)
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
        if plan.mode == "multi" and res_dir is not None:
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
            _write_results(res_dir / f"{type_key}{sfx}", type_summary, batch_results)
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
        "benchmark_mode": plan.mode,
        "benchmark": str(bench_path),
        "retrieval_k": settings.RETRIEVAL_K,
        "n": len(results),
        "mean_recall_on_ground_truth_tokens": round(mean_recall, 4),
        "by_question_type": per_type_meta,
    }
    if plan.mode == "single":
        out_path = _resolve_output_path()
        _write_results(out_path, summary, results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Результаты: {out_path}")
    else:
        assert res_dir is not None
        summary["results_dir"] = str(res_dir)
        _write_json(res_dir / "_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Сводка по типам: {res_dir / '_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
