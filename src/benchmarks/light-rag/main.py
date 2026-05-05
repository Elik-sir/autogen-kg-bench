"""
LightRAG: индексация текста и прогон бенчмарка.

Все параметры — константы в `settings.py`. Запуск:
  cd src/benchmarks/light-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py

Уже проиндексировано — только вопросы бенчмарка:
  (PowerShell)  $env:LIGHTRAG_QUERY_ONLY = "1"; uv run python main.py

В каждую запись результатов добавляется `contexts` — тексты чанков из retrieval
(LightRAG `aquery_llm` → `data.chunks`), чтобы RAGAS Faithfulness опиралась на
реальный контекст (см. `src/judge`, `load_eval_records`). Объём retrieval задаётся
в `settings.py` (`QUERY_*` / `LIGHTRAG_QUERY_*`), по умолчанию сужен для метрик.
Параллельные `aquery_llm`: `QUERY_CONCURRENCY` / env `LIGHTRAG_QUERY_CONCURRENCY` (по умолчанию 4).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402
from utils.benchmark_by_type import (  # noqa: E402
    build_benchmark_plan,
    output_suffix_from_setting,
    results_subdir,
)

REPO_ROOT = settings.LIGHT_RAG_DIR.parent.parent.parent


def _tokenize(s: str) -> set[str]:
    s = s.lower()
    return set(re.findall(r"[\w\.\-]+", s, re.UNICODE)) - {""}


def recall_overlap(ground_truth: str, answer: str) -> float:
    if not ground_truth.strip():
        return 1.0
    g = _tokenize(ground_truth)
    if not g:
        return 0.0
    a = _tokenize(answer) if answer else set()
    return len(g & a) / len(g)


def _contexts_from_aquery_llm(full: dict) -> list[str]:
    """Тексты чанков из `LightRAG.aquery_llm` (`data.chunks[].content`) для RAGAS."""
    data = full.get("data")
    if not isinstance(data, dict):
        return []
    chunks = data.get("chunks") or []
    if not isinstance(chunks, list):
        return []
    out: list[str] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        raw = ch.get("content")
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            out.append(s)
    return out


def _answer_from_aquery_llm(full: dict) -> str:
    lr = full.get("llm_response") or {}
    if lr.get("is_streaming"):
        return ""
    c = lr.get("content")
    return "" if c is None else str(c)


def _benchmark_query_param(mode: str):
    """Параметры запроса: те же лимиты, что режут контекст для ответа и для `contexts`."""
    from lightrag import QueryParam  # noqa: WPS433

    return QueryParam(
        mode=mode,
        stream=False,
        top_k=int(settings.QUERY_TOP_K),
        chunk_top_k=int(settings.QUERY_CHUNK_TOP_K),
        max_entity_tokens=int(settings.QUERY_MAX_ENTITY_TOKENS),
        max_relation_tokens=int(settings.QUERY_MAX_RELATION_TOKENS),
        max_total_tokens=int(settings.QUERY_MAX_TOTAL_TOKENS),
    )


async def _run_single_benchmark_query(
    rag: object,
    qp: object,
    semaphore: asyncio.Semaphore,
    print_lock: asyncio.Lock,
    *,
    index: int,
    item: dict,
    n_items: int,
) -> dict:
    """Один вопрос: `aquery_llm` под семафором, печать прогресса под lock."""
    q = item.get("question", "")
    ground_truth = item.get("ground_truth")
    complexity = item.get("complexity", "")
    is_subgraph_deep = complexity == "subgraph-deep-analytics"
    benchmark_answer = str(item.get("answer") or "").strip()
    reference = (
        benchmark_answer
        if is_subgraph_deep
        else (str(ground_truth).strip() if ground_truth is not None else "")
    )

    rag_contexts: list[str] = []
    rag_answer = ""
    async with semaphore:
        try:
            full = await rag.aquery_llm(q, param=qp)
            rag_answer = _answer_from_aquery_llm(full)
            rag_contexts = _contexts_from_aquery_llm(full)
        except Exception as e:  # noqa: BLE001
            rag_answer = f"[error] {e}"

    rdict = {
        "index": index,
        "complexity": complexity,
        "scoring_reference": "answer" if is_subgraph_deep else "ground_truth",
        "recall_on_ground_truth_tokens": round(
            recall_overlap(reference, str(rag_answer)), 4
        ),
        "question": q,
        "ground_truth": ground_truth,
        "answer": rag_answer,
        "contexts": rag_contexts,
    }
    if is_subgraph_deep:
        rdict["ideal_for_scoring"] = benchmark_answer

    sc = rdict["recall_on_ground_truth_tokens"]
    ref_tag = rdict["scoring_reference"]
    line = (
        f"  [{index}/{n_items}] recall@{ref_tag}={sc:.3f}  {q[:70]}…"
        if len(q) > 70
        else f"  [{index}/{n_items}] recall@{ref_tag}={sc:.3f}  {q}"
    )
    async with print_lock:
        print(line, flush=True)

    return rdict


def _resolve_working_dir() -> Path:
    s = settings.WORKING_DIR
    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (settings.LIGHT_RAG_DIR / p).resolve()


def _resolve_output_path() -> Path:
    s = settings.OUTPUT_FILE
    if s and str(s).strip():
        p = Path(s).expanduser()
        return p if p.is_absolute() else (settings.LIGHT_RAG_DIR / p).resolve()
    return (REPO_ROOT / "lightrag_benchmark_results.json").resolve()


async def _require_indexing_complete(rag) -> str | None:
    """Если индексация не дошла до успешного статуса, вернуть сообщение об ошибке."""
    from lightrag.base import DocStatus  # noqa: WPS433

    failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
    if failed:
        lines = [
            f"  {doc_id}: {(st.error_msg or str(st.status)).strip()}"
            for doc_id, st in failed.items()
        ]
        return (
            f"Индексация не завершена: {len(failed)} документ(ов) в статусе FAILED "
            f"(см. LIGHTRAG_LLM_TIMEOUT_SEC в settings.py, сейчас {settings.LLM_TIMEOUT_SEC}s).\n"
            + "\n".join(lines)
        )

    incomplete = await rag.doc_status.get_docs_by_statuses(
        [DocStatus.PENDING, DocStatus.PROCESSING]
    )
    if incomplete:
        lines = [f"  {doc_id}: {st.status}" for doc_id, st in incomplete.items()]
        return (
            "Индексация не завершена: остались документы PENDING/PROCESSING:\n"
            + "\n".join(lines)
        )
    return None


def _write_results(path: Path, summary: dict, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        lines = [json.dumps({"kind": "summary", **summary}, ensure_ascii=False)]
        lines += [json.dumps({"kind": "item", **it}, ensure_ascii=False) for it in items]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(
            json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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

    work = _resolve_working_dir()
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
            indexing_err = await _require_indexing_complete(rag)
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
        _qp = _benchmark_query_param(mode)
        _conc = int(getattr(settings, "QUERY_CONCURRENCY", 1))
        print(f"Параллельных запросов к RAG (aquery_llm): {_conc}", flush=True)

        _sem = asyncio.Semaphore(_conc)
        _plock = asyncio.Lock()
        results_all: list[dict] = []
        per_type_meta: list[dict] = []
        rem = remaining
        sfx = output_suffix_from_setting(settings.OUTPUT_FILE) if plan.mode == "multi" else ""
        res_dir = results_subdir(settings.LIGHT_RAG_DIR) if plan.mode == "multi" else None

        for complexity, bpath, raw_items in batches:
            chunk = raw_items
            if rem > 0:
                chunk = raw_items[:rem]
                rem -= len(chunk)
            if not chunk:
                continue
            n_items = len(chunk)
            _tasks = [
                _run_single_benchmark_query(
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
            if plan.mode == "multi" and res_dir is not None:
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
                _write_results(res_dir / f"{type_key}{sfx}", type_summary, batch_results)
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
            "benchmark_mode": plan.mode,
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
        }
        if plan.mode == "single":
            out_path = _resolve_output_path()
            _write_results(out_path, summary, results_all)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Результаты записаны: {out_path}")
        else:
            assert res_dir is not None
            summary["results_dir"] = str(res_dir)
            _write_json(res_dir / "_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Сводка по типам: {res_dir / '_summary.json'}")
    finally:
        await rag.finalize_storages()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
