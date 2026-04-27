"""
LightRAG: индексация текста и прогон бенчмарка.

Все параметры — константы в `settings.py`. Запуск:
  cd src/benchmarks/light-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py

Уже проиндексировано — только вопросы бенчмарка:
  (PowerShell)  $env:LIGHTRAG_QUERY_ONLY = "1"; uv run python main.py

После прогона опционально вызывается LLM-as-judge (accuracy); отключить:
  LIGHTRAG_ENABLE_LLM_JUDGE=0
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402

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


def _resolve_benchmark_path() -> Path:
    s = settings.BENCHMARK_FILE
    if not (s and str(s).strip()):
        return (REPO_ROOT / "graphrag_benchmark.json").resolve()
    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()
    a = (settings.LIGHT_RAG_DIR / p).resolve()
    if a.is_file():
        return a
    b = (REPO_ROOT / p).resolve()
    if b.is_file():
        return b
    return a


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


def _ideal_for_llm_judge(row: dict) -> str | None:
    """Тот же эталон, что для recall: ground_truth или ideal_for_scoring (subgraph)."""
    if row.get("scoring_reference") == "answer":
        ideal = row.get("ideal_for_scoring")
        if ideal is not None and str(ideal).strip():
            return str(ideal).strip()
        return None
    gt = row.get("ground_truth")
    if gt is None or not str(gt).strip():
        return None
    return str(gt).strip()


def _run_llm_accuracy_judge(results: list[dict], summary: dict) -> None:
    if not getattr(settings, "ENABLE_LLM_ACCURACY", True):
        summary["llm_accuracy"] = {"skipped": True, "reason": "ENABLE_LLM_ACCURACY=false"}
        return
    key = (settings.OPENROUTER_API_KEY or "").strip()
    if not key:
        summary["llm_accuracy"] = {"skipped": True, "reason": "no OPENROUTER_API_KEY"}
        return

    from llm_accuracy import judge_correct, judge_model, openai_client  # noqa: WPS433

    delay = float(getattr(settings, "METRICS_API_DELAY_SEC", 0.0))
    model = judge_model()
    try:
        client = openai_client()
    except Exception as e:  # noqa: BLE001
        summary["llm_accuracy"] = {"skipped": True, "reason": str(e)}
        return

    summary["llm_judge_model"] = model
    n_scored = 0
    n_correct = 0
    n_skipped_empty_ideal = 0

    for r in results:
        ideal = _ideal_for_llm_judge(r)
        if not ideal:
            n_skipped_empty_ideal += 1
            continue
        q = str(r.get("question", ""))
        ans = str(r.get("answer", ""))
        try:
            ok = judge_correct(client, model, q, ideal, ans)
            r["llm_accuracy_correct"] = ok
            r["llm_accuracy_error"] = None
            n_scored += 1
            if ok:
                n_correct += 1
            tag = "✓" if ok else "✗"
            print(f"  [judge] #{r.get('index')} accuracy {tag}", flush=True)
        except Exception as ex:  # noqa: BLE001
            r["llm_accuracy_correct"] = None
            r["llm_accuracy_error"] = str(ex)
            print(f"  [judge] #{r.get('index')} error: {ex}", flush=True)
        if delay > 0:
            time.sleep(delay)

    acc = round(n_correct / n_scored, 4) if n_scored else None
    summary["llm_accuracy"] = {
        "mean_accuracy": acc,
        "n_judged": n_scored,
        "n_correct": n_correct,
        "n_skipped_empty_ideal": n_skipped_empty_ideal,
    }


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


async def _run() -> int:
    from openrouter_lightrag import (  # noqa: WPS433
        apply_openrouter_env_defaults,
        build_rag,
        clear_working_dir,
        ensure_lightrag_available,
        indexing_llm_retry_scope,
    )
    from lightrag import QueryParam  # noqa: WPS433

    from raw_corpus import load_raw_text, resolved_corpus_path  # noqa: WPS433

    ensure_lightrag_available()
    apply_openrouter_env_defaults()

    bench_path = _resolve_benchmark_path()
    if not bench_path.is_file():
        print(f"Файл бенчмарка не найден: {bench_path}", file=sys.stderr)
        return 1

    with open(bench_path, encoding="utf-8") as f:
        items: list[dict] = json.load(f)
    limit = int(settings.LIMIT_QUESTIONS)
    if limit and limit > 0:
        items = items[:limit]

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
            print(f"Индексация успешна, вопросов в прогоне: {len(items)}")
        else:
            print(f"Вопросов в прогоне: {len(items)}")

        results: list[dict] = []
        for i, it in enumerate(items, 1):
            q = it.get("question", "")
            ground_truth = it.get("ground_truth")
            complexity = it.get("complexity", "")
            is_subgraph_deep = complexity == "subgraph-deep-analytics"
            benchmark_answer = str(it.get("answer") or "").strip()
            reference = (
                benchmark_answer
                if is_subgraph_deep
                else (str(ground_truth).strip() if ground_truth is not None else "")
            )

            try:
                rag_answer = await rag.aquery(
                    q,
                    param=QueryParam(mode=mode),
                )
            except Exception as e:  # noqa: BLE001
                rag_answer = f"[error] {e}"

            rdict = {
                "index": i,
                "complexity": complexity,
                "scoring_reference": "answer" if is_subgraph_deep else "ground_truth",
                "recall_on_ground_truth_tokens": round(
                    recall_overlap(reference, str(rag_answer)), 4
                ),
                "question": q,
                "ground_truth": ground_truth,
                "answer": rag_answer,
            }
            if is_subgraph_deep:
                rdict["ideal_for_scoring"] = benchmark_answer
            results.append(rdict)
            sc = rdict["recall_on_ground_truth_tokens"]
            ref_tag = rdict["scoring_reference"]
            print(
                f"  [{i}/{len(items)}] recall@{ref_tag}={sc:.3f}  {q[:70]}…"
                if len(q) > 70
                else f"  [{i}/{len(items)}] recall@{ref_tag}={sc:.3f}  {q}"
            )

        mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in results) / max(len(results), 1)
        summary = {
            "settings": "settings.py",
            "corpus": str(corpus_path),
            "benchmark": str(bench_path),
            "mode": mode,
            "n": len(results),
            "mean_recall_on_ground_truth_tokens": round(mean_recall, 4),
        }
        _run_llm_accuracy_judge(results, summary)
        out_path = _resolve_output_path()
        _write_results(out_path, summary, results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Результаты записаны: {out_path}")
    finally:
        await rag.finalize_storages()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
