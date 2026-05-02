"""
HippoRAG: индексация корпуса и прогон graphrag_benchmark.json.

Запуск:
  cd src/benchmarks/hippo-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402

REPO_ROOT = settings.HIPPO_RAG_DIR.parent.parent.parent


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
    a = (settings.HIPPO_RAG_DIR / p).resolve()
    if a.is_file():
        return a
    b = (REPO_ROOT / p).resolve()
    if b.is_file():
        return b
    return a


def _resolve_output_path() -> Path:
    s = settings.OUTPUT_FILE
    if s and str(s).strip():
        p = Path(s).expanduser()
        return p if p.is_absolute() else (settings.HIPPO_RAG_DIR / p).resolve()
    return (REPO_ROOT / "hipporag_benchmark_results.json").resolve()


def _resolve_working_dir() -> Path:
    p = Path(settings.WORKING_DIR).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (settings.HIPPO_RAG_DIR / p).resolve()


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


def _ideal_for_llm_judge(row: dict) -> str | None:
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
    key = (settings.OPENAI_API_KEY or "").strip()
    if not key:
        summary["llm_accuracy"] = {"skipped": True, "reason": "no OPENAI_API_KEY"}
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
            print(f"  [judge] #{r.get('index')} accuracy {'✓' if ok else '✗'}", flush=True)
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


def _prepare_hipporag_env() -> None:
    """Ключ и заголовки для OpenAI-совместимых клиентов (OpenRouter) внутри hipporag."""
    key = (settings.OPENAI_API_KEY or "").strip()
    if not key:
        key = (os.environ.get("OPENROUTER_API_KEY", "") or "").strip()
    os.environ["OPENAI_API_KEY"] = key
    if not (os.environ.get("OPENROUTER_API_KEY", "") or "").strip():
        os.environ["OPENROUTER_API_KEY"] = key
    os.environ.setdefault("OPENAI_BASE_URL", settings.OPENAI_API_BASE)
    if settings.OPENROUTER_HTTP_REFERER.strip():
        os.environ["OPENROUTER_HTTP_REFERER"] = settings.OPENROUTER_HTTP_REFERER.strip()
    if settings.OPENROUTER_APP_TITLE.strip():
        os.environ["OPENROUTER_APP_TITLE"] = settings.OPENROUTER_APP_TITLE.strip()


def _extract_answer_from_result(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    # HippoRAG.rag_qa возвращает (solutions, llm_messages, metadata[, ...]) — не str(dict).
    if isinstance(raw, tuple) and raw:
        solutions = raw[0]
        messages = raw[1] if len(raw) > 1 else None
        if isinstance(solutions, list) and solutions:
            sol0 = solutions[0]
            if hasattr(sol0, "answer"):
                ans = getattr(sol0, "answer", None)
                if ans is not None and str(ans).strip():
                    return str(ans).strip()
        if isinstance(messages, list) and messages:
            msg = messages[0]
            if isinstance(msg, str) and msg.strip():
                parts = msg.split("Answer:", 1)
                if len(parts) > 1:
                    return parts[1].strip()
                return msg.strip()
        return ""
    if isinstance(raw, list):
        if not raw:
            return ""
        first = raw[0]
        if hasattr(first, "answer"):
            ans = getattr(first, "answer", None)
            if ans is not None and str(ans).strip():
                return str(ans).strip()
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("answer", "response", "generated_text", "text", "prediction"):
                val = first.get(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
            return json.dumps(first, ensure_ascii=False)
        return str(first).strip()
    if isinstance(raw, dict):
        for key in ("answer", "response", "generated_text", "text", "prediction"):
            val = raw.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return json.dumps(raw, ensure_ascii=False)
    return str(raw).strip()


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

    bench_path = _resolve_benchmark_path()
    if not bench_path.is_file():
        print(f"Файл бенчмарка не найден: {bench_path}", file=sys.stderr)
        return 1
    with open(bench_path, encoding="utf-8") as f:
        items: list[dict] = json.load(f)
    limit = int(settings.LIMIT_QUESTIONS)
    if limit and limit > 0:
        items = items[:limit]

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

    results: list[dict] = []
    print(f"Вопросов в прогоне: {len(items)}")
    for i, it in enumerate(items, 1):
        q = str(it.get("question", ""))
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
            qa_raw = rag.rag_qa(queries=[q])
            answer = _extract_answer_from_result(qa_raw)
        except Exception as e:  # noqa: BLE001
            answer = f"[error] {e}"

        row = {
            "index": i,
            "complexity": complexity,
            "scoring_reference": "answer" if is_subgraph_deep else "ground_truth",
            "recall_on_ground_truth_tokens": round(recall_overlap(reference, str(answer)), 4),
            "question": q,
            "ground_truth": ground_truth,
            "answer": answer,
            "llm_accuracy_correct": None,
            "llm_accuracy_error": None,
        }
        if is_subgraph_deep:
            row["ideal_for_scoring"] = benchmark_answer
        results.append(row)
        sc = row["recall_on_ground_truth_tokens"]
        ref_tag = row["scoring_reference"]
        print(
            f"  [{i}/{len(items)}] recall@{ref_tag}={sc:.3f}  {q[:70]}…"
            if len(q) > 70
            else f"  [{i}/{len(items)}] recall@{ref_tag}={sc:.3f}  {q}"
        )

    mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in results) / max(len(results), 1)
    summary: dict = {
        "settings": "settings.py",
        "backend": "hipporag",
        "corpus": str(resolved_corpus_path()),
        "benchmark": str(bench_path),
        "retrieval_k": settings.RETRIEVAL_K,
        "n": len(results),
        "mean_recall_on_ground_truth_tokens": round(mean_recall, 4),
    }

    _run_llm_accuracy_judge(results, summary)
    out_path = _resolve_output_path()
    _write_results(out_path, summary, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Результаты: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
