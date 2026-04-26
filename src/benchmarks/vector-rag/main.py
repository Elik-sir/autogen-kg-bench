"""
Векторный RAG (LangChain: FAISS + top-k + LLM) и прогон `graphrag_benchmark.json`.

  cd src/benchmarks/vector-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py

Положите `corpus.txt` (или путь в settings.CORPUS_FILE) рядом или скопируйте из light-rag.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402

REPO_ROOT = settings.VECTOR_RAG_DIR.parent.parent.parent


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
    a = (settings.VECTOR_RAG_DIR / p).resolve()
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
    return (settings.VECTOR_RAG_DIR / p).resolve()


def _resolve_output_path() -> Path:
    s = settings.OUTPUT_FILE
    if s and str(s).strip():
        p = Path(s).expanduser()
        return p if p.is_absolute() else (settings.VECTOR_RAG_DIR / p).resolve()
    return (REPO_ROOT / "vector_benchmark_results.json").resolve()


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
    n_skipped_empty_gt = 0

    for r in results:
        gt = r.get("ground_truth")
        if gt is None or not str(gt).strip():
            n_skipped_empty_gt += 1
            continue
        q = str(r.get("question", ""))
        ans = str(r.get("answer", ""))
        try:
            ok = judge_correct(client, model, q, str(gt), ans)
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
        "n_skipped_empty_ground_truth": n_skipped_empty_gt,
    }


def run() -> int:
    from corpus_text import load_corpus, resolved_corpus_path  # noqa: WPS433
    from vector_rag import answer_from_store, build_or_load_vectorstore  # noqa: WPS433

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
        corpus = load_corpus()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    if not corpus.strip():
        print("Пустой корпус.", file=sys.stderr)
        return 1

    work = _resolve_working_dir()
    print(f"Индексация FAISS (langchain)… chunk={settings.CHUNK_SIZE}, k={settings.RETRIEVAL_K}")
    store = build_or_load_vectorstore(corpus, work)
    corpus_path = str(resolved_corpus_path())
    print(f"Готово, вопросов: {len(items)}")

    results: list[dict] = []
    for i, it in enumerate(items, 1):
        q = it.get("question", "")
        ground_truth = it.get("ground_truth")
        complexity = it.get("complexity", "")
        try:
            answer = answer_from_store(store, q)
        except Exception as e:  # noqa: BLE001
            answer = f"[error] {e}"
        rdict = {
            "index": i,
            "complexity": complexity,
            "recall_on_ground_truth_tokens": round(
                recall_overlap(ground_truth, str(answer)), 4
            ),
            "question": q,
            "ground_truth": ground_truth,
            "answer": answer,
            "llm_accuracy_correct": None,
            "llm_accuracy_error": None,
        }
        results.append(rdict)
        sc = rdict["recall_on_ground_truth_tokens"]
        print(
            f"  [{i}/{len(items)}] recall={sc:.3f}  {q[:70]}…"
            if len(q) > 70
            else f"  [{i}/{len(items)}] recall={sc:.3f}  {q}"
        )

    mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in results) / max(len(results), 1)
    summary: dict = {
        "settings": "settings.py",
        "backend": "vector-rag-langchain-faiss",
        "corpus": corpus_path,
        "benchmark": str(bench_path),
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
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
