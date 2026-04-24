"""
LightRAG: индексация текста и прогон бенчмарка.

Все параметры — константы в `settings.py`. Запуск:
  cd src/benchmarks/light-rag
  uv sync
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py
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
    try:
        corpus = load_raw_text(corpus_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    if not corpus.strip():
        print("Файл корпуса пустой: нечего индексировать.", file=sys.stderr)
        return 1

    work = _resolve_working_dir()
    if settings.REBUILD_CACHE:
        clear_working_dir(work)
    work.mkdir(parents=True, exist_ok=True)

    rag = build_rag(work)
    await rag.initialize_storages()

    mode = str(settings.QUERY_MODE).lower()
    if mode not in ("naive", "local", "global", "hybrid"):
        mode = "hybrid"

    try:
        print(f"Индексация (ainsert), источник: {corpus_path} …")
        await rag.ainsert(corpus)
        print(f"Готово, вопросов в прогоне: {len(items)}")

        results: list[dict] = []
        for i, it in enumerate(items, 1):
            q = it.get("question", "")
            ground_truth = it.get("ground_truth")
            complexity = it.get("complexity", "")

            try:
                answer = await rag.aquery(
                    q,
                    param=QueryParam(mode=mode),
                )
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
                "answer": answer
            }
            results.append(rdict)
            sc = rdict["recall_on_ground_truth_tokens"]
            print(f"  [{i}/{len(items)}] recall={sc:.3f}  {q[:70]}…" if len(q) > 70 else f"  [{i}/{len(items)}] recall={sc:.3f}  {q}")

        mean_recall = sum(r["recall_on_ground_truth_tokens"] for r in results) / max(len(results), 1)
        summary = {
            "settings": "settings.py",
            "corpus": str(corpus_path),
            "benchmark": str(bench_path),
            "mode": mode,
            "n": len(results),
            "mean_recall_on_ground_truth_tokens": round(mean_recall, 4),
        }
        out_path = _resolve_output_path()
        _write_results(out_path, summary, results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Результаты записаны: {out_path}")
    finally:
        await rag.finalize_storages()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
