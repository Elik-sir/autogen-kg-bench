"""
Считает метрики (context_recall, faithfulness, …) по файлам в ``results/``.

Запуск:
  cd src/benchmarks/light-rag
  (PowerShell)  $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python eval/main.py

Отчёты: ``results/eval/<тип>.eval.json``, ``results/eval/_summary.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_LR_DIR = _EVAL_DIR.parent
_SRC = _LR_DIR.parent.parent
# uv / PYTHONPATH могут уже добавить ``src``, но позже локальный ``utils`` бенчмарка;
# убираем дубликаты и ставим репозиторийный ``src`` первым.
for _p in (str(_LR_DIR), str(_SRC)):
    if _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_SRC))
sys.path.insert(1, str(_LR_DIR))

import settings  # noqa: E402
from utils.benchmark_eval_runner import run_benchmark_eval  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate light-rag result files.")
    parser.add_argument(
        "--results-dir",
        default=str((settings.LIGHT_RAG_DIR / "results").resolve()),
        help="Directory with benchmark result files (.jsonl/.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for evaluation reports (default: <results-dir>/eval).",
    )
    return parser.parse_args()


def run() -> int:
    args = _parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else None
    )
    return run_benchmark_eval(results_dir=results_dir, output_dir=output_dir)


if __name__ == "__main__":
    raise SystemExit(run())
