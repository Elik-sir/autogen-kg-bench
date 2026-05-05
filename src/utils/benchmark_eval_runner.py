"""
Прогон eval по файлам результатов бенчмарка (.jsonl / .json) в каталоге ``results/``.

Пишет отчёты в ``<output_dir>/<stem>.eval.json`` и ``_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from utils.benchmark_eval_metrics import aggregate_metrics, compute_item_metrics


def is_benchmark_result_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in {".jsonl", ".json"}:
        return False
    if path.name.startswith("_"):
        return False
    if path.name.endswith(".eval.json"):
        return False
    return True


def discover_benchmark_result_files(results_dir: Path) -> list[Path]:
    files = [path for path in results_dir.iterdir() if is_benchmark_result_file(path)]
    return sorted(files)


def read_jsonl_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if kind == "summary":
            summary = row
            continue
        if kind == "item" or ("question" in row and "answer" in row):
            items.append(row)
    return summary, items


def read_json_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        summary = payload.get("summary", {})
        items = payload.get("items", [])
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(items, list):
            items = []
        return summary, [item for item in items if isinstance(item, dict)]
    if isinstance(payload, list):
        return {}, [item for item in payload if isinstance(item, dict)]
    return {}, []


def load_benchmark_result_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl_result(path)
    return read_json_result(path)


def write_eval_report_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_benchmark_eval(*, results_dir: Path, output_dir: Path | None = None) -> int:
    """
    Считает метрики по всем подходящим файлам в ``results_dir``.
    ``output_dir`` по умолчанию: ``results_dir / "eval"``.
    """
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        print(f"Results dir not found: {results_dir}", file=sys.stderr)
        return 1

    out = (output_dir.expanduser().resolve() if output_dir else (results_dir / "eval").resolve())
    out.mkdir(parents=True, exist_ok=True)

    result_files = discover_benchmark_result_files(results_dir)
    if not result_files:
        print(f"No result files found in: {results_dir}", file=sys.stderr)
        return 1

    total_items = 0
    total_metric_rows: list[dict[str, float]] = []
    per_file_rows: list[dict[str, Any]] = []

    print(f"Evaluating {len(result_files)} result file(s) from: {results_dir}")
    for result_file in result_files:
        source_summary, items = load_benchmark_result_file(result_file)
        if not items:
            print(f"Skip {result_file.name}: no items")
            continue

        evaluated_items: list[dict[str, Any]] = []
        metric_rows: list[dict[str, float]] = []
        for item in items:
            metric_row = compute_item_metrics(item)
            metric_rows.append(metric_row)
            evaluated_items.append(
                {
                    "index": item.get("index"),
                    "complexity": item.get("complexity"),
                    "question": item.get("question"),
                    "metrics": metric_row,
                }
            )

        file_mean_metrics = aggregate_metrics(metric_rows)
        file_report = {
            "source_file": str(result_file),
            "n": len(evaluated_items),
            "mean_metrics": file_mean_metrics,
            "source_summary": source_summary,
            "items": evaluated_items,
        }
        report_path = out / f"{result_file.stem}.eval.json"
        write_eval_report_json(report_path, file_report)
        print(f"  {result_file.name}: n={len(evaluated_items)} -> {report_path.name}")

        per_file_rows.append(
            {
                "source_file": str(result_file),
                "report_file": str(report_path),
                "n": len(evaluated_items),
                "mean_metrics": file_mean_metrics,
            }
        )
        total_items += len(evaluated_items)
        total_metric_rows.extend(metric_rows)

    if total_items == 0:
        print("No evaluable items found.", file=sys.stderr)
        return 1

    overall_summary = {
        "results_dir": str(results_dir),
        "reports_dir": str(out),
        "n_files": len(per_file_rows),
        "n_items": total_items,
        "mean_metrics": aggregate_metrics(total_metric_rows),
        "files": per_file_rows,
    }
    summary_path = out / "_summary.json"
    write_eval_report_json(summary_path, overall_summary)
    print(json.dumps(overall_summary["mean_metrics"], ensure_ascii=False, indent=2))
    print(f"Evaluation summary: {summary_path}")
    return 0
