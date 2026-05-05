"""
Метрики Comprehensiveness, Diversity, Empowerment (CDE) по файлу `benchmark_data.jsonl`.

Протокол: LLM-судья (промпты и парсинг — `utils.eval`).

Запуск (из папки light-rag, с PYTHONPATH=../..):
  python cde_metrics.py
  python cde_metrics.py --input benchmark_data.jsonl --output benchmark_cde.json

Модель судьи: settings.METRICS_JUDGE_MODEL или settings.LLM_MODEL, ключ — как в main.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_LIGHT_RAG = Path(__file__).resolve().parent
_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402

from utils.eval import (  # noqa: E402
    CDE_JUDGE_SYSTEM,
    CdeScores,
    openai_client_for_judge,
    resolve_judge_model,
    score_cde_answer,
)

# Совместимость со старым API модуля
score_answer = score_cde_answer
JUDGE_SYSTEM = CDE_JUDGE_SYSTEM


def load_items_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") == "summary":
            continue
        if row.get("kind") == "item" or (
            "question" in row and "answer" in row and "index" in row
        ):
            rows.append(row)
    return rows


def run_cde(
    input_path: Path,
    output_path: Path,
    delay_sec: float | None = None,
) -> dict[str, Any]:
    if delay_sec is None:
        delay_sec = float(getattr(settings, "METRICS_API_DELAY_SEC", 0.0))
    model = resolve_judge_model(settings)
    client = openai_client_for_judge(settings)
    items = load_items_from_jsonl(input_path)
    if not items:
        raise ValueError(f"Нет записей kind=item в {input_path}")

    per_index: list[dict[str, Any]] = []
    for row in items:
        idx = row.get("index", len(per_index) + 1)
        q = str(row.get("question", ""))
        ideal = str(row.get("ideal_for_scoring") or row.get("ground_truth") or "")
        ans = str(row.get("answer", ""))
        try:
            s = score_cde_answer(client, model, q, ideal, ans, row)
            entry = {
                "index": idx,
                "comprehensiveness": s.comprehensiveness,
                "diversity": s.diversity,
                "empowerment": s.empowerment,
                "error": None,
            }
        except Exception as ex:  # noqa: BLE001
            entry = {
                "index": idx,
                "comprehensiveness": None,
                "diversity": None,
                "empowerment": None,
                "error": str(ex),
            }
        per_index.append(entry)
        print(f"  [CDE] index {idx} → {entry.get('comprehensiveness', '?')} / "
              f"{entry.get('diversity', '?')} / {entry.get('empowerment', '?')}", flush=True)
        if delay_sec > 0:
            time.sleep(delay_sec)

    valid = [p for p in per_index if p.get("error") is None and p.get("comprehensiveness") is not None]
    n = len(valid)
    if n == 0:
        means = None
    else:
        means = {
            "comprehensiveness": sum(p["comprehensiveness"] for p in valid) / n,
            "diversity": sum(p["diversity"] for p in valid) / n,
            "empowerment": sum(p["empowerment"] for p in valid) / n,
        }

    report = {
        "source_file": str(input_path.resolve()),
        "judge_model": model,
        "n_scored": n,
        "n_errors": len(per_index) - n,
        "mean": means,
        "items": per_index,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDE-метрики по benchmark_data.jsonl")
    p.add_argument(
        "--input",
        type=Path,
        default=_LIGHT_RAG / "benchmark_data.jsonl",
        help="JSONL с kind=item (question, ground_truth, answer)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_LIGHT_RAG / "benchmark_cde_metrics.json",
        help="Куда записать агрегат и оценки по индексам",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Пауза между запросами (сек); по умолчанию METRICS_API_DELAY_SEC в settings",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    if not args.input.is_file():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 1
    try:
        report = run_cde(args.input, args.output, delay_sec=args.delay)
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"mean": report["mean"], "n_scored": report["n_scored"]}, ensure_ascii=False))
    print(f"Сохранено: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
