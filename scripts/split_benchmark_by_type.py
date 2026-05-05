"""Собрать benchmark_questions_by_type/*.json из graphrag_benchmark.json."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_ORDER = (
    "simple",
    "multi-hop",
    "aggregation",
    "cross-branch",
    "subgraph-deep-analytics",
)


def main() -> int:
    src = ROOT / "graphrag_benchmark.json"
    if not src.is_file():
        print(f"Нет файла: {src}", file=sys.stderr)
        return 1
    with open(src, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        print("graphrag_benchmark.json: ожидался массив объектов.", file=sys.stderr)
        return 1

    by_type: dict[str, list[dict]] = defaultdict(list)
    unknown: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        c = str(row.get("complexity") or "").strip()
        if not c:
            unknown.append("(пустой complexity)")
            continue
        if c not in SRC_ORDER:
            unknown.append(c)
        by_type[c].append(row)

    if unknown:
        uniq = sorted(set(unknown))
        print(f"Предупреждение: нестандартные complexity: {uniq}", file=sys.stderr)

    out_dir = ROOT / "benchmark_questions_by_type"
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in SRC_ORDER:
        chunk = by_type.get(c, [])
        path = out_dir / f"{c}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(chunk)} вопросов")

    extra = [k for k in by_type if k not in SRC_ORDER]
    for k in sorted(extra):
        path = out_dir / f"{k}.json"
        path.write_text(json.dumps(by_type[k], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(by_type[k])} вопросов (вне порядка QUESTION_TYPE_ORDER)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
