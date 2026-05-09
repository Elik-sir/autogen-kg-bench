"""
Удалить из benchmark_questions_by_type/*.json элементы массива, у которых
поле ``answer`` (как строка) содержит заданную подстроку (по умолчанию Neo4j elementId).

  python scripts/filter_benchmark_answers.py
  python scripts/filter_benchmark_answers.py --dry-run
  python scripts/filter_benchmark_answers.py --substring "4:002aa201-35fa-46a4-8661-72e4560b6259"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "benchmark_questions_by_type"
DEFAULT_SUBSTRING = "4:002aa201-35fa-46a4-8661-72e4560b6259"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Каталог с json (по умолчанию {DEFAULT_DIR})",
    )
    ap.add_argument(
        "--substring",
        default=DEFAULT_SUBSTRING,
        help="Подстрока в поле answer; записи с совпадением удаляются",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Только печать статистики, файлы не перезаписывать",
    )
    args = ap.parse_args()
    bench_dir: Path = args.dir.resolve()
    needle: str = str(args.substring)
    if not needle:
        print("Пустая --substring.", file=sys.stderr)
        return 1
    if not bench_dir.is_dir():
        print(f"Нет каталога: {bench_dir}", file=sys.stderr)
        return 1

    removed_total = 0
    for path in sorted(bench_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"Пропуск (не массив): {path}", file=sys.stderr)
            continue
        kept: list = []
        removed = 0
        for item in data:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            ans = item.get("answer")
            if ans is not None and needle in str(ans):
                removed += 1
                continue
            kept.append(item)
        if removed:
            print(f"{path.name}: удалено {removed}, осталось {len(kept)}")
            removed_total += removed
            if not args.dry_run:
                path.write_text(
                    json.dumps(kept, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
    if removed_total == 0:
        print("Ни одной записи с такой подстрокой в answer не найдено.")
    else:
        print(f"Всего удалено записей: {removed_total}" + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
