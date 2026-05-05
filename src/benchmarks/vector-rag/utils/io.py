from __future__ import annotations

import json
from pathlib import Path


def resolve_working_dir(*, working_dir_setting: str, vector_rag_dir: Path) -> Path:
    p = Path(working_dir_setting).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (vector_rag_dir / p).resolve()


def resolve_output_path(
    *,
    output_file_setting: str,
    vector_rag_dir: Path,
    repo_root: Path,
) -> Path:
    if output_file_setting and str(output_file_setting).strip():
        p = Path(output_file_setting).expanduser()
        return p if p.is_absolute() else (vector_rag_dir / p).resolve()
    return (repo_root / "vector_benchmark_results.json").resolve()


def write_results(path: Path, summary: dict, items: list[dict]) -> None:
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


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
