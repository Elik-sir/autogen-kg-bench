"""Загрузка вопросов бенчмарка по типам (simple → multi-hop → …) и пути к `results/`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Порядок прогона: сначала все simple, затем все multi-hop и т.д.
QUESTION_TYPE_ORDER: tuple[str, ...] = (
    "simple",
    "multi-hop",
    "aggregation",
    "cross-branch",
    "subgraph-deep-analytics",
)


def repo_root_from_benchmark_pkg(benchmark_pkg_dir: Path) -> Path:
    """.../src/benchmarks/<system> → корень репозитория."""
    return benchmark_pkg_dir.resolve().parent.parent.parent


def default_questions_dir(repo_root: Path) -> Path:
    return (repo_root / "benchmark_questions_by_type").resolve()


def resolve_benchmark_questions_dir(setting_value: str, repo_root: Path) -> Path:
    s = (setting_value or "").strip()
    if s:
        p = Path(s).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (repo_root / p).resolve()
    return default_questions_dir(repo_root)


def output_suffix_from_setting(output_file: str) -> str:
    suf = Path(output_file or "benchmark_data.jsonl").suffix.lower()
    return suf if suf in (".json", ".jsonl") else ".jsonl"


@dataclass(frozen=True)
class BenchmarkSource:
    mode: Literal["single", "multi"]
    # single: one path; items carry their own complexity
    single_path: Path | None = None
    # multi: ordered (complexity, path, items)
    multi_parts: tuple[tuple[str, Path, list[dict]], ...] = ()


def build_benchmark_plan(
    *,
    repo_root: Path,
    benchmark_pkg_dir: Path,
    benchmark_file_setting: str,
    benchmark_questions_dir_setting: str,
) -> BenchmarkSource:
    """
    Если задан существующий файл в BENCHMARK_FILE — один файл (как раньше).
    Иначе, если каталог benchmark_questions_by_type (или BENCHMARK_QUESTIONS_DIR)
    содержит json по типам — режим multi с прогоном в QUESTION_TYPE_ORDER.
    Иначе — один файл graphrag_benchmark.json в корне репо.
    """
    bf = (benchmark_file_setting or "").strip()
    if bf:
        p = Path(bf).expanduser()
        if not p.is_absolute():
            a = (benchmark_pkg_dir / p).resolve()
            p = a if a.is_file() else (repo_root / p).resolve()
        else:
            p = p.resolve()
        if p.is_file():
            return BenchmarkSource(mode="single", single_path=p)

    qdir = resolve_benchmark_questions_dir(benchmark_questions_dir_setting, repo_root)
    parts: list[tuple[str, Path, list[dict]]] = []
    for complexity in QUESTION_TYPE_ORDER:
        fp = qdir / f"{complexity}.json"
        if not fp.is_file():
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Ожидался JSON-массив в {fp}")
        items = [x for x in data if isinstance(x, dict)]
        if items:
            parts.append((complexity, fp, items))

    if parts:
        return BenchmarkSource(mode="multi", multi_parts=tuple(parts))

    mono = (repo_root / "graphrag_benchmark.json").resolve()
    return BenchmarkSource(mode="single", single_path=mono)


def results_subdir(benchmark_pkg_dir: Path) -> Path:
    d = (benchmark_pkg_dir / "results").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d
