"""Загрузка вопросов бенчмарка по типам (simple → multi-hop → …) и пути к `results/`."""

from __future__ import annotations

import json
from pathlib import Path

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


class BenchmarkSource:
    def __init__(self, multi_parts: tuple[tuple[str, Path, list[dict]], ...]) -> None:
        self.mode = "multi"
        self.multi_parts = multi_parts


def build_benchmark_plan(
    *,
    repo_root: Path,
    benchmark_pkg_dir: Path,
    benchmark_file_setting: str,
    benchmark_questions_dir_setting: str,
) -> BenchmarkSource:
    """
    Загружает вопросы только в multi-режиме:
    каталог benchmark_questions_by_type (или BENCHMARK_QUESTIONS_DIR)
    содержит json по типам с прогоном в QUESTION_TYPE_ORDER.
    """
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

    return BenchmarkSource(multi_parts=tuple(parts))


def results_subdir(benchmark_pkg_dir: Path) -> Path:
    d = (benchmark_pkg_dir / "results").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d
