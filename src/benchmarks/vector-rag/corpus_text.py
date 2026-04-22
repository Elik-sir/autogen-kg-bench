"""Загрузка сырого текста корпуса из `settings.CORPUS_FILE`."""

from __future__ import annotations

from pathlib import Path

import settings


def resolved_corpus_path() -> Path:
    rel = settings.CORPUS_FILE
    p = Path(rel).expanduser()
    if not p.is_absolute():
        p = (settings.VECTOR_RAG_DIR / p).resolve()
    return p


def load_corpus() -> str:
    path = resolved_corpus_path()
    if not path.is_file():
        raise FileNotFoundError(f"Корпус не найден: {path}")
    return path.read_text(encoding="utf-8")
