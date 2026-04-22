"""Путь к корпусу и чтение текста — из `settings.CORPUS_FILE`."""

from __future__ import annotations

from pathlib import Path

import settings


def resolved_corpus_path() -> Path:
    rel = settings.CORPUS_FILE
    p = Path(rel).expanduser()
    if not p.is_absolute():
        p = (settings.LIGHT_RAG_DIR / p).resolve()
    return p


def load_raw_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Текстовый корпус не найден: {path}")
    return path.read_text(encoding="utf-8")
