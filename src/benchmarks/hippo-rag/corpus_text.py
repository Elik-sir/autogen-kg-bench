"""Загрузка сырого корпуса из `settings.CORPUS_FILE`."""

from __future__ import annotations

import re
from pathlib import Path

import settings


def resolved_corpus_path() -> Path:
    p = Path(settings.CORPUS_FILE).expanduser()
    if not p.is_absolute():
        p = (settings.HIPPO_RAG_DIR / p).resolve()
    return p


def load_corpus_chunks() -> list[str]:
    path = resolved_corpus_path()
    if not path.is_file():
        raise FileNotFoundError(f"Корпус не найден: {path}")
    text = path.read_text(encoding="utf-8")
    # HippoRAG принимает список документов; делим по пустым строкам.
    chunks = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
    if not chunks and text.strip():
        chunks = [text.strip()]
    return chunks
