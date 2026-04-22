"""
Константы бенчмарка векторного RAG. При пустом OPENROUTER_API_KEY подхватывается .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_VR = Path(__file__).resolve().parent
load_dotenv(_VR / ".env", override=False)

OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str = "qwen/qwen3-235b-a22b-2507"
EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"
OPENROUTER_API_KEY: str = ""
if not (OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip()):
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_HTTP_REFERER: str = ""
OPENROUTER_APP_TITLE: str = "autogen-kg-bench"

# По умолчанию — тот же корпус, что у light-rag (относительно этой папки)
CORPUS_FILE: str = "../light-rag/corpus.txt"
BENCHMARK_FILE: str = ""
OUTPUT_FILE: str = "vector_benchmark_data.jsonl"
WORKING_DIR: str = ".vector_rag_data"

# Чанкинг и поиск
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150
RETRIEVAL_K: int = 5
LLM_TEMPERATURE: float = 0.2

REBUILD_INDEX: bool = False
LIMIT_QUESTIONS: int = 0

VECTOR_RAG_DIR: Path = _VR
