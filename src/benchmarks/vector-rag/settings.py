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
LLM_MODEL: str =  os.environ.get("LLM_MODEL", "")
EMBEDDING_MODEL: str =  os.environ.get("EMBEDDING_MODEL", "")

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_HTTP_REFERER: str = ""
OPENROUTER_APP_TITLE: str = "autogen-kg-bench"
CORPUS_FILE: str = "../light-rag/corpus.txt"
BENCHMARK_FILE: str = ""
# Пусто → каталог `benchmark_questions_by_type` в корне репозитория (после split-скрипта)
BENCHMARK_QUESTIONS_DIR: str = ""
OUTPUT_FILE: str = "vector_benchmark_data.jsonl"
WORKING_DIR: str = ".vector_rag_data"

# Чанкинг и поиск
CHUNK_SIZE: int = 600
CHUNK_OVERLAP: int = 100
RETRIEVAL_K: int = 5
LLM_TEMPERATURE: float = 0.2

REBUILD_INDEX: bool = False
LIMIT_QUESTIONS: int = 0

# Сколько вопросов бенчмарка обрабатывать одновременно (потоки: FAISS + HTTP к LLM/эмбеддингам).
QUESTION_CONCURRENCY: int = 32

# Зарезервировано под вспомогательные LLM-метрики (пустая строка = тот же LLM_MODEL)
METRICS_JUDGE_MODEL: str = ""
METRICS_API_DELAY_SEC: float = 0.0

VECTOR_RAG_DIR: Path = _VR
