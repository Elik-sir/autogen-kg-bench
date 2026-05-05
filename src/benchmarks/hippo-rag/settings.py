"""
Константы бенчмарка HippoRAG.
При пустом OPENAI_API_KEY подхватывается из окружения или локального .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_HIPPO = Path(__file__).resolve().parent
load_dotenv(_HIPPO / ".env", override=False)

OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str =  os.environ.get("LLM_MODEL", "")
EMBEDDING_MODEL: str =  os.environ.get("EMBEDDING_MODEL", "")

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
# OpenRouter рекомендует Referer; пустой иногда даёт проблемы с HTTP-клиентом.
OPENROUTER_HTTP_REFERER: str = os.getenv(
    "OPENROUTER_HTTP_REFERER", "https://localhost"
)
OPENROUTER_APP_TITLE: str = os.getenv(
    "OPENROUTER_APP_TITLE", "autogen-kg-bench"
)

# По умолчанию — тот же корпус, что у light-rag
CORPUS_FILE: str = "../light-rag/corpus.txt"
BENCHMARK_FILE: str = ""
BENCHMARK_QUESTIONS_DIR: str = ""
OUTPUT_FILE: str = "benchmark_data.jsonl"
WORKING_DIR: str = ".hipporag_data"

LIMIT_QUESTIONS: int = 0
REBUILD_INDEX: bool = False
RETRIEVAL_K: int = 5

# Зарезервировано под вспомогательные LLM-метрики
METRICS_JUDGE_MODEL: str = ""
METRICS_API_DELAY_SEC: float = 0.0

HIPPO_RAG_DIR: Path = _HIPPO
