"""
Все настройки — константы в этом файле. При необходимости подгружается `light-rag/.env`
только для подстановки OPENROUTER_API_KEY, если ниже оставлено пусто.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_LIGHT_RAG = Path(__file__).resolve().parent
load_dotenv(_LIGHT_RAG / ".env", override=False)

# --- API (OpenRouter) ---
OPENAI_API_BASE: str = "https://openrouter.ai/api/v1"
LLM_MODEL: str = "qwen/qwen3-235b-a22b-2507"
EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"
EMBEDDING_DIM: int = 4096
OPENROUTER_API_KEY: str = ""
if not (OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip()):
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_HTTP_REFERER: str = ""
OPENROUTER_APP_TITLE: str = "autogen-kg-bench"

# --- Пути: относительно папки light-rag, если не абсолютные ---
CORPUS_FILE: str = "corpus.txt"
# Пусто = graphrag_benchmark.json в корне репозитория
BENCHMARK_FILE: str = ""
OUTPUT_FILE: str = "benchmark_data.jsonl"
WORKING_DIR: str = ".lightrag_data"

# --- Запуск бенчмарка ---
QUERY_MODE: str = "hybrid"
REBUILD_CACHE: bool = False
LIMIT_QUESTIONS: int = 0

# Корень пакета (папка light-rag)
LIGHT_RAG_DIR: Path = _LIGHT_RAG

# --- cde_metrics.py: оценщик CDE (пусто = тот же LLM_MODEL) ---
METRICS_JUDGE_MODEL: str = ""
METRICS_API_DELAY_SEC: float = 0.0
