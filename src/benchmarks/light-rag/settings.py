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

# Таймаут одного LLM-вызова при извлечении сущностей (сек). В LightRAG воркер обрывает
# задачу примерно через 2× это значение (см. priority_limit_async_func_call).
# По умолчанию в библиотеке 180 с → 360 с на воркер; для больших чанков/медленного API задайте выше.
LLM_TIMEOUT_SEC: int = int(os.getenv("LIGHTRAG_LLM_TIMEOUT_SEC", "600"))

# Повторы LLM только на этапе индексации (ainsert): один «внешний» вызов llm_model_func
# обёрнут LightRAG в wait_for(~2 * LLM_TIMEOUT_SEC). Внутри него делаем короткие попытки
# openai_complete_if_cache и при таймауте перезапускаем, пока не исчерпан бюджет воркера.
# 0 = вычислить как max(120, LLM_TIMEOUT_SEC // 2).
LLM_INDEX_PER_ATTEMPT_TIMEOUT_SEC: int = int(
    os.getenv("LIGHTRAG_INDEX_PER_ATTEMPT_SEC", "0")
)
LLM_INDEX_RETRY_BASE_DELAY_SEC: float = float(
    os.getenv("LIGHTRAG_INDEX_RETRY_DELAY_SEC", "2")
)
LLM_INDEX_MAX_ATTEMPTS: int = int(os.getenv("LIGHTRAG_INDEX_MAX_ATTEMPTS", "80"))

# --- Пути: относительно папки light-rag, если не абсолютные ---
CORPUS_FILE: str = "corpus.txt"
# Пусто = graphrag_benchmark.json в корне репозитория
BENCHMARK_FILE: str = ""
OUTPUT_FILE: str = "benchmark_data.jsonl"
WORKING_DIR: str = ".lightrag_data"

# --- Graph storage (Neo4j) ---
# Включает graph_storage="Neo4JStorage" в LightRAG.
USE_NEO4J: bool = os.getenv("LIGHTRAG_USE_NEO4J", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Можно оставить пустым, тогда будут использованы значения из окружения.
NEO4J_URI: str = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", ""))
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
# Имя БД в Neo4j (по умолчанию neo4j).
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "lightrag")
# Рабочее пространство LightRAG внутри Neo4j (метка для изоляции графов).
NEO4J_WORKSPACE: str = os.getenv("NEO4J_WORKSPACE", "lightrag")

# Вторая метка узла в Neo4j из entity_type (Person, Organization, …) — удобно для бенчмарка.
NEO4J_ENTITY_TYPE_AS_LABEL: bool = os.getenv(
    "LIGHTRAG_NEO4J_ENTITY_TYPE_AS_LABEL", "true"
).strip().lower() in ("1", "true", "yes")

# Импорт GraphML → Neo4j (import_graphml_to_neo4j.py)
GRAPHML_FILE: str = os.getenv("LIGHTRAG_GRAPHML_FILE", "graph_chunk_entity_relation.graphml")
# lightrag — тип связи DIRECTED, поле keywords в свойствах (как Neo4JStorage).
# keywords_primary — тип связи Neo4j из первого ключевого слова в keywords.
NEO4J_IMPORT_REL_TYPE_MODE: str = os.getenv(
    "LIGHTRAG_IMPORT_REL_TYPE_MODE", "keywords_primary"
).strip().lower() or "lightrag"

# --- Запуск бенчмарка ---
QUERY_MODE: str = "naive"
REBUILD_CACHE: bool = False
LIMIT_QUESTIONS: int = 0

# Только дорисовать индексацию: вызвать apipeline_process_enqueue_documents без повторного
# enqueue того же текста (после сбоя документ остаётся FAILED/PENDING в doc_status).
# Env: LIGHTRAG_RESUME_PIPELINE_ONLY=1
RESUME_PIPELINE_ONLY: bool = (
    os.getenv("LIGHTRAG_RESUME_PIPELINE_ONLY", "").strip().lower()
    in ("1", "true", "yes")
)

# Только ответы на вопросы бенчмарка: не вызывать ainsert / pipeline, не проверять doc_status.
# Env: LIGHTRAG_QUERY_ONLY=1 (когда граф и вектора уже в WORKING_DIR)
QUERY_ONLY: bool = os.getenv("LIGHTRAG_QUERY_ONLY", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Корень пакета (папка light-rag)
LIGHT_RAG_DIR: Path = _LIGHT_RAG

# --- LLM-as-judge (main.py accuracy, cde_metrics.py): пусто = тот же LLM_MODEL ---
METRICS_JUDGE_MODEL: str = ""
METRICS_API_DELAY_SEC: float = 0.0
# Вызов судьи после прогона; отключить: LIGHTRAG_ENABLE_LLM_JUDGE=0
ENABLE_LLM_ACCURACY: bool = os.getenv(
    "LIGHTRAG_ENABLE_LLM_JUDGE", "1"
).strip().lower() not in ("0", "false", "no")
