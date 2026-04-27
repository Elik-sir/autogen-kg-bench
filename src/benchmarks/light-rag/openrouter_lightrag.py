"""
LightRAG + OpenAI-совместимый API. Параметры — в `settings.py` (константы).
"""

from __future__ import annotations

import asyncio
import contextvars
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np

import settings

_LIGHRAG_IMPORT_ERROR: Exception | None = None
_NEO4J_ENTITY_STORAGE_PATCHED: bool = False
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import logger, wrap_embedding_func_with_attrs
except Exception as e:  # pragma: no cover
    LightRAG = None  # type: ignore[assignment, misc]
    QueryParam = None  # type: ignore[assignment, misc]
    _LIGHRAG_IMPORT_ERROR = e


def _api_key() -> str:
    return settings.OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")


def _base_url() -> str:
    return settings.OPENAI_API_BASE or "https://openrouter.ai/api/v1"


def _openai_client_extras() -> dict[str, Any]:
    ref = settings.OPENROUTER_HTTP_REFERER
    title = settings.OPENROUTER_APP_TITLE
    if not (ref and str(ref).strip()):
        return {}
    return {
        "default_headers": {
            "HTTP-Referer": ref,
            "X-Title": title,
        }
    }


def _llm_model_name() -> str:
    return settings.LLM_MODEL


def _embed_model_name() -> str:
    return settings.EMBEDDING_MODEL


def _embed_dim() -> int:
    return int(settings.EMBEDDING_DIM)


def apply_openrouter_env_defaults() -> None:
    if _api_key() and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = _api_key()
    if not (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")):
        os.environ["OPENAI_API_BASE"] = _base_url()
    if settings.USE_NEO4J:
        # neo4j_impl читает параметры подключения только из env/config.ini.
        if settings.NEO4J_URI:
            os.environ["NEO4J_URI"] = settings.NEO4J_URI
        if settings.NEO4J_USERNAME:
            os.environ["NEO4J_USERNAME"] = settings.NEO4J_USERNAME
        if settings.NEO4J_PASSWORD:
            os.environ["NEO4J_PASSWORD"] = settings.NEO4J_PASSWORD
        if settings.NEO4J_DATABASE:
            os.environ["NEO4J_DATABASE"] = settings.NEO4J_DATABASE
        if settings.NEO4J_WORKSPACE:
            os.environ["NEO4J_WORKSPACE"] = settings.NEO4J_WORKSPACE


# Включено только внутри `indexing_llm_retry_scope()` на время `ainsert`.
_indexing_llm_retries: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_indexing_llm_retries", default=False
)


@contextmanager
def indexing_llm_retry_scope() -> Generator[None, None, None]:
    """Включить повторные попытки LLM (таймаут → sleep → снова) только для индексации."""
    token = _indexing_llm_retries.set(True)
    try:
        yield
    finally:
        _indexing_llm_retries.reset(token)


async def _openrouter_llm_core(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs: Any,
) -> str:
    model = _llm_model_name()
    extras = _openai_client_extras()
    if extras and "openai_client_configs" not in kwargs:
        kwargs = {**kwargs, "openai_client_configs": extras}
    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        base_url=_base_url(),
        api_key=_api_key() or None,
        **kwargs,
    )


async def openrouter_style_llm(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    **kwargs: Any,
) -> str:
    if not _indexing_llm_retries.get():
        return await _openrouter_llm_core(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    import time

    worker_budget = max(30, 2 * int(settings.LLM_TIMEOUT_SEC) - 10)
    per_attempt = int(settings.LLM_INDEX_PER_ATTEMPT_TIMEOUT_SEC)
    if per_attempt <= 0:
        per_attempt = max(120, int(settings.LLM_TIMEOUT_SEC) // 2)

    max_attempts = int(settings.LLM_INDEX_MAX_ATTEMPTS)
    if max_attempts <= 0:
        max_attempts = 999_999

    t0 = time.monotonic()
    attempt = 0
    last_exc: BaseException | None = None

    while attempt < max_attempts:
        elapsed = time.monotonic() - t0
        remaining = worker_budget - elapsed
        if remaining < 5:
            break
        attempt += 1
        slice_timeout = min(float(per_attempt), float(remaining))
        try:
            return await asyncio.wait_for(
                _openrouter_llm_core(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs,
                ),
                timeout=slice_timeout,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            last_exc = e
            logger.warning(
                "LLM индексация: попытка %s оборвана по таймауту (≤%.0fs), "
                "осталось ~%.0fs бюджета воркера — повтор",
                attempt,
                slice_timeout,
                max(0.0, remaining - slice_timeout),
            )
            backoff = min(
                float(settings.LLM_INDEX_RETRY_BASE_DELAY_SEC) * (1.5 ** (attempt - 1)),
                60.0,
            )
            if time.monotonic() - t0 + backoff >= worker_budget:
                break
            await asyncio.sleep(backoff)

    if last_exc is not None:
        raise last_exc
    raise TimeoutError(
        f"Исчерпан бюджет LLM при индексации (~{worker_budget}s воркера LightRAG)"
    )


def _raw_openai_embed():
    return getattr(openai_embed, "func", openai_embed)


@wrap_embedding_func_with_attrs(
    embedding_dim=_embed_dim(),
    max_token_size=8192,
    model_name=_embed_model_name(),
)
async def openrouter_style_embed(texts: list[str], **kwargs: Any) -> np.ndarray:
    ex = _openai_client_extras() or None
    merged: dict[str, Any] = {
        "model": _embed_model_name(),
        "base_url": _base_url(),
        "api_key": _api_key() or None,
    }
    if ex and kwargs.get("client_configs") is None:
        merged["client_configs"] = ex
    inner_kw = {k: v for k, v in kwargs.items() if k in (
        "token_tracker",
        "embedding_dim",
        "max_token_size",
        "use_azure",
        "azure_deployment",
        "api_version",
        "client_configs",
    )}
    merged.update(inner_kw)
    if kwargs.get("client_configs") is not None:
        merged["client_configs"] = kwargs["client_configs"]
    return await _raw_openai_embed()(texts, **merged)


def _files_to_delete() -> list[str]:
    return [
        "graph_chunk_entity_relation.graphml",
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
    ]


def clear_working_dir(working_dir: Path) -> None:
    working_dir.mkdir(parents=True, exist_ok=True)
    for name in _files_to_delete():
        path = working_dir / name
        if path.exists():
            path.unlink()
            logger.info("Removed cache file %s", path)


def _register_neo4j_entity_type_label_storage() -> None:
    """Регистрирует кастомный graph storage в lightrag.kg и подключает его в LightRAG._get_storage_class."""
    global _NEO4J_ENTITY_STORAGE_PATCHED
    import lightrag.kg as kg

    name = "Neo4JEntityTypeLabelStorage"
    impls = kg.STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"]
    if name not in impls:
        impls.append(name)
    if name not in kg.STORAGE_ENV_REQUIREMENTS:
        kg.STORAGE_ENV_REQUIREMENTS[name] = list(
            kg.STORAGE_ENV_REQUIREMENTS["Neo4JStorage"]
        )

    if _NEO4J_ENTITY_STORAGE_PATCHED:
        return
    from lightrag.lightrag import LightRAG as LightRAGCls

    _orig = LightRAGCls._get_storage_class

    def _get_storage_class_patched(self, storage_name: str):
        if storage_name == name:
            from neo4j_entity_type_storage import Neo4JEntityTypeLabelStorage

            return Neo4JEntityTypeLabelStorage
        return _orig(self, storage_name)

    LightRAGCls._get_storage_class = _get_storage_class_patched  # type: ignore[method-assign]
    _NEO4J_ENTITY_STORAGE_PATCHED = True


def ensure_lightrag_available() -> None:
    if LightRAG is None or _LIGHRAG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Пакет lightrag-hku не установлен или импорт не удался. "
            "Выполните: uv sync (в папке light-rag). "
            f"Ошибка: {_LIGHRAG_IMPORT_ERROR}"
        ) from _LIGHRAG_IMPORT_ERROR


def build_rag(working_dir: str | Path) -> Any:
    ensure_lightrag_available()
    working_dir = Path(working_dir)
    apply_openrouter_env_defaults()
    if not _api_key():
        raise RuntimeError("Задайте OPENROUTER_API_KEY в settings.py или в .env")
    rag_kwargs: dict[str, Any] = {}
    if settings.USE_NEO4J:
        if settings.NEO4J_ENTITY_TYPE_AS_LABEL:
            _register_neo4j_entity_type_label_storage()
            rag_kwargs["graph_storage"] = "Neo4JEntityTypeLabelStorage"
        else:
            rag_kwargs["graph_storage"] = "Neo4JStorage"

    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=openrouter_style_llm,
        llm_model_name=_llm_model_name(),
        embedding_func=openrouter_style_embed,
        default_llm_timeout=int(settings.LLM_TIMEOUT_SEC),
        **rag_kwargs,
    )
