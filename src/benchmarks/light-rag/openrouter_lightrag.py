"""
LightRAG + OpenAI-совместимый API. Параметры — в `settings.py` (константы).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

import settings

_LIGHRAG_IMPORT_ERROR: Exception | None = None
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


async def openrouter_style_llm(
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
    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=openrouter_style_llm,
        llm_model_name=_llm_model_name(),
        embedding_func=openrouter_style_embed,
    )
