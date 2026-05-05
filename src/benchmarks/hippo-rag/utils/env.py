from __future__ import annotations

import os

import settings  # noqa: WPS433 — пакет настроек каталога hippo-rag


def prepare_hipporag_env() -> None:
    """Ключ и заголовки для OpenAI-совместимых клиентов (OpenRouter) внутри hipporag."""
    key = (settings.OPENAI_API_KEY or "").strip()
    if not key:
        key = (os.environ.get("OPENROUTER_API_KEY", "") or "").strip()
    os.environ["OPENAI_API_KEY"] = key
    if not (os.environ.get("OPENROUTER_API_KEY", "") or "").strip():
        os.environ["OPENROUTER_API_KEY"] = key
    os.environ.setdefault("OPENAI_BASE_URL", settings.OPENAI_API_BASE)
    if settings.OPENROUTER_HTTP_REFERER.strip():
        os.environ["OPENROUTER_HTTP_REFERER"] = settings.OPENROUTER_HTTP_REFERER.strip()
    if settings.OPENROUTER_APP_TITLE.strip():
        os.environ["OPENROUTER_APP_TITLE"] = settings.OPENROUTER_APP_TITLE.strip()
