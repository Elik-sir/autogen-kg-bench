"""Настройки клиента и моделей для прогона RAGAS."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RagasJudgeSettings(BaseModel):
    """
    Параметры подключения к OpenAI-совместимому API (OpenAI, OpenRouter, vLLM и т.д.).

    ``llm_factory`` / ``embedding_factory`` из RAGAS принимают стандартный ``OpenAI``-клиент.
    """

    api_key: str = Field(..., min_length=1, description="API-ключ провайдера.")
    base_url: str | None = Field(
        default=None,
        description="Необязательный base_url (например https://openrouter.ai/api/v1).",
    )
    llm_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Идентификатор модели для LLM-метрик RAGAS.",
    )
    llm_max_tokens: int = Field(
        default=16384,
        ge=256,
        le=200_000,
        description=(
            "Лимит токенов ответа для ``llm_factory`` (RAGAS по умолчанию 1024). "
            "Faithfulness отдаёт длинный JSON — при обрезке будет ошибка про max_tokens."
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Модель эмбеддингов для метрик, где нужны векторы (например answer_relevancy).",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Доп. заголовки HTTP (OpenRouter: HTTP-Referer, X-Title).",
    )
    openai_chat_extra_body: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Пробрасывается в ``chat.completions.create(..., extra_body=...)`` (OpenRouter и др.). "
            "Пример только Google Vertex: ``{\"provider\": {\"only\": [\"google-vertex\"]}}``."
        ),
    )
    answer_relevancy_strictness: int = Field(
        default=1,
        ge=1,
        le=10,
        description=(
            "Сколько вариантов генерирует answer_relevancy (RAGAS). "
            "Многие OpenAI-совместимые API возвращают только одну генерацию при n>1 — "
            "тогда RAGAS пишет предупреждение; strictness=1 этого избегает."
        ),
    )
    ragas_max_workers: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Параллельность RAGAS ``evaluate`` (``RunConfig.max_workers``, asyncio-семафор).",
    )
    ragas_run_max_retries: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "``RunConfig.max_retries`` для RAGAS (повторы LLM при временных сбоях). "
            "Раньше в бенче стояло 1 — мало для нестабильного JSON."
        ),
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def _empty_base_url(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def openai_client_kwargs(self) -> dict[str, Any]:
        """Аргументы для ``openai.OpenAI(...)``."""
        kw: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kw["base_url"] = self.base_url.rstrip("/")
        if self.default_headers:
            kw["default_headers"] = dict(self.default_headers)
        return kw
