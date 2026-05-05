"""Судья бенчмарка: реализация в `utils.eval`, клиент/модель из локального `settings`."""

from __future__ import annotations

import settings  # noqa: E402
from utils.eval import (  # noqa: E402
    ACCURACY_SYSTEM,
    ACCURACY_USER_TEMPLATE,
    judge_correct,
    openai_client_for_judge,
    resolve_judge_model,
)

__all__ = [
    "ACCURACY_SYSTEM",
    "ACCURACY_USER_TEMPLATE",
    "judge_correct",
    "judge_model",
    "openai_client",
]


def openai_client():
    return openai_client_for_judge(settings)


def judge_model() -> str:
    return resolve_judge_model(settings)
