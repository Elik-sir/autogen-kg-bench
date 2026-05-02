# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Сначала только тяжёлые зависимости — слой кэшируется, пока не тронете версии ниже.
# Раньше был `COPY .` перед `pip install`: любой файл в репо сбрасывал кэш и качал всё заново.
COPY src/benchmarks/hippo-rag/pyproject.toml \
    src/benchmarks/hippo-rag/patch_hipporag_openai_embedding.py \
    /tmp/hippo-rag-bench/
WORKDIR /tmp/hippo-rag-bench
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    "python-dotenv>=1.0.0" \
    "openai>=1.0.0" \
    "hipporag" \
    && python patch_hipporag_openai_embedding.py
WORKDIR /app
COPY . /app

WORKDIR /app/src/benchmarks/hippo-rag
# Локальный раннер не обязан ставиться как пакет: скрипты лежат в WORKDIR.
CMD ["python", "main.py"]
