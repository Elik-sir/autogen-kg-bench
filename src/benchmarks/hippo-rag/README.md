HippoRAG benchmark runner.

Запуск:
- `cd src/benchmarks/hippo-rag`
- `uv sync`
- `(PowerShell) $env:PYTHONPATH = (Resolve-Path ..\\..).Path; uv run python main.py`

По умолчанию:
- читает `graphrag_benchmark.json` из корня репозитория;
- использует корпус `../light-rag/corpus.txt`;
- пишет результат в `benchmark_data.jsonl`.

Важно для Windows:
- HippoRAG тянет `vllm`, а некоторые версии `vllm` на Windows падают на импорте `resource`.
- Надёжный вариант запуска — WSL/Linux.

Docker (Linux внутри контейнера):
- Сборка из корня репозитория:
  - `docker build -f src/benchmarks/hippo-rag/Dockerfile -t hippo-rag-bench .`
- Запуск:
  - `docker run --rm -e OPENROUTER_API_KEY=... hippo-rag-bench`

Docker Compose:
- Перейти в папку:
  - `cd src/benchmarks/hippo-rag`
- Ключ API: скопируйте `.env.example` → `.env` и задайте `OPENAI_API_KEY=...` (или отредактируйте уже созданный `.env`).
- Запустить:
  - `docker compose up --build`

В compose добавлен volume `../../..:/app`, поэтому результаты и кэш пишутся сразу в ваш локальный репозиторий.

Примечания:
- Контейнер использует `src/benchmarks/hippo-rag/settings.py`.
- Эмбеддинги: HippoRAG принимает OpenAI-совместимые модели только если в **имени** есть подстрока `text-embedding` (см. `hipporag/embedding_model/__init__.py`). По умолчанию стоит `openai/text-embedding-3-small` на OpenRouter. Свой вариант: `HIPPORAG_EMBEDDING_MODEL=...` (в т.ч. в `docker-compose.yml`).
- По умолчанию берутся:
  - `graphrag_benchmark.json` (из корня репо),
  - `src/benchmarks/light-rag/corpus.txt`,
  - output `src/benchmarks/hippo-rag/benchmark_data.jsonl`.
- Для ускорения сборки добавлен корневой `.dockerignore` (чтобы не отправлять `.venv` и кэши в build context).
- В Docker-образе используется `pip` (без `uv`) для более стабильной сборки.
- Зависимости в образе ставятся отдельным слоем до `COPY` всего репо; при добавлении пакетов в `pyproject.toml` обновите список `pip install` в `Dockerfile`.
- После `pip install hipporag` в образе запускается `patch_hipporag_openai_embedding.py`: убирает отладочный `ipdb` и подставляет `OpenAI(api_key=..., default_headers=...)` для OpenRouter. Локально: `python patch_hipporag_openai_embedding.py` (из этой папки, после установки зависимостей).
- Если видите `openai.APIConnectionError: Connection error` из контейнера: проверьте, что ключ задан (`OPENROUTER_API_KEY` / в compose продублирован `OPENAI_API_KEY`), из контейнера есть выход в интернет (`docker compose run --rm hippo-rag-bench python -c "import urllib.request; urllib.request.urlopen('https://openrouter.ai', timeout=10)"`), при необходимости DNS в Docker Desktop.
