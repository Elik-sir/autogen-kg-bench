# Benchmark Generator

Папка `src/benchmark_generator` содержит модульный пайплайн генерации бенчмарка для Neo4j/GraphRAG.

## Структура

```text
src/benchmark_generator/
  __init__.py
  README.md
  generator.py
  pipeline.py
  question_generation.py
  validation.py
  answer_builder.py
  dedup.py
  query_filters.py
  prompt_settings.py
  utils/
    __init__.py
    prompt_builder.py
    schema_context.py
    company_subgraph_context.py
    same_type_common_context.py
    llm_response_parser.py
    benchmark_validation.py
```

## Роли файлов

- `generator.py` — фасад `BenchmarkGenerator`: инициализирует `Neo4jManager`, `LLMClient`, делегирует в pipeline и закрывает DB.
- `pipeline.py` — orchestration: общий цикл по типам вопросов (`simple`, `multi-hop-2`, `multi-hop-3`, `multi-hop-4`, `aggregation`, `cross-branch`, `subgraph-deep-analytics`), прогресс, попытки, финальная запись JSON.
- `question_generation.py` — генерация кандидатов через LLM:
  - формирование промптов,
  - парсинг JSON-ответа модели,
  - спец-логика для `same-type-common` и `subgraph-deep-analytics`.
- `validation.py` — валидация кандидатов:
  - дедупликация вопросов,
  - проверка Cypher и выполнение запроса в Neo4j,
  - фильтрация low-signal результатов,
  - сбор `ground_truth` и генерация референсного `answer`.
- `answer_builder.py` — построение эталонного ответа из `ground_truth` через LLM.
- `dedup.py` — нормализация текста вопроса и near-duplicate проверка.
- `query_filters.py` — утилиты для Cypher (`LIMIT`, low-signal aggregate фильтр).
- `prompt_settings.py` — общие константы и лимиты для промптов и компактизации входа.

### Папка `utils/`

- `prompt_builder.py` — сборка system/user промптов для каждого типа вопроса.
- `anchor_subgraph_context.py` — stratified anchor extraction и построение локальной онтологии/путей для `multi-hop-X`.
- `schema_context.py` — извлечение схемы и sample-данных из Neo4j.
- `company_subgraph_context.py` — подготовка контекстов подграфа компании для deep analytics.
- `same_type_common_context.py` — поиск пар узлов одного типа с общей сущностью.
- `llm_response_parser.py` — парсер LLM-ответа в JSON.
- `benchmark_validation.py` — базовые проверки/преобразования для ground truth.

## Общая логика пайплайна

1. Точка входа в проекте (`src/main.py`) создает `BenchmarkGenerator` и вызывает `run(...)`.
2. `BenchmarkGenerator.run()` передает управление в `run_generation_pipeline(...)`.
3. Pipeline:
   - читает схему и сэмплы из Neo4j;
   - формирует план генерации по типам;
   - в цикле вызывает генератор вопросов и валидатор до достижения целевых количеств.
4. Генерация:
   - для каждого типа строится промпт на основе схемы/сэмплов;
   - LLM возвращает JSON с кандидатами (`question`, `cypher`, и т.д.).
5. Валидация:
   - удаляются дубликаты и тривиальные запросы;
   - Cypher выполняется в Neo4j;
   - вычисляется/собирается `ground_truth`;
   - строится краткий эталонный `answer`.
6. Результат инкрементально и финально пишется в выходной JSON-файл бенчмарка.

## Зависимости между слоями

- Верхний уровень: `generator.py` -> `pipeline.py`
- Domain-слой:
  - `pipeline.py` -> `question_generation.py`, `validation.py`
  - `validation.py` -> `dedup.py`, `query_filters.py`, `answer_builder.py`
- Prompt/context слой: `question_generation.py` -> `utils/*`, `prompt_settings.py`

Такое разделение сделано, чтобы:
- изолировать бизнес-шаги (генерация, валидация, orchestration),
- упростить точечные правки и тесты,
- избежать “god-file” с неявными связями.
