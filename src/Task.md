Вот описание задачи в формате Markdown, оптимизированное для индексации в Cursor. Этот файл поможет нейросети понять архитектуру проекта, логику генерации данных и требования к формату бенчмарка.

---

# Задача: Автоматический генератор бенчмарков для GraphRAG систем

## 1. Обзор проекта

**Цель:** Создать систему для автоматической генерации набора тестовых данных (бенчмарка) на основе существующего графа знаний в Neo4j. Бенчмарк предназначен для оценки эффективности различных RAG-фреймворков (GraphRAG, LightRAG и т.д.) на конкретной бизнес-онтологии.

## 2. Проблема

Публичные бенчмарки не отражают специфику частных корпоративных данных. Нужен эталон: вопрос + проверяемый артефакт (ответ из БД или явный эталон от LLM по выгруженному контексту), чтобы сравнивать системы извлечения и ответа.

## 3. Этапы реализации (Pipeline)

### А. Извлечение контекста БД

1. **Схема:** `CALL apoc.meta.schema()` — структура (labels, relationships, properties). Реализация: `utils/schema_context.py` → `get_schema()`.

2. **Примеры данных (Data Grounding):** `get_samples()` в `utils/schema_context.py`:
   - для каждого node-label из схемы подбираются узлы так, чтобы **покрыть типы связей**, допустимые схемой для этой метки (жадный set cover), см. `utils/rel_type_cover.py` → `select_nodes_covering_schema_rel_types()`;
   - лимит примеров на метку задаётся константой **`DATA_SAMPLES_PER_LABEL_LIMIT`** (см. `schema_context.py`); при нехватке узлов от покрытия контекст **добирается** дополнительными узлами той же метки;
   - при сборе логируется краткий отчёт: fallback-метки, типы связей из схемы, которых нет в данных (`missing_in_graph`).

### Б. Генерация вопросов (несколько типов, отдельные промпты)

Оркестратор: `src/main.py` → класс **`BenchmarkGenerator`**. Для каждого типа вопросов — **отдельная функция генерации** и **отдельный промпт** в `utils/prompt_builder.py`:

| Тип (`complexity`) | Промпт | Суть |
|--------------------|--------|------|
| `simple` | `build_simple_prompts` | 1 hop, конкретная привязка к сущности |
| `multi-hop` | `build_multi_hop_prompts` | цепочки 2–4 связей |
| `aggregation` | `build_aggregation_prompts` | COUNT / агрегаты / топы |
| `cross-branch` | `build_cross_branch_prompts` | параллельные ветки от anchor, маскирование сущностей в формулировке вопроса |
| `subgraph-deep-analytics` | `build_subgraph_deep_analytics_prompts` | см. отдельный подраздел ниже |

Общий парсинг ответа LLM: `utils/llm_response_parser.py` → `parse_qa_pairs_response()` (строгий JSON-массив).

**Дополнительный модуль (алгоритмический):** `utils/cross_branch_reasoning.py` — программная генерация cross-branch кейсов по схеме и Neo4j; при необходимости его можно снова подключить в пайплайн отдельно от LLM-ветки `cross-branch`.

#### Подтип: `subgraph-deep-analytics`

1. **Выбор компании-якоря:** `utils/company_subgraph_context.py` → `build_company_subgraph_contexts()`:
   - label компании: предпочтительно метка с подстрокой `company` в имени, иначе первый node-label из схемы;
   - среди компаний выбираются якоря с **наибольшей степенью** и **разнообразием типов связей** (`degree`, `rel_type_variety`), `LIMIT anchors_limit`.

2. **Выгрузка подграфа:** для `elementId(c)` строится snapshot: соседи 1-hop и цепочки 2-hop (с лимитами в Cypher), затем:
   - **`_sanitize_snapshot`**: рекурсивно удаляются свойства вроде `embedding` / `vector` (и ключи, содержащие `embedding`), чтобы они **не попадали в контекст и не сохранялись**;
   - **`subgraph_context`**: человекочитаемый текст снимка (в т.ч. для отладки);
   - **`useful_context`**: сжатый текст «полезных сигналов» (name, title, headline, description, summary, text, даты, метрики и т.д.; без эмбеддингов; дедупликация повторяющихся сигналов).

3. **LLM для этого типа:** в промпт передаётся **только бизнес-контекст** (dict с `anchor_props`, `useful_context` и т.д.), без упора на граф/Neo4j/Cypher в инструкции. Модель генерирует пару **`question` + `answer`** и опционально **`analysis_focus`**.

4. **Что пишется в бенчмарк после постобработки в `main.py`:**
   - **`answer`** — эталонный ответ, сгенерированный LLM;
   - **`ground_truth`** — **контекст**, на основе которого генерировались вопрос и ответ (строка `useful_context`);
   - **`cypher` + `params`** — только **debug-выгрузка** того же подграфа (`debug_cypher` / `debug_params`); флаг **`debug_only_cypher: true`**;
   - **`subgraph_context`**, **`useful_context`** — для трассировки и отладки.

### В. Валидация и запись в файл

1. **Cypher:** для обычных типов (`simple`, `multi-hop`, `aggregation`, `cross-branch`) запрос выполняется в Neo4j с опциональными **`params`**. Пустой результат или синтаксическая ошибка → кейс отбрасывается.

2. **`subgraph-deep-analytics`:** `cypher` выполняется как диагностика; **`ground_truth` не строится из результата запроса** — он уже задан контекстом. Пустой результат debug-запроса не должен отменять кейс, если `ground_truth` уже есть.

3. **Тривиальные запросы:** отсекаются `utils/benchmark_validation.py` → `is_trivial_self_return()` (если задан непустой `cypher`).

4. **Текстовый эталон для типов с «настоящим» Cypher:** `result_to_ground_truth()` в `benchmark_validation.py` (ранее алиас `result_to_ideal_context`) — сериализация строк результата Neo4j в одну строку для сравнения с ответом RAG.

5. **Сохранение:** `graphrag_benchmark.json` (или путь из `run(...)`).

## 4. Спецификация выходного файла (`graphrag_benchmark.json`)

Массив объектов. Ключ **`ground_truth`** заменяет устаревшее **`ideal_context`** в основном генераторе. Вспомогательные скрипты бенчмарков (`benchmarks/light-rag`, `benchmarks/vector-rag`, `cde_metrics.py`) при чтении поддерживают **fallback** на `ideal_context` для старых файлов.

### Общий случай (Cypher из LLM)

```json
{
  "complexity": "multi-hop",
  "question": "…",
  "cypher": "MATCH … RETURN …",
  "params": null,
  "ground_truth": "Текстовое представление результата выполнения Cypher в Neo4j"
}
```

### `subgraph-deep-analytics`

```json
{
  "complexity": "subgraph-deep-analytics",
  "question": "…",
  "answer": "Эталонный ответ LLM по переданному контексту",
  "ground_truth": "Текст useful_context — контекст, на котором строились вопрос и ответ",
  "analysis_focus": ["сигнал 1", "сигнал 2"],
  "cypher": "DEBUG: выгрузка подграфа",
  "params": { "anchor_id": "…" },
  "debug_only_cypher": true,
  "subgraph_context": "…",
  "useful_context": "…"
}
```

## 5. Используемый стек и компоненты

| Компонент | Назначение |
|-----------|------------|
| `neo4j_manager.py` → `Neo4jManager` | Сессии Neo4j, `run_query(query, parameters=None)` |
| `llm_client.py` → `LLMClient` | Вызов LLM (OpenRouter и т.д.) |
| `main.py` → `BenchmarkGenerator` | Цикл: схема + samples → генерация по типам → валидация → JSON |
| `utils/schema_context.py` | `get_schema`, `get_samples` (покрытие типов связей, лимиты, логи) |
| `utils/rel_type_cover.py` | Set cover по типам рёбер для примеров по label |
| `utils/prompt_builder.py` | Отдельные промпты по типам вопросов |
| `utils/company_subgraph_context.py` | Подграф компании, санитизация, `useful_context` |
| `utils/benchmark_validation.py` | Тривиальные запросы, `result_to_ground_truth` |
| `utils/cross_branch_reasoning.py` | Опциональная программная генерация cross-branch |

## 6. Прогон бенчмарка (LightRAG / vector-rag)

- В результатах прогона эталон читается как **`ground_truth`**, при отсутствии — **`ideal_context`** (старые файлы).
- Метрика пересечения по токенам: **`recall_on_ground_truth_tokens`** / **`mean_recall_on_ground_truth_tokens`** (со fallback на старые имена полей при необходимости).

## 7. Ограничения и заметки

- Нужен **APOC** для `apoc.meta.schema` (и ранее для части примеров через APOC в схеме выборки).
- Размер промпта ограничен контекстным окном LLM; подграф и `useful_context` усечены лимитами в Cypher и в постобработке.
- Поля **`embedding` / векторные представления** намеренно вычищаются из данных, идущих в контекст и в сохранённый JSON, чтобы не раздувать бенчмарк и не утекали в LLM.

---

### Инструкция для Cursor

> При добавлении новых типов вопросов: отдельная пара «функция генерации в `BenchmarkGenerator` + `build_*_prompts` в `prompt_builder.py`», единый формат JSON для парсера. Для полей эталона используй **`ground_truth`**; для типов с отдельной семантикой (как `subgraph-deep-analytics`) явно документируй, что кладётся в `ground_truth` vs `answer` vs debug `cypher`.
