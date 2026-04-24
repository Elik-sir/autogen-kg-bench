BASE_SYSTEM_PROMPT = (
    "Ты — Data Scientist. Твоя задача — создать бенчмарк для тестирования систем GraphRAG. "
    "Тебе даны схема графовой базы Neo4j и ПРИМЕРЫ реальных данных из нее. "
    "Ты должен вернуть строго валидный JSON-массив объектов. Без markdown, без текста до/после. "
    "Генерируй только вопросы, на которые можно дать однозначный ответ по данным графа."
)

USEFUL_ENTITY_KEYS = {
    "name",
    "title",
    "description",
    "summary",
    "headline",
    "sector",
    "industry",
    "country",
    "region",
    "category",
    "status",
    "date",
    "year",
    "amount",
    "value",
    "revenue",
    "profit",
    "risk",
    "sentiment",
    "score",
    "impact",
    "ticker",
}


def _base_user_prompt(schema, data_samples):
    return f"""
Ты — Senior Neo4j Architect и эксперт по оценке систем GraphRAG.
Твоя задача — создать "Золотой стандарт" датасета для оценки качества извлечения знаний из графа.

=== ВХОДНЫЕ ДАННЫЕ ===
1. СХЕМА ГРАФА (Labels, Relationships, Properties):
{schema}

2. ПРИМЕРЫ ДАННЫХ:
{data_samples}

=== ОБЩИЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО) ===
1. Используй только Labels/Relationships/Properties, существующие в схеме.
2. Используй реальные значения из примеров данных в фильтрах WHERE/паттернах, чтобы запросы не были пустыми.
3. Формулируй вопрос естественно, как бизнес-аналитик.
4. Возвращай в RETURN конкретные поля, а не голые узлы.
5. Привязывайся к конкретным сущностям по идентифицирующим полям (name/title/ticker/id), избегай только категориальных фильтров.
"""


def _output_format_prompt(complexity):
    return f"""
=== ФОРМАТ ВЫВОДА ===
Верни только валидный JSON-массив:
[
  {{
    "complexity": "{complexity}",
    "question": "Текст вопроса на русском языке",
    "cypher": "MATCH ... RETURN ..."
  }}
]
"""


def build_simple_prompts(schema, data_samples, count):
    user_prompt = (
        _base_user_prompt(schema, data_samples)
        + f"""
=== ТИП ЗАДАЧИ: SIMPLE ===
Сгенерируй {count} вопросов типа "simple":
- доступ к атрибутам одного узла или его прямых соседей (1 hop),
- минимум один конкретный фильтр по сущности.
"""
        + _output_format_prompt("simple")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_multi_hop_prompts(schema, data_samples, count):
    user_prompt = (
        _base_user_prompt(schema, data_samples)
        + f"""
=== ТИП ЗАДАЧИ: MULTI-HOP ===
Сгенерируй {count} вопросов типа "multi-hop":
- путь 2-4 связи через разные типы узлов,
- фокус на скрытых связях/зависимостях.
"""
        + _output_format_prompt("multi-hop")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_aggregation_prompts(schema, data_samples, count):
    user_prompt = (
        _base_user_prompt(schema, data_samples)
        + f"""
=== ТИП ЗАДАЧИ: AGGREGATION ===
Сгенерируй {count} вопросов типа "aggregation":
- используй COUNT, MAX, MIN, AVG, ORDER BY или LIMIT,
- формулировка должна быть аналитической (топы, сравнения, динамика).
"""
        + _output_format_prompt("aggregation")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_cross_branch_prompts(schema, data_samples, count):
    user_prompt = (
        _base_user_prompt(schema, data_samples)
        + f"""
=== ТИП ЗАДАЧИ: CROSS-BRANCH (SUMMARIZATION/ANALYTICS) ===
Сгенерируй {count} вопросов типа "cross-branch" по алгоритму:
1) Выбери центральный Anchor-узел.
2) Построй ветку A от Anchor к Entity_A.
3) Построй независимую ветку B от Anchor к Entity_B.
4) В тексте вопроса избегай прямых имён Entity_A/Entity_B (Entity Masking), но Cypher должен извлекать их явно.

Критерий успешности: ответ требует объединить контекст обеих цепочек Anchor->Entity_A и Anchor->Entity_B.
"""
        + _output_format_prompt("cross-branch")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_subgraph_deep_analytics_prompts(schema, subgraph_contexts, count):
    system_prompt = (
        "Ты — аналитик. Твоя задача: по бизнес-контексту сформулировать сложные вопросы "
        "для проверки способности находить скрытые зависимости. "
        "Не упоминай графы, узлы, рёбра, связи, hop, cypher, schema."
    )

    def _is_useful_key(key: str) -> bool:
        lowered = key.lower()
        if "embedding" in lowered:
            return False
        if lowered in {"vector", "vectors"}:
            return False
        return lowered in USEFUL_ENTITY_KEYS

    def _pick_useful_props(props):
        if not isinstance(props, dict):
            return {}
        out = {}
        for k, v in props.items():
            if not _is_useful_key(str(k)):
                continue
            if isinstance(v, (str, int, float)) and v not in ("", None):
                out[k] = v
        return out

    contexts_text = ""
    for idx, ctx in enumerate(subgraph_contexts, 1):
        useful_lines = []
        if isinstance(ctx, dict):
            anchor = _pick_useful_props(ctx.get("anchor_props", {}))
            if anchor:
                useful_lines.append(f"anchor: {anchor}")
            # Полный контекст уже заранее очищен в company_subgraph_context,
            # но здесь дополнительно фиксируем, что нужны только полезные параметры сущностей.
            if ctx.get("useful_context"):
                useful_lines.append(str(ctx.get("useful_context")))
        else:
            useful_lines.append(str(ctx))
        contexts_text += f"\nКОНТЕКСТ {idx}:\n" + "\n".join(useful_lines) + "\n"
    print("contexts_text = ",contexts_text)
    user_prompt = f"""
Ниже даны фрагменты бизнес-контекста по компаниям.
Сгенерируй {count} пар "сложный аналитический вопрос + эталонный ответ".
Основная задача сделать вопрос полезным для аналитика, пусть он будет направлен на суммаризацию фактов из статей.
Также из предоставленного контекста надо будет исключить некоторые узлы и связи, для того, чтобы вопрос был направлен на поиск этого узла или связи
Критично:
1) Вопросы должны звучать как вопросы для стратегии/риска/рынка/операций.
2) Вопросы должны требовать синтеза нескольких сигналов, а не одного факта.
3) Не используй термины структуры данных (граф, узел, ребро, связь, путь, hop, cypher).
4) Вопрос может быть открытым, но должен быть проверяемым по данному контексту.
5) Ответ должен быть кратким, точным и опираться только на данный контекст (без выдумок).
6) Используй только полезные параметры сущностей (например: name, title, description и т.п.),
   игнорируй технические поля вроде embedding/vector.

КОНТЕКСТЫ:
{contexts_text}

Верни только JSON-массив:
[
  {{
    "complexity": "subgraph-deep-analytics",
    "question": "Текст сложного аналитического вопроса",
    "answer": "Краткий эталонный ответ на этот вопрос",
    "analysis_focus": ["сигнал 1", "сигнал 2", "сигнал 3"]
  }}
]
"""
    return system_prompt, user_prompt
