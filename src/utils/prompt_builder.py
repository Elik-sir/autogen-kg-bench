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


def _existing_questions_prompt(existing_questions):
    questions = [str(q).strip() for q in (existing_questions or []) if str(q).strip()]
    if not questions:
        return ""
    formatted = "\n".join(f"- {q}" for q in questions[-200:])
    return f"""

3. УЖЕ СГЕНЕРИРОВАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯТЬ):
{formatted}

=== АНТИДУБЛИКАТНЫЕ ПРАВИЛА (ОБЯЗАТЕЛЬНО) ===
- Запрещено дословно повторять любой вопрос из списка.
- Запрещено делать близкий парафраз уже существующего вопроса.
- Если кандидат слишком похож по смыслу, выбери другую сущность, другую метрику или другой ракурс.
"""


def _base_user_prompt(schema, data_samples, existing_questions=None):
    return f"""
Ты — Senior Neo4j Architect и эксперт по оценке систем GraphRAG.
Твоя задача — создать "Золотой стандарт" датасета для оценки качества извлечения знаний из графа.

=== ВХОДНЫЕ ДАННЫЕ ===
1. СХЕМА ГРАФА (Labels, Relationships, Properties):
{schema}

2. ПРИМЕРЫ ДАННЫХ:
{data_samples}
{_existing_questions_prompt(existing_questions)}

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


def build_simple_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== ТИП ЗАДАЧИ: SIMPLE ===
Сгенерируй {count} вопросов типа "simple":
- доступ к атрибутам одного узла или его прямых соседей (1 hop),
- минимум один конкретный фильтр по сущности.
"""
        + _output_format_prompt("simple")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_multi_hop_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== ТИП ЗАДАЧИ: MULTI-HOP ===
Сгенерируй {count} вопросов типа "multi-hop":
- путь 2-4 связи через разные типы узлов,
- фокус на скрытых связях/зависимостях.

=== ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА ДЛЯ НЕПУСТОГО РЕЗУЛЬТАТА (ОБЯЗАТЕЛЬНО) ===
1. Каждый вопрос должен быть привязан минимум к одному "якорю" (name/title/ticker), который ЯВНО присутствует в ПРИМЕРАХ ДАННЫХ.
2. Не используй в WHERE редкие/экзотические значения, если они не встречаются в ПРИМЕРАХ ДАННЫХ.
3. Избегай чрезмерно узких комбинаций фильтров (город + отрасль + ресурс + ключевое слово) в одном запросе.
4. Перед финализацией Cypher проведи внутреннюю self-check:
   - есть ли в запросе хотя бы один конкретный якорь из примеров;
   - не приведет ли набор фильтров к пустому пересечению;
   - нельзя ли сделать запрос менее хрупким без потери multi-hop-смысла.
5. Если запрос агрегатный (COUNT/SUM/AVG/MIN/MAX), он должен быть сформулирован так, чтобы результат был информативным (не null и не тривиальный ноль).
6. Предпочитай паттерны, где хотя бы один hop "подтвержден" примерами (т.е. сущности и связи встречаются в предоставленных данных).
"""
        + _output_format_prompt("multi-hop")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_aggregation_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== ТИП ЗАДАЧИ: AGGREGATION ===
Сгенерируй {count} вопросов типа "aggregation":
- используй COUNT, MAX, MIN, AVG, ORDER BY или LIMIT,
- формулировка должна быть аналитической (топы, сравнения, динамика).
"""
        + _output_format_prompt("aggregation")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_cross_branch_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
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


def build_same_type_common_prompts(schema, data_samples, pair_context: dict, existing_questions=None):
    """
    Один кейс за вызов: pair_context из same_type_common_context.find_same_type_common_contexts.
    """
    ctx = pair_context if isinstance(pair_context, dict) else {}
    lbl = ctx.get("node_label", "?")
    pa = ctx.get("props_a") or {}
    pb = ctx.get("props_b") or {}
    cl = ctx.get("common_labels") or []
    cc = ctx.get("common_props") or {}
    hop_a = ctx.get("hop1_text_a", "")
    hop_b = ctx.get("hop1_text_b", "")
    da = ctx.get("dist_a")
    db = ctx.get("dist_b")
    pha = ctx.get("path_hint_a")
    phb = ctx.get("path_hint_b")

    path_lines = []
    if da is not None and db is not None:
        path_lines.append(
            f"Длины найденных путей A→общая и B→общая (в рёбрах): {da} и {db} (каждый не более 3)."
        )
    if pha:
        path_lines.append(f"Пример кратчайшего пути A→общая: {pha}")
    if phb:
        path_lines.append(f"Пример кратчайшего пути B→общая: {phb}")
    path_block = "\n".join(path_lines) if path_lines else ""

    case_block = f"""
КЕЙС (метка узлов: {lbl})
Узел A (с B нет прямой связи): {pa}
Узел B (с A нет прямой связи): {pb}
Окрестность 1-hop у A:
{hop_a}
Окрестность 1-hop у B:
{hop_b}
{path_block}

Служебно для синтеза Cypher (не повторяй в вопросе дословно): общая сущность — метки {cl}, ключевые поля {cc}.
Формулировка вопроса: «что общего / что объединяет / общий элемент контекста» для A и B, без явного названия значений из {cc}.
"""

    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + """
=== ТИП ЗАДАЧИ: SAME-TYPE-COMMON ===
Сгенерируй ровно 1 вопрос по кейсу ниже.

Логика:
- A и B — узлы одной метки, между ними нет ребра.
- Есть сущность, достижимая от A и от B по цепочкам длиной 1–3 рёбер (в т.ч. только через 2–3 шага, без общего прямого соседа).
- Локальные 1-hop списки и подсказки по путям даны для контекста; вопрос должен нацеливаться на эту общую сущность.

Требования:
1) В вопросе ссылайся на A и B по полям из кейса (name/title/ticker и т.д.).
2) Не называй в вопросе общую сущность — ответ должен получаться запросом.
3) Cypher однозначно возвращает эту общую сущность (RETURN понятных полей узла). Допускаются пути фиксированной длины или *1..3, если это следует из схемы. Только метки/типы рёбер из схемы.
"""
        + case_block
        + _output_format_prompt("same-type-common")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_subgraph_deep_analytics_prompts(schema, subgraph_contexts, count, existing_questions=None):
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

    single_block = len(subgraph_contexts) == 1 and count == 1
    if single_block:
        intro = """Ниже дан один фрагмент бизнес-контекста по компании.
Сгенерируй ровно одну пару «сложный аналитический вопрос + эталонный ответ».
Вопрос и ответ должны полностью опираться только на этот фрагмент: любой факт в ответе должен выводиться из приведённых сигналов (без внешних знаний)."""
    else:
        intro = f"""Ниже даны фрагменты бизнес-контекста по компаниям.
Сгенерируй {count} пар «сложный аналитический вопрос + эталонный ответ».
Критично для нумерации: i-я пара в JSON-массиве опирается только на КОНТЕКСТ i — не смешивай факты разных компаний и не переноси сигналы между блоками."""

    user_prompt = f"""
{intro}
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
7) Не повторяй и не перефразируй уже сгенерированные вопросы.

УЖЕ СГЕНЕРИРОВАННЫЕ ВОПРОСЫ (НЕ ПОВТОРЯТЬ):
{chr(10).join(f"- {q}" for q in (existing_questions or [])[-200:]) if existing_questions else "- (пока нет)"}

КОНТЕКСТЫ:
{contexts_text}

Верни только JSON-массив из {count} объектов (ровно {count}):
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
