"""
Подбор сущностей заданной метки (например Person), чтобы покрыть все типы связей,
которые схема APOC допускает для этой метки.

Схема (apoc.meta.schema) задаёт «вселенную» типов рёбер инцидентных узлу с меткой L.
В данных у конкретного узла может быть лишь подмножество — суперузел с максимальной
степенью не гарантирует полное покрытие типов. Здесь жадный set cover: на каждом шаге
берём узел, закрывающий максимум ещё не покрытых типов, пока не исчерпаем цель или кандидаты.

Если в графе нет ни одного экземпляра некоторого типа из схемы, этот тип попадёт в missing.
"""

from __future__ import annotations

from typing import Any, Iterable


def required_relationship_types_from_schema(schema: dict[str, Any], label: str) -> set[str]:
    """
    Типы связей из apoc.meta.schema() для узла с меткой `label`.
    Берём ключи словаря relationships у записи узла.
    """
    entry = schema.get(label)
    if entry is None:
        raise KeyError(f"В схеме нет метки {label!r}")
    if entry.get("type") != "node":
        raise ValueError(f"{label!r} в схеме не является узлом (type={entry.get('type')!r})")
    rels = entry.get("relationships") or {}
    if not isinstance(rels, dict):
        return set()
    return set(rels.keys())


def _greedy_set_cover(
    universe: set[str],
    candidates: list[tuple[str, set[str]]],
) -> tuple[list[str], set[str]]:
    """
    universe — что нужно покрыть; candidates — (element_id, множество типов рёбер узла).
    Возвращает (выбранные id по порядку, оставшиеся непокрытыми типы).
    """
    uncovered = set(universe)
    chosen: list[str] = []
    id_to_types = {nid: set(ts) for nid, ts in candidates}

    while uncovered:
        best_id: str | None = None
        best_gain = 0
        for nid in id_to_types:
            if nid in chosen:
                continue
            gain = len(id_to_types[nid] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_id = nid
        if best_id is None or best_gain == 0:
            break
        chosen.append(best_id)
        uncovered -= id_to_types[best_id]

    return chosen, uncovered


def fetch_label_nodes_incident_rel_types(
    db_manager,
    label: str,
) -> list[tuple[str, set[str]]]:
    """
    Для каждого узла с меткой `label`: elementId и множество type(r) для всех
    инцидентных рёбер (вход + выход). Результат стримится драйвером — не материализуем граф в Python заранее.
    """
    # Экранируем метку для инъекций в шаблоне (метка из схемы / конфига, не от пользователя в сыром виде)
    safe = label.replace("`", "``")
    q = f"""
    MATCH (n:`{safe}`)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, collect(DISTINCT type(r)) AS types
    WHERE any(t IN types WHERE t IS NOT NULL)
    RETURN elementId(n) AS element_id, [t IN types WHERE t IS NOT NULL] AS rel_types
    """
    rows = db_manager.run_query(q)
    out: list[tuple[str, set[str]]] = []
    for row in rows:
        eid = row["element_id"]
        types = row.get("rel_types") or []
        out.append((eid, set(types)))
    return out


def select_nodes_covering_schema_rel_types(
    schema: dict[str, Any],
    label: str,
    db_manager,
) -> dict[str, Any]:
    """
    Возвращает:
      - required: типы из схемы для метки
      - present_in_data: объединение типов, встречающихся у узлов с меткой
      - selected_element_ids: жадный набор узлов, покрывающий required ∩ present_in_data
      - missing_in_graph: типы из схемы, ни у одного узла с меткой не встречаются
      - unreachable_schema_types: синоним missing_in_graph (явно для отчёта)
    """
    required = required_relationship_types_from_schema(schema, label)
    candidates = fetch_label_nodes_incident_rel_types(db_manager, label)

    present_in_data: set[str] = set()
    for _, ts in candidates:
        present_in_data |= ts

    target = required & present_in_data
    missing_in_graph = required - present_in_data

    chosen, still_uncovered = _greedy_set_cover(target, candidates)

    return {
        "label": label,
        "required_rel_types": sorted(required),
        "present_rel_types_in_data": sorted(present_in_data),
        "target_rel_types_to_cover": sorted(target),
        "selected_element_ids": chosen,
        "missing_in_graph": sorted(missing_in_graph),
        "uncovered_after_greedy": sorted(still_uncovered),
        "candidate_node_count": len(candidates),
    }
