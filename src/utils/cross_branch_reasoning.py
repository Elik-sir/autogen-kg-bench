from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BranchPath:
    """Описание одной ветки от anchor до целевой сущности."""

    start_label: str
    rel_type: str
    end_label: str


def _node_labels_from_schema(schema: dict[str, Any]) -> list[str]:
    return [
        label
        for label, entry in schema.items()
        if isinstance(entry, dict) and entry.get("type") == "node"
    ]


def _candidate_anchor_labels(schema: dict[str, Any]) -> list[str]:
    anchors: list[str] = []
    for label in _node_labels_from_schema(schema):
        entry = schema.get(label, {})
        rels = entry.get("relationships") or {}
        if isinstance(rels, dict) and len(rels) >= 2:
            anchors.append(label)
    return anchors


def _safe_label(label: str) -> str:
    return label.replace("`", "``")


def _extract_rel_edges_for_label(schema: dict[str, Any], label: str) -> list[BranchPath]:
    entry = schema.get(label, {})
    rels = entry.get("relationships") or {}
    if not isinstance(rels, dict):
        return []

    edges: list[BranchPath] = []
    for rel_type in rels.keys():
        # В apoc.meta.schema не всегда можно детерминированно достать "end label",
        # поэтому end_label определяется на уровне данных (см. _discover_end_labels).
        edges.append(BranchPath(start_label=label, rel_type=rel_type, end_label="*"))
    return edges


def _discover_end_labels(db_manager, anchor_label: str, rel_type: str, limit: int = 5) -> list[str]:
    q = f"""
    MATCH (a:`{_safe_label(anchor_label)}`)-[r:`{rel_type}`]-(b)
    WITH DISTINCT labels(b) AS lbs
    UNWIND lbs AS l
    RETURN DISTINCT l AS label
    LIMIT $limit
    """
    rows = db_manager.run_query(q, {"limit": limit})
    return [row["label"] for row in rows if row.get("label")]


def _sample_anchor_entity(db_manager, anchor_label: str) -> dict[str, Any] | None:
    q = f"""
    MATCH (a:`{_safe_label(anchor_label)}`)
    RETURN elementId(a) AS anchor_id, properties(a) AS anchor_props
    LIMIT 1
    """
    rows = db_manager.run_query(q)
    if not rows:
        return None
    return rows[0]


def _pick_anchor_identifier(anchor_props: dict[str, Any]) -> tuple[str, Any] | None:
    preferred_keys = ("name", "title", "ticker", "id", "uuid")
    for key in preferred_keys:
        if key in anchor_props and anchor_props[key] not in (None, ""):
            return key, anchor_props[key]

    for key, value in anchor_props.items():
        if isinstance(value, (str, int, float)) and value not in ("", None):
            return key, value
    return None


def _entity_mask(label: str, rel_type: str) -> str:
    return f"[MASK::{label} via {rel_type}]"


def _build_cross_branch_cypher(
    anchor_label: str,
    anchor_id_key: str,
    anchor_id_value: Any,
    branch_a: BranchPath,
    branch_b: BranchPath,
) -> str:
    # Здесь используем симметричный паттерн -(r)-, чтобы не завязываться
    # на направление, если оно неоднородно в данных.
    return (
        f"MATCH (a:`{_safe_label(anchor_label)}` {{{anchor_id_key}: $anchor_value}})\n"
        f"MATCH (a)-[:`{branch_a.rel_type}`]-(x:`{_safe_label(branch_a.end_label)}`)\n"
        f"MATCH (a)-[:`{branch_b.rel_type}`]-(y:`{_safe_label(branch_b.end_label)}`)\n"
        "RETURN\n"
        "  properties(a) AS anchor,\n"
        "  collect(DISTINCT properties(x)) AS branch_a_entities,\n"
        "  collect(DISTINCT properties(y)) AS branch_b_entities,\n"
        f"  '{branch_a.rel_type}' AS branch_a_rel_type,\n"
        f"  '{branch_b.rel_type}' AS branch_b_rel_type"
    )


def _build_masked_question(anchor_label: str, branch_a: BranchPath, branch_b: BranchPath) -> str:
    mask_a = _entity_mask(branch_a.end_label, branch_a.rel_type)
    mask_b = _entity_mask(branch_b.end_label, branch_b.rel_type)
    return (
        f"Сделай аналитическое резюме для узла типа {anchor_label}: "
        f"как сущности {mask_a} и {mask_b} совместно влияют на итоговый контекст? "
        "Ответ должен объединять обе ветки и объяснять скрытые зависимости."
    )


def generate_cross_branch_reasoning_items(
    db_manager,
    schema: dict[str, Any],
    items_count: int = 5,
    rng_seed: int | None = 42,
) -> list[dict[str, Any]]:
    """
    Генерирует задачи типа "summarization/analytics" по алгоритму Cross-Branch Reasoning:
      1) Выбор anchor-метки.
      2) Извлечение двух независимых веток.
      3) Маскирование прямых названий целевых сущностей в тексте вопроса.
      4) Подготовка Cypher для проверки gold-context.

    Возвращает элементы в совместимом формате:
      {
        "complexity": "cross-branch",
        "question": "...",
        "cypher": "...",
        "params": {"anchor_value": ...},
        "reasoning_type": "summarization_analytics",
        "metadata": {...}
      }
    """
    rnd = random.Random(rng_seed)
    out: list[dict[str, Any]] = []

    anchor_labels = _candidate_anchor_labels(schema)
    rnd.shuffle(anchor_labels)

    for anchor_label in anchor_labels:
        if len(out) >= items_count:
            break

        anchor_entity = _sample_anchor_entity(db_manager, anchor_label)
        if not anchor_entity:
            continue

        anchor_props = anchor_entity.get("anchor_props") or {}
        anchor_identifier = _pick_anchor_identifier(anchor_props)
        if not anchor_identifier:
            continue
        anchor_id_key, anchor_id_value = anchor_identifier

        all_rel_edges = _extract_rel_edges_for_label(schema, anchor_label)
        if len(all_rel_edges) < 2:
            continue

        # Для каждой связи уточняем end-label в данных.
        expanded: list[BranchPath] = []
        for edge in all_rel_edges:
            end_labels = _discover_end_labels(db_manager, anchor_label, edge.rel_type)
            for end_label in end_labels:
                if end_label == anchor_label:
                    continue
                expanded.append(
                    BranchPath(
                        start_label=edge.start_label,
                        rel_type=edge.rel_type,
                        end_label=end_label,
                    )
                )

        if len(expanded) < 2:
            continue

        # Ищем 2 разные ветки: разный rel_type и разная целевая метка.
        branch_pair: tuple[BranchPath, BranchPath] | None = None
        for i in range(len(expanded)):
            for j in range(i + 1, len(expanded)):
                a = expanded[i]
                b = expanded[j]
                if a.rel_type != b.rel_type and a.end_label != b.end_label:
                    branch_pair = (a, b)
                    break
            if branch_pair:
                break

        if not branch_pair:
            continue
        branch_a, branch_b = branch_pair

        cypher = _build_cross_branch_cypher(
            anchor_label=anchor_label,
            anchor_id_key=anchor_id_key,
            anchor_id_value=anchor_id_value,
            branch_a=branch_a,
            branch_b=branch_b,
        )
        question = _build_masked_question(anchor_label, branch_a, branch_b)

        out.append(
            {
                "complexity": "cross-branch",
                "reasoning_type": "summarization_analytics",
                "question": question,
                "cypher": cypher,
                "params": {"anchor_value": anchor_id_value},
                "metadata": {
                    "anchor_label": anchor_label,
                    "anchor_identifier_key": anchor_id_key,
                    "branch_a_rel_type": branch_a.rel_type,
                    "branch_a_end_label": branch_a.end_label,
                    "branch_b_rel_type": branch_b.rel_type,
                    "branch_b_end_label": branch_b.end_label,
                    "success_criterion": (
                        "Framework reconstructs both chains "
                        "Anchor->Entity_A and Anchor->Entity_B, then merges contexts."
                    ),
                },
            }
        )

    return out

