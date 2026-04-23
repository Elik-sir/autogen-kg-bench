
from __future__ import annotations

from utils.rel_type_cover import select_nodes_covering_schema_rel_types


def get_schema(db_manager):
    """Извлекает схему БД, чтобы LLM не галлюцинировала значения."""
    print("Извлечение схемы данных...")

    schema_result = db_manager.run_query("CALL apoc.meta.schema() YIELD value RETURN value")
    if not schema_result:
        raise ValueError("APOC не установлен или база пуста.")
    schema = schema_result[0]["value"]

    return schema

def get_samples(db_manager):
    """Извлекает примеры реальных данных, чтобы LLM не галлюцинировала значения."""
    print("Извлечение примеров данных...")

    try:
        schema = get_schema(db_manager)
        node_labels = [
            label
            for label, entry in schema.items()
            if isinstance(entry, dict) and entry.get("type") == "node"
        ]

        samples: dict[str, list[dict]] = {}
        fallback_labels: list[str] = []
        labels_with_missing_rel_types: dict[str, list[str]] = {}

        for label in node_labels:
            cover_result = select_nodes_covering_schema_rel_types(schema, label, db_manager)
            selected_ids = cover_result.get("selected_element_ids", [])
            missing_rel_types = cover_result.get("missing_in_graph", [])
            if missing_rel_types:
                labels_with_missing_rel_types[label] = missing_rel_types

            # Если по покрытию ничего не выбралось (например, label без рёбер),
            # берём до 2 узлов как fallback.
            if not selected_ids:
                safe = label.replace("`", "``")
                fallback_query = f"""
                MATCH (n:`{safe}`)
                RETURN properties(n) AS props
                LIMIT 2
                """
                fallback_rows = db_manager.run_query(fallback_query)
                samples[label] = [row["props"] for row in fallback_rows if row.get("props")]
                fallback_labels.append(label)
                continue

            samples_query = """
            UNWIND $ids AS eid
            MATCH (n)
            WHERE elementId(n) = eid
            RETURN properties(n) AS props
            """
            samples_rows = db_manager.run_query(samples_query, {"ids": selected_ids})
            samples[label] = [row["props"] for row in samples_rows if row.get("props")]

        print(
            f"[SAMPLES] labels={len(node_labels)}, fallback={len(fallback_labels)}, "
            f"with_missing_rel_types={len(labels_with_missing_rel_types)}"
        )
        if fallback_labels:
            print(f"[SAMPLES] Fallback labels: {', '.join(sorted(fallback_labels))}")
        for label in sorted(labels_with_missing_rel_types):
            missing = ", ".join(labels_with_missing_rel_types[label])
            print(f"[SAMPLES] Missing rel types for {label}: {missing}")
    except Exception as error:
        print(f"Не удалось получить примеры (возможно нет APOC): {error}")
        samples = "Примеры недоступны"

    return samples