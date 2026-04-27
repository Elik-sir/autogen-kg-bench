from __future__ import annotations

import random
from typing import Any

from utils.prompt_builder import USEFUL_ENTITY_KEYS

_SCAN_LIMIT = 400
_HOP1_LIMIT = 28
_MAX_PATH_QUERY_LEN = 8


def _embeddingish_key(key: str) -> bool:
    lowered = key.lower()
    return "embedding" in lowered or lowered in {"vector", "vectors"}


def _useful_props(props: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(props, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in props.items():
        ks = str(k)
        if _embeddingish_key(ks):
            continue
        if ks not in USEFUL_ENTITY_KEYS:
            continue
        if isinstance(v, (str, int, float)) and v not in ("", None):
            out[ks] = v
    return out


def _sanitize_props(props: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(props, dict):
        return {}
    return {k: v for k, v in props.items() if not _embeddingish_key(str(k))}


def _hop1_rows(db_manager, node_element_id: str) -> list[dict[str, Any]]:
    q = """
    MATCH (n)-[r]-(m)
    WHERE elementId(n) = $eid
    RETURN type(r) AS rel_type, labels(m) AS node_labels, properties(m) AS node_props
    LIMIT $limit
    """
    return db_manager.run_query(q, {"eid": node_element_id, "limit": _HOP1_LIMIT})


def _format_hop1(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        rt = row.get("rel_type")
        labels = row.get("node_labels") or []
        props = _useful_props(row.get("node_props"))
        if not props and not labels:
            continue
        lines.append(f"  — через {rt}: {list(labels)} {props}")
    return "\n".join(lines) if lines else "  (нет соседей в выборке)"


def _shortest_path_hint(
    db_manager, node_element_id: str, target_element_id: str
) -> str | None:
    q = f"""
    MATCH (a), (c)
    WHERE elementId(a) = $eid_a AND elementId(c) = $eid_c
    MATCH p = shortestPath((a)-[*..{_MAX_PATH_QUERY_LEN}]-(c))
    RETURN length(p) AS hops, [r IN relationships(p) | type(r)] AS rel_types
    LIMIT 1
    """
    rows = db_manager.run_query(
        q, {"eid_a": node_element_id, "eid_c": target_element_id}
    )
    if not rows:
        return None
    hops = rows[0].get("hops")
    rts = rows[0].get("rel_types") or []
    if hops is None:
        return None
    return f"{int(hops)} рёбер: {' → '.join(str(t) for t in rts)}"


def _pick_best_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Один лучший вариант (a,b,common) на пару (a,b): предпочитаем случаи,
    где общая сущность не сосед обоим одновременно (min(da,db) >= 2), иначе короче сумма путей.
    """
    best_row: dict[tuple[str, str], dict[str, Any]] = {}
    best_key: dict[tuple[str, str], tuple] = {}
    for row in rows:
        id_a = row.get("id_a")
        id_b = row.get("id_b")
        if not id_a or not id_b:
            continue
        pkey = (str(id_a), str(id_b))
        da = row.get("dist_a")
        db = row.get("dist_b")
        try:
            da_i = int(da) if da is not None else 99
            db_i = int(db) if db is not None else 99
        except (TypeError, ValueError):
            continue
        deep = 1 if min(da_i, db_i) >= 2 else 0
        score = (deep, -(da_i + db_i), str(row.get("id_common", "")))
        prev = best_key.get(pkey)
        if prev is None or score > prev:
            best_key[pkey] = score
            best_row[pkey] = row
    return list(best_row.values())


def find_same_type_common_contexts(
    db_manager,
    *,
    rng_seed: int | None = 42,
    max_contexts: int = 24,
    scan_limit: int = _SCAN_LIMIT,
) -> list[dict[str, Any]]:
    """
    Пары узлов одной метки без прямого ребра между ними, но с общей сущностью `common`
    в пределах 1–3 шагов от каждого (в т.ч. только через цепочки 2–3 hop, без общего 1-hop соседа).
    """
    q = """
    MATCH (a)-[ra*1..3]-(common)
    MATCH (b)-[rb*1..3]-(common)
    WHERE size(labels(a)) = 1
      AND labels(a) = labels(b)
      AND elementId(a) < elementId(b)
      AND NOT (a)--(b)
      AND elementId(common) <> elementId(a)
      AND elementId(common) <> elementId(b)
    RETURN
      labels(a)[0] AS node_label,
      elementId(a) AS id_a,
      elementId(b) AS id_b,
      elementId(common) AS id_common,
      labels(common) AS common_labels,
      properties(a) AS props_a,
      properties(b) AS props_b,
      properties(common) AS props_common,
      size(ra) AS dist_a,
      size(rb) AS dist_b
    LIMIT $scan_limit
    """
    rows = db_manager.run_query(q, {"scan_limit": scan_limit})
    if not rows:
        return []

    rows = _pick_best_rows(rows)
    rng = random.Random(rng_seed)
    rng.shuffle(rows)

    contexts: list[dict[str, Any]] = []

    for row in rows:
        id_a = row.get("id_a")
        id_b = row.get("id_b")
        id_c = row.get("id_common")
        if not id_a or not id_b or not id_c:
            continue

        u_common = _useful_props(row.get("props_common"))
        if not u_common:
            continue

        da = row.get("dist_a")
        db = row.get("dist_b")
        hop_a = _hop1_rows(db_manager, str(id_a))
        hop_b = _hop1_rows(db_manager, str(id_b))
        hint_a = _shortest_path_hint(db_manager, str(id_a), str(id_c))
        hint_b = _shortest_path_hint(db_manager, str(id_b), str(id_c))

        contexts.append(
            {
                "node_label": row.get("node_label"),
                "id_a": str(id_a),
                "id_b": str(id_b),
                "id_common": str(id_c),
                "dist_a": da,
                "dist_b": db,
                "props_a": _useful_props(row.get("props_a")),
                "props_b": _useful_props(row.get("props_b")),
                "common_labels": list(row.get("common_labels") or []),
                "common_props": u_common,
                "common_props_full": _sanitize_props(row.get("props_common")),
                "hop1_text_a": _format_hop1(hop_a),
                "hop1_text_b": _format_hop1(hop_b),
                "path_hint_a": hint_a,
                "path_hint_b": hint_b,
            }
        )

        if len(contexts) >= max_contexts:
            break

    return contexts
