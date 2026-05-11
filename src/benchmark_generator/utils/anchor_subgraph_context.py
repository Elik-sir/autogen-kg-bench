from __future__ import annotations

from collections import defaultdict
import random
from typing import Any

from benchmark_generator.prompt_settings import ANCHOR_PRESELECT_CAP, PATH_EXPAND_BEAM


def _safe_label(label: str) -> str:
    return str(label).replace("`", "``")


def _stringify_value(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).strip()
    if len(text) <= 120:
        return text
    return text[:117] + "..."


def _pick_node_display_props(props: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(props, dict):
        return {}
    preferred_keys = (
        "name",
        "title",
        "ticker",
        "id",
        "uuid",
        "symbol",
        "category",
        "industry",
        "country",
        "date",
        "year",
    )
    out: dict[str, Any] = {}
    for key in preferred_keys:
        if key in props and props.get(key) not in (None, ""):
            out[key] = props.get(key)
        if len(out) >= 4:
            break
    if out:
        return out

    for idx, (key, value) in enumerate(props.items()):
        if idx >= 4:
            break
        if value in (None, ""):
            continue
        out[str(key)] = value
    return out


def _node_title(node: dict[str, Any]) -> str:
    labels = node.get("labels") or []
    props = node.get("props") or {}
    label = labels[0] if labels else "Node"
    name = (
        props.get("name")
        or props.get("title")
        or props.get("ticker")
        or props.get("id")
        or props.get("uuid")
        or node.get("element_id")
        or "?"
    )
    return f"({label}: {name})"


def get_all_node_labels(db_manager) -> list[str]:
    rows = db_manager.run_query("CALL db.labels() YIELD label RETURN label ORDER BY label")
    labels = [str(row.get("label", "")).strip() for row in rows if row.get("label")]
    return [label for label in labels if label]


def get_anchor_candidates_by_label(
    db_manager,
    *,
    label: str,
    limit_per_label: int,
) -> list[dict[str, Any]]:
    # Раньше: подзапрос с [*2..4] и подсчётом всех простых путей — очень дорого на плотных графах.
    # Сейчас: топ узлов по разнообразию/степени + EXISTS на один простой 2-hop (планировщик обычно обрывает на первом совпадении).
    raw_pre = int(max(limit_per_label * 5, 50))
    preselect_limit = min(raw_pre, ANCHOR_PRESELECT_CAP)
    query = f"""
    MATCH (n:`{_safe_label(label)}`)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) AS degree, count(DISTINCT type(r)) AS diversity
    ORDER BY diversity DESC, degree DESC
    LIMIT $preselect_limit
    WITH n, degree, diversity,
      EXISTS {{
        MATCH (n)-[r1]-(m)-[r2]-(t)
        WHERE elementId(t) <> elementId(n)
          AND elementId(m) <> elementId(n)
          AND elementId(m) <> elementId(t)
          AND r1 <> r2
      }} AS has_simple_2hop
    ORDER BY has_simple_2hop DESC, diversity DESC, degree DESC
    LIMIT $limit_per_label
    RETURN
      elementId(n) AS element_id,
      labels(n) AS labels,
      properties(n) AS props,
      degree,
      diversity,
      CASE WHEN has_simple_2hop THEN 2 ELSE 0 END AS max_hops,
      CASE WHEN has_simple_2hop THEN 1 ELSE 0 END AS long_path_count
    """
    rows = db_manager.run_query(
        query,
        {
            "limit_per_label": int(max(1, limit_per_label)),
            "preselect_limit": preselect_limit,
        },
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        element_id = row.get("element_id")
        if not element_id:
            continue
        out.append(
            {
                "label": label,
                "node_id": None,
                "element_id": element_id,
                "labels": row.get("labels") or [label],
                "props": row.get("props") or {},
                "degree": int(row.get("degree") or 0),
                "diversity": int(row.get("diversity") or 0),
                "max_hops": int(row.get("max_hops") or 0),
                "long_path_count": int(row.get("long_path_count") or 0),
            }
        )
    return out


def get_stratified_anchor_pool(
    db_manager,
    *,
    limit_per_label: int,
) -> dict[str, list[dict[str, Any]]]:
    labels = get_all_node_labels(db_manager)
    n_labels = len(labels)
    print(
        f"[anchors] Пул якорей: {n_labels} меток (быстрый прескоринг: степень + EXISTS 2-hop, cap={ANCHOR_PRESELECT_CAP})...",
        flush=True,
    )
    out: dict[str, list[dict[str, Any]]] = {}
    step = max(1, n_labels // 8)
    for i, label in enumerate(labels, start=1):
        show = n_labels <= 15 or i == 1 or i == n_labels or (i % step == 0)
        if show:
            print(f"[anchors] метка {i}/{n_labels}: `{label}`", flush=True)
        anchors = get_anchor_candidates_by_label(
            db_manager,
            label=label,
            limit_per_label=limit_per_label,
        )
        if anchors:
            out[label] = anchors
    print(f"[anchors] Готово: якоря для {len(out)} меток.", flush=True)
    return out


def _extract_unique_paths_for_anchor(
    db_manager,
    *,
    anchor_element_id: str,
    hop_count: int,
    max_paths: int,
) -> list[dict[str, Any]]:
    """Простые пути фиксированной длины k (без повторов узлов и рёбер).

    Для k∈{2,3} используется пошаговое расширение с лимитом beam и случайной прореживкой —
    O(beam·degree) вместо полного перебора variable-length path + ORDER BY rand().
    """
    safe_hop_count = int(max(1, hop_count))
    max_paths_i = int(max(1, max_paths))
    beam = max(max_paths_i, PATH_EXPAND_BEAM)

    if safe_hop_count == 2:
        query = """
        MATCH (anchor)
        WHERE elementId(anchor) = $anchor_element_id
        MATCH (anchor)-[r1]-(n1)
        WHERE elementId(n1) <> elementId(anchor)
        WITH anchor, r1, n1
        ORDER BY rand()
        LIMIT $beam
        MATCH (n1)-[r2]-(target)
        WHERE elementId(target) <> elementId(anchor)
          AND elementId(target) <> elementId(n1)
          AND r1 <> r2
        WITH anchor, r1, n1, r2, target
        ORDER BY rand()
        LIMIT $max_paths
        RETURN
          [
            { element_id: elementId(anchor), node_id: null, labels: labels(anchor), props: properties(anchor) },
            { element_id: elementId(n1), node_id: null, labels: labels(n1), props: properties(n1) },
            { element_id: elementId(target), node_id: null, labels: labels(target), props: properties(target) }
          ] AS nodes,
          [ { type: type(r1) }, { type: type(r2) } ] AS relationships
        """
    elif safe_hop_count == 3:
        query = """
        MATCH (anchor)
        WHERE elementId(anchor) = $anchor_element_id
        MATCH (anchor)-[r1]-(n1)
        WHERE elementId(n1) <> elementId(anchor)
        WITH anchor, r1, n1
        ORDER BY rand()
        LIMIT $beam
        MATCH (n1)-[r2]-(n2)
        WHERE elementId(n2) <> elementId(anchor)
          AND elementId(n2) <> elementId(n1)
          AND r1 <> r2
        WITH anchor, r1, n1, r2, n2
        ORDER BY rand()
        LIMIT $beam
        MATCH (n2)-[r3]-(target)
        WHERE elementId(target) <> elementId(anchor)
          AND elementId(target) <> elementId(n1)
          AND elementId(target) <> elementId(n2)
          AND r3 <> r2 AND r3 <> r1
        WITH anchor, r1, n1, r2, n2, r3, target
        ORDER BY rand()
        LIMIT $max_paths
        RETURN
          [
            { element_id: elementId(anchor), node_id: null, labels: labels(anchor), props: properties(anchor) },
            { element_id: elementId(n1), node_id: null, labels: labels(n1), props: properties(n1) },
            { element_id: elementId(n2), node_id: null, labels: labels(n2), props: properties(n2) },
            { element_id: elementId(target), node_id: null, labels: labels(target), props: properties(target) }
          ] AS nodes,
          [ { type: type(r1) }, { type: type(r2) }, { type: type(r3) } ] AS relationships
        """
    else:
        query = f"""
        MATCH (anchor)
        WHERE elementId(anchor) = $anchor_element_id
        MATCH p=(anchor)-[*1..{safe_hop_count}]-(target)
        WHERE length(p) = {safe_hop_count}
          AND elementId(target) <> elementId(anchor)
          AND ALL(rel IN relationships(p) WHERE single(x IN relationships(p) WHERE x = rel))
          AND ALL(node IN nodes(p) WHERE single(x IN nodes(p) WHERE x = node))
        WITH p
        ORDER BY rand()
        LIMIT $max_paths
        RETURN
          [n IN nodes(p) | {{
            element_id: elementId(n),
            node_id: null,
            labels: labels(n),
            props: properties(n)
          }}] AS nodes,
          [r IN relationships(p) | {{
            type: type(r)
          }}] AS relationships
        """

    params = {
        "anchor_element_id": anchor_element_id,
        "max_paths": max_paths_i,
        "beam": int(beam),
    }
    if safe_hop_count not in (2, 3):
        del params["beam"]

    rows = db_manager.run_query(query, params)
    out: list[dict[str, Any]] = []
    for row in rows:
        nodes = row.get("nodes") or []
        rels = row.get("relationships") or []
        if not nodes or not rels:
            continue
        if len(rels) != safe_hop_count:
            continue
        if len(nodes) != safe_hop_count + 1:
            continue
        out.append({"nodes": nodes, "relationships": rels})
    return out


def _build_local_ontology_text(
    *,
    anchor: dict[str, Any],
    paths: list[dict[str, Any]],
    hop_count: int,
) -> str:
    label_set: set[str] = set()
    rel_type_set: set[str] = set()
    entities: dict[str, dict[str, Any]] = {}
    chain_examples: list[str] = []

    for path in paths:
        nodes = path.get("nodes") or []
        rels = path.get("relationships") or []
        chunks: list[str] = []
        for idx, node in enumerate(nodes):
            element_id = str(node.get("element_id") or "")
            labels = [str(x) for x in (node.get("labels") or []) if str(x).strip()]
            props = node.get("props") or {}
            for lbl in labels:
                label_set.add(lbl)
            if element_id and element_id not in entities:
                entities[element_id] = {
                    "labels": labels,
                    "node_id": node.get("node_id"),
                    "props": _pick_node_display_props(props),
                }
            chunks.append(_node_title(node))
            if idx < len(rels):
                rel_type = str(rels[idx].get("type") or "").strip()
                if rel_type:
                    rel_type_set.add(rel_type)
                chunks.append(f"-[{rel_type or '?'}]->")
        if chunks:
            chain_examples.append(" ".join(chunks))

    lines: list[str] = []
    lines.append(f"HOP TARGET: {hop_count}")
    lines.append("LOCAL ONTOLOGY (only from extracted paths):")
    lines.append(f"- Node labels: {', '.join(sorted(label_set)) if label_set else '(none)'}")
    lines.append(
        f"- Relationship types: {', '.join(sorted(rel_type_set)) if rel_type_set else '(none)'}"
    )
    lines.append("")
    lines.append("ANCHOR:")
    lines.append(
        "- "
        + _node_title(
            {
                "labels": anchor.get("labels") or [anchor.get("label") or "Node"],
                "props": anchor.get("props") or {},
                "element_id": anchor.get("element_id"),
            }
        )
        + f" [id={anchor.get('node_id')}, element_id={anchor.get('element_id')}]"
    )
    lines.append("")
    lines.append("UNIQUE ENTITIES:")
    for entity_id, entity in list(entities.items())[:40]:
        labels = entity.get("labels") or []
        label = labels[0] if labels else "Node"
        props = entity.get("props") or {}
        props_text = ", ".join(
            f"{k}={_stringify_value(v)}" for k, v in list(props.items())[:4]
        ) or "no-key-props"
        lines.append(
            f"- {label} [element_id={entity_id}, id={entity.get('node_id')}]: {props_text}"
        )
    lines.append("")
    lines.append("PATH EXAMPLES:")
    for chain in chain_examples[:12]:
        lines.append(f"- {chain}")

    return "\n".join(lines).strip()


def build_anchor_subgraph_context(
    db_manager,
    *,
    anchor: dict[str, Any],
    hop_count: int,
    max_paths_per_anchor: int,
) -> dict[str, Any] | None:
    anchor_element_id = str(anchor.get("element_id") or "").strip()
    if not anchor_element_id:
        return None

    paths = _extract_unique_paths_for_anchor(
        db_manager,
        anchor_element_id=anchor_element_id,
        hop_count=hop_count,
        max_paths=max_paths_per_anchor,
    )
    if not paths:
        return None

    local_ontology = _build_local_ontology_text(anchor=anchor, paths=paths, hop_count=hop_count)
    return {
        "anchor_label": anchor.get("label"),
        "anchor_labels": anchor.get("labels") or [],
        "anchor_node_id": anchor.get("node_id"),
        "anchor_element_id": anchor_element_id,
        "anchor_props": anchor.get("props") or {},
        "hop_count": hop_count,
        "paths_found": len(paths),
        "paths": paths,
        "local_ontology": local_ontology,
    }


def build_balanced_anchor_order(anchor_pool: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped = {label: list(anchors) for label, anchors in (anchor_pool or {}).items() if anchors}
    for anchors in grouped.values():
        random.shuffle(anchors)
    cursors = defaultdict(int)
    order: list[dict[str, Any]] = []
    while True:
        progressed = False
        for label in sorted(grouped.keys()):
            anchors = grouped[label]
            idx = cursors[label]
            if idx >= len(anchors):
                continue
            order.append(anchors[idx])
            cursors[label] += 1
            progressed = True
        if not progressed:
            break
    return order
