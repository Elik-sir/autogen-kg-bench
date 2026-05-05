from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from benchmark_generator.prompt_settings import MAX_SUBGRAPH_NODES

_MAX_PROP_VALUE_CHARS = 140
_MAX_PROPS_PER_NODE = 5
_MAX_EDGE_EXAMPLES_PER_REL = 8
_MAX_EDGES = 260
_DROP_PROP_KEYS = {"vector", "vectors", "embedding", "embeddings"}
_PREFERRED_PROP_KEYS = (
    "name",
    "title",
    "ticker",
    "symbol",
    "id",
    "uuid",
    "headline",
    "description",
    "summary",
    "sector",
    "industry",
    "country",
    "region",
    "year",
    "date",
    "risk",
    "impact",
    "score",
    "category",
)


def _safe_label(label: str) -> str:
    return str(label).replace("`", "``")


def _truncate_text(value: Any, max_chars: int = _MAX_PROP_VALUE_CHARS) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return f"{text[:keep_head]}...[{len(text) - max_chars} chars omitted]...{text[-keep_tail:]}"


def _choose_company_label(schema: dict[str, Any]) -> str | None:
    labels = [
        label
        for label, entry in schema.items()
        if isinstance(entry, dict) and str(entry.get("type", "")).lower() == "node"
    ]
    for label in labels:
        if "company" in str(label).lower():
            return label
    return labels[0] if labels else None


def _pick_anchor_candidates(db_manager, company_label: str, limit: int) -> list[dict[str, Any]]:
    query = f"""
    MATCH (c:`{_safe_label(company_label)}`)
    OPTIONAL MATCH (c)-[r]-()
    WITH c, count(r) AS degree, count(DISTINCT type(r)) AS rel_type_variety
    WHERE degree > 0
    ORDER BY degree DESC, rel_type_variety DESC
    RETURN
      elementId(c) AS anchor_id,
      properties(c) AS anchor_props,
      degree,
      rel_type_variety
    LIMIT $limit
    """
    rows = db_manager.run_query(query, {"limit": max(1, int(limit))})
    return rows if isinstance(rows, list) else []


def _dense_subgraph_query() -> str:
    return """
MATCH (anchor)
WHERE elementId(anchor) = $anchor_id
MATCH (anchor)-[*1..2]-(n)
WITH anchor, collect(DISTINCT n) AS near_nodes
WITH [anchor] + near_nodes[..$max_neighbors] AS subgraph_nodes
UNWIND subgraph_nodes AS n
WITH
  subgraph_nodes,
  collect(DISTINCT {
    id: elementId(n),
    labels: labels(n),
    props: properties(n)
  }) AS nodes
UNWIND subgraph_nodes AS n1
MATCH (n1)-[r]-(n2)
WHERE n2 IN subgraph_nodes
WITH
  nodes,
  collect(DISTINCT {
    source: elementId(startNode(r)),
    type: type(r),
    target: elementId(endNode(r))
  })[..$max_edges] AS edges
RETURN nodes, edges
LIMIT 1
""".strip()


def _sanitize_props(raw_props: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_props, dict):
        return {}
    chosen: dict[str, Any] = {}
    for key in _PREFERRED_PROP_KEYS:
        if key not in raw_props:
            continue
        value = raw_props.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, (str, int, float, bool)):
            chosen[key] = _truncate_text(value) if isinstance(value, str) else value
        if len(chosen) >= _MAX_PROPS_PER_NODE:
            return chosen
    for raw_key, raw_value in raw_props.items():
        key = str(raw_key)
        lowered = key.lower()
        if lowered in _DROP_PROP_KEYS or "embedding" in lowered:
            continue
        if raw_value in (None, ""):
            continue
        if isinstance(raw_value, (str, int, float, bool)):
            chosen[key] = _truncate_text(raw_value) if isinstance(raw_value, str) else raw_value
        elif isinstance(raw_value, list):
            normalized = [
                _truncate_text(v) if isinstance(v, str) else v
                for v in raw_value[:3]
                if isinstance(v, (str, int, float, bool))
            ]
            if normalized:
                chosen[key] = normalized
        if len(chosen) >= _MAX_PROPS_PER_NODE:
            break
    return chosen


def _sanitize_nodes(nodes: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id or node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        labels = node.get("labels")
        out.append(
            {
                "id": node_id,
                "labels": labels if isinstance(labels, list) else [],
                "props": _sanitize_props(node.get("props")),
            }
        )
    return out


def _sanitize_edges(edges: list[Any], valid_node_ids: set[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        rel_type = str(edge.get("type") or "").strip()
        if not source or not target or not rel_type:
            continue
        if source not in valid_node_ids or target not in valid_node_ids:
            continue
        key = (source, rel_type, target)
        if key in seen:
            continue
        seen.add(key)
        out.append({"source": source, "type": rel_type, "target": target})
        if len(out) >= _MAX_EDGES:
            break
    return out


def _dense_subgraph_snapshot(db_manager, anchor_id: str) -> dict[str, Any]:
    rows = db_manager.run_query(
        _dense_subgraph_query(),
        {
            "anchor_id": anchor_id,
            "max_neighbors": max(1, MAX_SUBGRAPH_NODES - 1),
            "max_edges": _MAX_EDGES,
        },
    )
    if not isinstance(rows, list) or not rows:
        return {}
    row = rows[0] if isinstance(rows[0], dict) else {}
    nodes = _sanitize_nodes(row.get("nodes") if isinstance(row, dict) else [])
    if not nodes:
        return {}
    valid_ids = {str(n.get("id")) for n in nodes if n.get("id")}
    edges = _sanitize_edges(row.get("edges") if isinstance(row, dict) else [], valid_ids)
    return {"nodes": nodes, "edges": edges}


def _node_primary_label(node: dict[str, Any]) -> str:
    labels = node.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return "Entity"


def _node_key(node: dict[str, Any]) -> str:
    props = node.get("props") if isinstance(node.get("props"), dict) else {}
    for key in ("name", "title", "ticker", "symbol", "id", "uuid"):
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    return str(node.get("id") or "unknown")


def _compute_topology_metrics(nodes: list[dict[str, Any]], edges: list[dict[str, str]]) -> dict[str, Any]:
    node_ids = {str(node.get("id")) for node in nodes if node.get("id")}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    rel_counter: Counter[str] = Counter()
    undirected_edges: set[tuple[str, str]] = set()
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        rel_counter[edge["type"]] += 1
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
            if source != target:
                pair = tuple(sorted((source, target)))
                undirected_edges.add(pair)

    hubs: list[dict[str, Any]] = []
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    for node_id, neighbors in adjacency.items():
        node = node_by_id.get(node_id) or {}
        hubs.append(
            {
                "id": node_id,
                "label": _node_primary_label(node),
                "key": _node_key(node),
                "degree": len(neighbors),
            }
        )
    hubs.sort(key=lambda item: (-int(item["degree"]), str(item["label"]), str(item["key"])))

    node_count = len(node_ids)
    edge_count = len(undirected_edges)
    max_undirected_edges = (node_count * (node_count - 1)) / 2 if node_count > 1 else 0
    density = (edge_count / max_undirected_edges) if max_undirected_edges else 0.0

    return {
        "node_count": node_count,
        "directed_edge_count": len(edges),
        "undirected_edge_count": edge_count,
        "density": round(density, 4),
        "relationship_type_distribution": dict(rel_counter.most_common(8)),
        "top_hubs": hubs[:3],
    }


def _props_preview(props: dict[str, Any]) -> str:
    if not isinstance(props, dict) or not props:
        return "{}"
    parts = [f"{k}={v!r}" for k, v in props.items()]
    return "{ " + ", ".join(parts) + " }"


def _nodes_block(nodes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        label = _node_primary_label(node)
        props = node.get("props") if isinstance(node.get("props"), dict) else {}
        lines.append(f"- {node_id} | {label} | {_props_preview(props)}")
    return "\n".join(lines)


def _edges_by_type_block(edges: list[dict[str, str]], node_by_id: dict[str, dict[str, Any]]) -> str:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        grouped[str(edge["type"])].append((edge["source"], edge["target"]))
    lines: list[str] = []
    for rel_type in sorted(grouped.keys()):
        pairs = grouped[rel_type]
        examples: list[str] = []
        for source, target in pairs[:_MAX_EDGE_EXAMPLES_PER_REL]:
            source_name = _node_key(node_by_id.get(source, {"id": source}))
            target_name = _node_key(node_by_id.get(target, {"id": target}))
            examples.append(f"{source_name} -> {target_name}")
        lines.append(f"- {rel_type} ({len(pairs)}): " + "; ".join(examples))
    return "\n".join(lines)


def _format_subgraph_context(
    *,
    anchor_id: str,
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    nodes = snapshot.get("nodes") if isinstance(snapshot.get("nodes"), list) else []
    edges = snapshot.get("edges") if isinstance(snapshot.get("edges"), list) else []
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    anchor_node = node_by_id.get(anchor_id, {})
    anchor_key = _node_key(anchor_node)
    anchor_label = _node_primary_label(anchor_node)

    top_hubs = metrics.get("top_hubs") if isinstance(metrics.get("top_hubs"), list) else []
    hub_lines = [
        f"- {hub.get('key')} ({hub.get('label')}): degree={hub.get('degree')}" for hub in top_hubs[:3]
    ]
    rel_distribution = metrics.get("relationship_type_distribution") or {}
    rel_summary = ", ".join(f"{k}:{v}" for k, v in rel_distribution.items()) or "none"

    return (
        "SUBGRAPH ANALYTICS CONTEXT\n"
        f"Anchor: {anchor_key} [{anchor_label}] id={anchor_id}\n"
        f"Node count: {metrics.get('node_count', 0)}\n"
        f"Edge count (directed): {metrics.get('directed_edge_count', 0)}\n"
        f"Density (undirected): {metrics.get('density', 0)}\n"
        f"Relationship mix: {rel_summary}\n"
        "\nTOP HUBS:\n"
        + ("\n".join(hub_lines) if hub_lines else "- none")
        + "\n\nNODES:\n"
        + _nodes_block(nodes)
        + "\n\nTOPOLOGY (EDGES GROUPED BY TYPE):\n"
        + _edges_by_type_block(edges, node_by_id)
    ).strip()


def _format_useful_context(
    *,
    anchor_id: str,
    snapshot: dict[str, Any],
    metrics: dict[str, Any],
    max_lines: int = 26,
) -> str:
    nodes = snapshot.get("nodes") if isinstance(snapshot.get("nodes"), list) else []
    edges = snapshot.get("edges") if isinstance(snapshot.get("edges"), list) else []
    node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    anchor_node = node_by_id.get(anchor_id, {})
    lines = [
        "ANALYTICS BRIEF:",
        f"- anchor: {_node_key(anchor_node)} ({_node_primary_label(anchor_node)})",
        f"- scope: {metrics.get('node_count', 0)} nodes, {metrics.get('directed_edge_count', 0)} edges, density={metrics.get('density', 0)}",
    ]
    for idx, hub in enumerate(metrics.get("top_hubs") or []):
        if idx >= 3:
            break
        lines.append(
            f"- hub_{idx + 1}: {hub.get('key')} ({hub.get('label')}), degree={hub.get('degree')}"
        )

    grouped: dict[str, int] = defaultdict(int)
    for edge in edges:
        grouped[str(edge["type"])] += 1
    for rel_type, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0]))[:8]:
        lines.append(f"- rel_mix: {rel_type}={count}")

    seen_nodes: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        if not node_id or node_id == anchor_id or node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        lines.append(
            f"- entity: {_node_key(node)} ({_node_primary_label(node)}) {_props_preview(node.get('props') or {})}"
        )
        if len(lines) >= max_lines:
            break
    return "\n".join(lines).strip()


def build_company_subgraph_contexts(
    db_manager,
    schema: dict[str, Any],
    anchors_limit: int = 3,
) -> list[dict[str, Any]]:
    company_label = _choose_company_label(schema)
    if not company_label:
        return []

    anchors = _pick_anchor_candidates(db_manager, company_label, anchors_limit)
    out: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_id = str(anchor.get("anchor_id") or "").strip()
        if not anchor_id:
            continue
        snapshot = _dense_subgraph_snapshot(db_manager, anchor_id)
        if not snapshot:
            continue
        nodes = snapshot.get("nodes") if isinstance(snapshot.get("nodes"), list) else []
        edges = snapshot.get("edges") if isinstance(snapshot.get("edges"), list) else []
        if len(nodes) < 6 or len(edges) < 6:
            continue
        metrics = _compute_topology_metrics(nodes, edges)
        subgraph_context = _format_subgraph_context(
            anchor_id=anchor_id,
            snapshot=snapshot,
            metrics=metrics,
        )
        useful_context = _format_useful_context(
            anchor_id=anchor_id,
            snapshot=snapshot,
            metrics=metrics,
        )
        node_by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
        anchor_props = node_by_id.get(anchor_id, {}).get("props") or {}
        out.append(
            {
                "company_label": company_label,
                "anchor_id": anchor_id,
                "anchor_props": anchor_props,
                "subgraph_context": subgraph_context,
                "useful_context": useful_context,
                "topology_metrics": metrics,
                "subgraph_nodes": nodes,
                "subgraph_edges": edges,
                "debug_cypher": _dense_subgraph_query(),
                "debug_params": {
                    "anchor_id": anchor_id,
                    "max_neighbors": max(1, MAX_SUBGRAPH_NODES - 1),
                    "max_edges": _MAX_EDGES,
                },
            }
        )
    return out
