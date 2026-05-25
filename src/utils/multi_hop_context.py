from __future__ import annotations

import random
from typing import Any

from utils.prompt_builder import USEFUL_ENTITY_KEYS

_SCAN_LIMIT = 300
_MAX_CONTEXTS_DEFAULT = 24


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


def _has_anchor(props: dict[str, Any] | None) -> bool:
    up = _useful_props(props)
    return bool(up.get("name") or up.get("ticker") or up.get("title"))


def _format_node_line(labels: list[str], props: dict[str, Any]) -> str:
    useful = _useful_props(props)
    if useful:
        return f"{list(labels)} {useful}"
    compact = _sanitize_props(props)
    trimmed = {}
    for k, v in compact.items():
        if isinstance(v, (str, int, float)) and v not in ("", None):
            trimmed[str(k)] = v
        if len(trimmed) >= 6:
            break
    return f"{list(labels)} {trimmed or compact}"


def _format_path_text(nodes: list[dict[str, Any]], relationships: list[str]) -> str:
    if not nodes:
        return ""
    lines = [f"Start: {_format_node_line(nodes[0]['labels'], nodes[0]['props'])}"]
    for rel, node in zip(relationships, nodes[1:]):
        lines.append(f"  -[:{rel}]-> {_format_node_line(node['labels'], node['props'])}")
    return "\n".join(lines)


def _path_signature(nodes: list[dict[str, Any]], relationships: list[str]) -> tuple:
    start = _useful_props(nodes[0]["props"]) if nodes else {}
    end = _useful_props(nodes[-1]["props"]) if nodes else {}
    start_key = start.get("ticker") or start.get("name") or start.get("title") or nodes[0].get("id")
    end_key = end.get("ticker") or end.get("name") or end.get("title") or nodes[-1].get("id")
    label_chain = tuple(tuple(n["labels"]) for n in nodes)
    return (tuple(relationships), label_chain, str(start_key), str(end_key))


def _row_to_context(row: dict[str, Any], hop_count: int) -> dict[str, Any] | None:
    node_keys = ["a", "b", "c", "d"][: hop_count + 1]
    nodes: list[dict[str, Any]] = []
    for key in node_keys:
        node_id = row.get(f"id_{key}")
        if not node_id:
            return None
        nodes.append(
            {
                "id": str(node_id),
                "labels": list(row.get(f"labels_{key}") or []),
                "props": _sanitize_props(row.get(f"props_{key}")),
                "useful_props": _useful_props(row.get(f"props_{key}")),
            }
        )
    relationships = [str(row.get(f"rel{idx}") or "") for idx in range(1, hop_count + 1)]
    if any(not rel for rel in relationships):
        return None
    if not _has_anchor(nodes[0]["props"]):
        return None
    return {
        "hop_count": hop_count,
        "nodes": nodes,
        "relationships": relationships,
        "path_text": _format_path_text(nodes, relationships),
        "path_signature": _path_signature(nodes, relationships),
    }


def _score_context(ctx: dict[str, Any]) -> tuple:
    nodes = ctx.get("nodes") or []
    useful_count = sum(len(n.get("useful_props") or {}) for n in nodes)
    end_useful = len(nodes[-1].get("useful_props") or {}) if nodes else 0
    rel_variety = len(set(ctx.get("relationships") or []))
    label_variety = len({tuple(n.get("labels") or []) for n in nodes})
    return (useful_count, end_useful, rel_variety, label_variety)


def _path_query(hop_count: int) -> str:
    if hop_count == 2:
        return """
        MATCH (a)-[r1]->(b)-[r2]->(c)
        WHERE elementId(a) <> elementId(b)
          AND elementId(b) <> elementId(c)
          AND elementId(a) <> elementId(c)
        RETURN
          elementId(a) AS id_a, labels(a) AS labels_a, properties(a) AS props_a,
          type(r1) AS rel1,
          elementId(b) AS id_b, labels(b) AS labels_b, properties(b) AS props_b,
          type(r2) AS rel2,
          elementId(c) AS id_c, labels(c) AS labels_c, properties(c) AS props_c
        LIMIT $scan_limit
        """
    if hop_count == 3:
        return """
        MATCH (a)-[r1]->(b)-[r2]->(c)-[r3]->(d)
        WHERE elementId(a) <> elementId(b)
          AND elementId(b) <> elementId(c)
          AND elementId(c) <> elementId(d)
          AND elementId(a) <> elementId(c)
          AND elementId(a) <> elementId(d)
          AND elementId(b) <> elementId(d)
        RETURN
          elementId(a) AS id_a, labels(a) AS labels_a, properties(a) AS props_a,
          type(r1) AS rel1,
          elementId(b) AS id_b, labels(b) AS labels_b, properties(b) AS props_b,
          type(r2) AS rel2,
          elementId(c) AS id_c, labels(c) AS labels_c, properties(c) AS props_c,
          type(r3) AS rel3,
          elementId(d) AS id_d, labels(d) AS labels_d, properties(d) AS props_d
        LIMIT $scan_limit
        """
    raise ValueError("hop_count must be 2 or 3")


def find_multi_hop_path_contexts(
    db_manager,
    hop_count: int,
    *,
    rng_seed: int | None = 42,
    max_contexts: int = _MAX_CONTEXTS_DEFAULT,
    scan_limit: int = _SCAN_LIMIT,
) -> list[dict[str, Any]]:
    """
    Находит реальные directed-пути длины hop_count в Neo4j и возвращает контексты для промпта.
    """
    if hop_count not in (2, 3):
        raise ValueError("hop_count must be 2 or 3")

    rows = db_manager.run_query(_path_query(hop_count), {"scan_limit": scan_limit})
    if not rows:
        return []

    candidates: list[dict[str, Any]] = []
    seen_signatures: set[tuple] = set()
    for row in rows:
        ctx = _row_to_context(row, hop_count)
        if not ctx:
            continue
        signature = ctx.get("path_signature")
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        candidates.append(ctx)

    candidates.sort(key=_score_context, reverse=True)

    rng = random.Random(rng_seed)
    top_pool = candidates[: max(max_contexts * 3, max_contexts)]
    rng.shuffle(top_pool)

    out: list[dict[str, Any]] = []
    used_signatures: set[tuple] = set()
    for ctx in top_pool:
        signature = ctx.get("path_signature")
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        out.append(ctx)
        if len(out) >= max_contexts:
            break
    return out
