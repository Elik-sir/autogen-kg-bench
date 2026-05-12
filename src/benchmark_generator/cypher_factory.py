from __future__ import annotations

from dataclasses import dataclass
from typing import Any


IDENTITY_KEYS = ("name", "title", "ticker", "id", "uuid", "symbol")
NUMERIC_KEYS = ("amount", "value", "revenue", "profit", "score", "impact", "year")
RELATION_HINTS = {
    "INVESTED_IN": "investment ties",
    "OWNS": "ownership links",
    "CEO_OF": "executive leadership links",
    "MENTIONED_IN": "news co-mentions",
    "OPERATES_IN_INDUSTRY": "industry links",
    "PRODUCES": "product portfolio links",
    "WORKS_AT": "employment links",
    "SUPPLIES": "supply-chain links",
    "PARTNERS_WITH": "partnership links",
    "DEVELOPS": "technology development links",
    "BELONGS_TO": "classification links",
}
TECHNICAL_LABELS = {
    "searchable",
    "embedding",
    "vector",
    "entity",
    "baseentity",
}


@dataclass
class CypherCandidate:
    complexity: str
    cypher: str
    params: dict[str, Any]
    anchor_info: dict[str, Any]
    target_meta: dict[str, Any]
    provenance: dict[str, Any]
    question: str = ""

    def to_item(self) -> dict[str, Any]:
        return {
            "complexity": self.complexity,
            "question": self.question,
            "cypher": self.cypher,
            "params": self.params,
            "anchor_info": self.anchor_info,
            "target_meta": self.target_meta,
            "provenance": self.provenance,
        }


def _safe_label(value: str) -> str:
    return str(value).replace("`", "``")


def _safe_prop(value: str) -> str:
    return str(value).replace("`", "``")


def _safe_rel(value: str) -> str:
    return str(value).replace("`", "``")


def _pick_identity_filter(node: dict[str, Any]) -> tuple[str, Any] | None:
    props = node.get("props") if isinstance(node, dict) else None
    if not isinstance(props, dict):
        return None
    for key in IDENTITY_KEYS:
        value = props.get(key)
        if value not in (None, ""):
            return key, value
    for key, value in props.items():
        if value in (None, ""):
            continue
        if isinstance(value, (str, int, float, bool)):
            return str(key), value
    return None


def _node_label(node: dict[str, Any], fallback: str = "Entity") -> str:
    labels = node.get("labels") if isinstance(node, dict) else None
    if isinstance(labels, list) and labels:
        normalized = [str(label).strip() for label in labels if str(label).strip()]
        semantic = [label for label in normalized if label.lower() not in TECHNICAL_LABELS]
        if semantic:
            return semantic[0]
        return normalized[0]
    return fallback


def _node_name(node: dict[str, Any], fallback: str = "the anchor entity") -> str:
    props = node.get("props") if isinstance(node, dict) else None
    if not isinstance(props, dict):
        return fallback
    for key in IDENTITY_KEYS:
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    return fallback


def _coalesce_expr(var_name: str, target_node: dict[str, Any] | None = None) -> str:
    props = {}
    if isinstance(target_node, dict):
        maybe_props = target_node.get("props")
        if isinstance(maybe_props, dict):
            props = maybe_props

    preferred_order = ("name", "title", "ticker", "id", "uuid", "symbol")
    chosen_keys = [key for key in preferred_order if key in props]

    # If path snapshot has no props, keep a conservative fallback with common keys.
    if not chosen_keys:
        chosen_keys = ["name", "title", "ticker", "id", "symbol"]

    parts = [f"{var_name}.`{_safe_prop(key)}`" for key in chosen_keys]
    # Stable last-resort value without elementId() (blocked by validation policy).
    parts.append(f"toString(id({var_name}))")
    return "coalesce(" + ", ".join(parts) + ")"


def _context_expr(var_name: str) -> str:
    context_keys = (
        "summary",
        "description",
        "content",
        "text",
        "body",
        "headline",
        "snippet",
    )
    parts = [f"{var_name}.`{_safe_prop(key)}`" for key in context_keys]
    parts.append("''")
    return "coalesce(" + ", ".join(parts) + ")"


def _path_signature(path: dict[str, Any]) -> str:
    rels = path.get("relationships") if isinstance(path, dict) else None
    if not isinstance(rels, list):
        return "unknown"
    rel_types = [str((rel or {}).get("type") or "RELATED_TO") for rel in rels]
    return "->".join(rel_types)


def _label_to_natural(label: str) -> str:
    text = str(label or "entity").replace("_", " ").strip()
    if not text:
        return "entity"
    return text[0].lower() + text[1:]


def _relation_hint(rel_type: str) -> str:
    key = str(rel_type or "").strip().upper()
    if key in RELATION_HINTS:
        return RELATION_HINTS[key]
    return f"{key.replace('_', ' ').lower()} links" if key else "business links"


def _relation_story(rels: list[dict[str, Any]], max_hints: int = 3) -> str:
    hints: list[str] = []
    seen: set[str] = set()
    for rel in rels:
        rel_type = str((rel or {}).get("type") or "").strip()
        hint = _relation_hint(rel_type)
        if hint in seen:
            continue
        seen.add(hint)
        hints.append(hint)
        if len(hints) >= max_hints:
            break
    if not hints:
        return "business links"
    if len(hints) == 1:
        return hints[0]
    if len(hints) == 2:
        return f"{hints[0]} and {hints[1]}"
    return f"{', '.join(hints[:-1])}, and {hints[-1]}"


def build_multi_hop_candidate(
    *,
    path: dict[str, Any],
    hop_count: int,
    complexity: str,
    path_index: int = 0,
) -> CypherCandidate | None:
    nodes = path.get("nodes") if isinstance(path, dict) else None
    rels = path.get("relationships") if isinstance(path, dict) else None
    if not isinstance(nodes, list) or not isinstance(rels, list) or not nodes or not rels:
        return None
    if len(nodes) != len(rels) + 1 or len(rels) != hop_count:
        return None

    anchor_filter = _pick_identity_filter(nodes[0])
    if not anchor_filter:
        return None

    pattern_parts = ["(n0)"]
    for idx, rel in enumerate(rels):
        rel_type = str((rel or {}).get("type") or "").strip()
        if not rel_type:
            return None
        node = nodes[idx + 1]
        label = _node_label(node, fallback="")
        label_hint = f":`{_safe_label(label)}`" if label else ""
        pattern_parts.append(f"-[:`{_safe_rel(rel_type)}`]-")
        pattern_parts.append(f"(n{idx + 1}{label_hint})")
    pattern = "".join(pattern_parts)

    where_lines = [f"n0.`{_safe_prop(anchor_filter[0])}` = $anchor_value"]
    target_filter = _pick_identity_filter(nodes[-1])
    if target_filter:
        where_lines.append(f"n{hop_count}.`{_safe_prop(target_filter[0])}` = $target_guard")
    where_block = "\n  AND ".join(where_lines)
    cypher = f"""
MATCH p={pattern}
WHERE {where_block}
RETURN DISTINCT
  {_coalesce_expr(f"n{hop_count}", nodes[-1])} AS target_value,
  {_coalesce_expr(f"n{hop_count}", nodes[-1])} AS target_title,
  {_context_expr(f"n{hop_count}")} AS target_context
LIMIT 10
""".strip()

    anchor_label = _node_label(nodes[0])
    target_label = _node_label(nodes[-1])
    anchor_name = _node_name(nodes[0])
    target_label_natural = _label_to_natural(target_label)
    relation_story = _relation_story(rels)
    question = (
        f"Which {target_label_natural} is most closely linked to {anchor_name} "
        f"when you follow {hop_count} steps across {relation_story}?"
    )
    params: dict[str, Any] = {"anchor_value": anchor_filter[1]}
    if target_filter:
        params["target_guard"] = target_filter[1]
    return CypherCandidate(
        complexity=complexity,
        question=question,
        cypher=cypher,
        params=params,
        anchor_info={
            "label": anchor_label,
            "identity_key": anchor_filter[0],
            "identity_value": anchor_filter[1],
        },
        target_meta={
            "label": target_label,
            "hop_count": hop_count,
            "return_alias": "target_value",
        },
        provenance={
            "source": "path",
            "path_index": path_index,
            "path_signature": _path_signature(path),
        },
    )


def build_simple_candidate_from_path(
    *,
    path: dict[str, Any],
    path_index: int = 0,
) -> CypherCandidate | None:
    nodes = path.get("nodes") if isinstance(path, dict) else None
    rels = path.get("relationships") if isinstance(path, dict) else None
    if not isinstance(nodes, list) or not isinstance(rels, list):
        return None
    if len(nodes) != 2 or len(rels) != 1:
        return None
    rel_type = str((rels[0] or {}).get("type") or "").strip()
    if not rel_type:
        return None

    anchor = nodes[0]
    target = nodes[1]
    anchor_filter = _pick_identity_filter(anchor)
    if not anchor_filter:
        return None
    target_label = _node_label(target)
    anchor_label = _node_label(anchor)
    target_label_natural = _label_to_natural(target_label)
    rel_hint = _relation_hint(rel_type)
    cypher = f"""
MATCH (n0:`{_safe_label(anchor_label)}`)-[:`{_safe_rel(rel_type)}`]-(n1:`{_safe_label(target_label)}`)
WHERE n0.`{_safe_prop(anchor_filter[0])}` = $anchor_value
RETURN DISTINCT
  {_coalesce_expr("n1", target)} AS target_value,
  {_coalesce_expr("n1", target)} AS target_title,
  {_context_expr("n1")} AS target_context
LIMIT 10
""".strip()
    anchor_name = _node_name(anchor)
    question = f"Which {target_label_natural} is directly connected to {anchor_name} via {rel_hint}?"
    return CypherCandidate(
        complexity="simple",
        question=question,
        cypher=cypher,
        params={"anchor_value": anchor_filter[1]},
        anchor_info={
            "label": anchor_label,
            "identity_key": anchor_filter[0],
            "identity_value": anchor_filter[1],
        },
        target_meta={
            "label": target_label,
            "relationship_type": rel_type,
            "return_alias": "target_value",
        },
        provenance={
            "source": "path",
            "path_index": path_index,
            "template_id": "simple_direct_neighbor",
        },
    )


def _find_numeric_prop(node: dict[str, Any]) -> str | None:
    props = node.get("props") if isinstance(node, dict) else None
    if not isinstance(props, dict):
        return None
    for key in NUMERIC_KEYS:
        value = props.get(key)
        if isinstance(value, (int, float)):
            return str(key)
    for key, value in props.items():
        if isinstance(value, (int, float)):
            return str(key)
    return None


def build_aggregation_candidates_from_path(
    *,
    path: dict[str, Any],
    path_index: int = 0,
) -> list[CypherCandidate]:
    nodes = path.get("nodes") if isinstance(path, dict) else None
    rels = path.get("relationships") if isinstance(path, dict) else None
    if not isinstance(nodes, list) or not isinstance(rels, list):
        return []
    if len(nodes) != 2 or len(rels) != 1:
        return []
    rel_type = str((rels[0] or {}).get("type") or "").strip()
    if not rel_type:
        return []

    anchor = nodes[0]
    target = nodes[1]
    anchor_filter = _pick_identity_filter(anchor)
    if not anchor_filter:
        return []
    anchor_name = _node_name(anchor)
    anchor_label = _node_label(anchor)
    target_label = _node_label(target)
    anchor_label_natural = _label_to_natural(anchor_label)
    target_label_natural = _label_to_natural(target_label)
    rel_hint = _relation_hint(rel_type)

    common_anchor = {
        "label": anchor_label,
        "identity_key": anchor_filter[0],
        "identity_value": anchor_filter[1],
    }
    common_target = {
        "label": target_label,
        "relationship_type": rel_type,
        "return_alias": "target_value",
    }
    common_provenance = {
        "source": "path",
        "path_index": path_index,
        "path_signature": _path_signature(path),
    }
    params = {"anchor_value": anchor_filter[1]}

    out: list[CypherCandidate] = []
    count_cypher = f"""
MATCH (n0:`{_safe_label(anchor_label)}`)-[:`{_safe_rel(rel_type)}`]-(n1:`{_safe_label(target_label)}`)
WHERE n0.`{_safe_prop(anchor_filter[0])}` = $anchor_value
RETURN count(DISTINCT n1) AS target_value
LIMIT 1
""".strip()
    out.append(
        CypherCandidate(
            complexity="aggregation",
            question=(
                f"How many distinct {target_label_natural} entities are connected to {anchor_name} via {rel_hint}?"
            ),
            cypher=count_cypher,
            params=dict(params),
            anchor_info=common_anchor,
            target_meta={**common_target, "aggregate_fn": "count_distinct"},
            provenance={**common_provenance, "template_id": "agg_count_distinct_neighbors"},
        )
    )

    numeric_prop = _find_numeric_prop(target)
    if numeric_prop:
        for agg_fn in ("max", "min", "avg"):
            cypher = f"""
MATCH (n0:`{_safe_label(anchor_label)}`)-[:`{_safe_rel(rel_type)}`]-(n1:`{_safe_label(target_label)}`)
WHERE n0.`{_safe_prop(anchor_filter[0])}` = $anchor_value
  AND n1.`{_safe_prop(numeric_prop)}` IS NOT NULL
RETURN {agg_fn}(toFloat(n1.`{_safe_prop(numeric_prop)}`)) AS target_value
LIMIT 1
""".strip()
            out.append(
                CypherCandidate(
                    complexity="aggregation",
                    question=(
                        f"For {anchor_name} ({anchor_label_natural}), what is the {agg_fn.upper()} {numeric_prop} "
                        f"across related {target_label_natural} entities?"
                    ),
                    cypher=cypher,
                    params=dict(params),
                    anchor_info=common_anchor,
                    target_meta={
                        **common_target,
                        "aggregate_fn": agg_fn,
                        "aggregate_prop": numeric_prop,
                    },
                    provenance={
                        **common_provenance,
                        "template_id": f"agg_{agg_fn}_{numeric_prop}",
                    },
                )
            )
    return out

