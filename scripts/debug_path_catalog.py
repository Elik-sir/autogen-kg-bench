"""
Ontology-first path catalog for benchmark generation.

Algorithm:
1) Discover ontology path templates (labels + relationship-type chains).
2) For each template, run a generic Cypher without label/property filters.
3) Get path arrays from Neo4j.
4) Instantiate readable concrete Cypher queries from sampled paths.
5) Optionally execute instantiated queries and attach answer previews.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neo4j_manager import Neo4jManager


def _safe_ident(label: str) -> str:
    raw = str(label).strip()
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
        return raw
    return f"`{raw.replace('`', '``')}`"


def _node_label(node: dict[str, Any], fallback: str = "entity") -> str:
    labels = node.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return fallback


def _node_name(node: dict[str, Any], fallback: str = "entity") -> str:
    props = node.get("props")
    if isinstance(props, dict):
        for key in ("name", "title", "ticker", "symbol", "id", "uuid"):
            val = props.get(key)
            if val not in (None, ""):
                return str(val)
    return fallback


def _to_cypher_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _pick_identity_prop(node: dict[str, Any]) -> tuple[str, Any] | None:
    props = node.get("props") if isinstance(node, dict) else None
    if not isinstance(props, dict):
        return None
    for key in ("uuid", "id", "ticker", "symbol", "name", "title"):
        value = props.get(key)
        if value in (None, ""):
            continue
        return key, value
    return None


def _build_node_match_fragment(*, var_name: str, node: dict[str, Any], include_identity: bool) -> str:
    label = _node_label(node, fallback="")
    label_hint = f":{_safe_ident(label)}" if label else ""
    if include_identity:
        identity = _pick_identity_prop(node)
        if identity:
            key, value = identity
            return f"({var_name}{label_hint} {{{key}: {_to_cypher_literal(value)}}})"
    return f"({var_name}{label_hint})"


def _extract_ontology_templates(
    db: Neo4jManager, *, min_hops: int, max_hops: int, template_limit: int
) -> list[dict[str, Any]]:
    safe_min = int(max(1, min_hops))
    safe_max = int(max(safe_min, max_hops))
    query = f"""
    MATCH p=(n0)-[*{safe_min}..{safe_max}]-(nk)
    WHERE length(p) >= $min_hops
      AND n0 <> nk
      AND ALL(rel IN relationships(p) WHERE single(x IN relationships(p) WHERE x = rel))
      AND ALL(node IN nodes(p) WHERE single(x IN nodes(p) WHERE x = node))
    WITH
      length(p) AS hops,
      [n IN nodes(p) | coalesce(head(labels(n)), 'Node')] AS node_labels,
      [r IN relationships(p) | type(r)] AS rel_types
    WITH DISTINCT hops, node_labels, rel_types,
      reduce(sig = "", lbl IN node_labels | sig + "(" + lbl + ")")
      + reduce(sig = "", t IN rel_types | sig + "-[" + t + "]-") AS template_signature
    ORDER BY hops ASC, template_signature ASC
    LIMIT $template_limit
    RETURN
      hops,
      node_labels,
      rel_types,
      template_signature
    """
    rows = db.run_query(
        query, {"min_hops": safe_min, "template_limit": int(max(1, template_limit))}
    )
    templates: list[dict[str, Any]] = []
    for row in rows:
        rel_types = [str(x).strip() for x in (row.get("rel_types") or []) if str(x).strip()]
        hops = int(row.get("hops") or 0)
        if hops < safe_min or hops > safe_max or len(rel_types) != hops:
            continue
        templates.append(
            {
                "hops": hops,
                "node_labels": row.get("node_labels") or [],
                "rel_types": rel_types,
                "template_signature": str(row.get("template_signature") or ""),
            }
        )
    return templates


def _build_generic_template_query(rel_types: list[str]) -> str:
    # No labels or property filters here (as requested).
    parts = ["(n0)"]
    for idx, rel_type in enumerate(rel_types, start=1):
        safe_rel = _safe_ident(rel_type)
        parts.append(f"-[:{safe_rel}]-")
        parts.append(f"(n{idx})")
    pattern = "".join(parts)
    return f"""
MATCH p={pattern}
WITH p
LIMIT $sample_limit
RETURN
  [n IN nodes(p) | {{labels: labels(n), props: properties(n)}}] AS nodes,
  [r IN relationships(p) | {{type: type(r)}}] AS relationships
""".strip()


def _sample_paths_for_template(
    db: Neo4jManager, *, rel_types: list[str], sample_limit: int
) -> list[dict[str, Any]]:
    query = _build_generic_template_query(rel_types)
    rows = db.run_query(query, {"sample_limit": int(max(1, sample_limit))})
    out: list[dict[str, Any]] = []
    for row in rows:
        nodes = row.get("nodes") or []
        relationships = row.get("relationships") or []
        if not isinstance(nodes, list) or not isinstance(relationships, list):
            continue
        if len(nodes) != len(relationships) + 1:
            continue
        out.append({"nodes": nodes, "relationships": relationships})
    return out


def _instantiate_cypher_from_sample(sample: dict[str, Any]) -> str | None:
    nodes = sample.get("nodes") or []
    rels = sample.get("relationships") or []
    if not nodes or not rels or len(nodes) != len(rels) + 1:
        return None
    pattern_parts: list[str] = []
    last_idx = len(nodes) - 1
    for idx, node in enumerate(nodes):
        include_identity = idx in (0, last_idx)
        pattern_parts.append(
            _build_node_match_fragment(
                var_name=f"n{idx}",
                node=node,
                include_identity=include_identity,
            )
        )
        if idx < len(rels):
            rel_type = str((rels[idx] or {}).get("type") or "").strip()
            if not rel_type:
                return None
            pattern_parts.append(f"-[:{_safe_ident(rel_type)}]-")
    hop_count = len(rels)
    return f"""
MATCH p={''.join(pattern_parts)}
RETURN DISTINCT coalesce(
  n{hop_count}.name,
  n{hop_count}.title,
  n{hop_count}.ticker,
  n{hop_count}.symbol,
  n{hop_count}.id,
  n{hop_count}.uuid
) AS target_value
LIMIT 10
""".strip()


def _identity_binding(node: dict[str, Any], var_name: str) -> dict[str, Any]:
    label = _node_label(node, fallback="Node")
    identity = _pick_identity_prop(node)
    out: dict[str, Any] = {"var": var_name, "label": label}
    if identity:
        out["identity_key"] = identity[0]
        out["identity_value"] = identity[1]
    return out


def _sample_canonical_signature(sample: dict[str, Any]) -> str:
    nodes = sample.get("nodes") or []
    rels = sample.get("relationships") or []
    if not nodes or len(nodes) != len(rels) + 1:
        return ""

    def node_token(node: dict[str, Any]) -> str:
        label = _node_label(node, fallback="Node")
        identity = _pick_identity_prop(node)
        if identity:
            key, value = identity
            return f"{label}|{key}={value}"
        return label

    rel_types = [str((r or {}).get("type") or "").strip() for r in rels]
    fwd = "->".join([node_token(n) for n in nodes]) + "||" + "->".join(rel_types)
    rev = "->".join([node_token(n) for n in reversed(nodes)]) + "||" + "->".join(reversed(rel_types))
    return min(fwd, rev)


def _preview_answers(db: Neo4jManager, cypher: str, max_answers: int) -> list[str]:
    rows = db.run_query(cypher)
    values: list[str] = []
    for row in rows:
        val = row.get("target_value")
        if val in (None, ""):
            continue
        values.append(str(val))
        if len(values) >= max_answers:
            break
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description="Ontology-first template -> instantiated Cypher catalog.")
    ap.add_argument("--min-hops", type=int, default=1)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--max-templates", type=int, default=250)
    ap.add_argument("--samples-per-template", type=int, default=30)
    ap.add_argument("--answer-preview-limit", type=int, default=5)
    ap.add_argument(
        "--skip-answer-preview",
        action="store_true",
        help="Do not execute instantiated Cypher to fetch answer previews.",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "path_catalog_debug.json")
    args = ap.parse_args()

    if args.min_hops < 1 or args.max_hops < args.min_hops:
        print("Invalid hops range.", file=sys.stderr)
        return 1

    db = Neo4jManager()
    try:
        templates = _extract_ontology_templates(
            db,
            min_hops=int(args.min_hops),
            max_hops=int(args.max_hops),
            template_limit=int(max(1, args.max_templates)),
        )
        out_templates: list[dict[str, Any]] = []
        for idx, template in enumerate(templates, start=1):
            rel_types = template.get("rel_types") or []
            generic_cypher = _build_generic_template_query(rel_types)
            sample_paths = _sample_paths_for_template(
                db,
                rel_types=rel_types,
                sample_limit=int(max(1, args.samples_per_template)),
            )
            if not sample_paths:
                continue
            instantiated_items: list[dict[str, Any]] = []
            seen_signatures: set[str] = set()
            for sample in sample_paths:
                signature = _sample_canonical_signature(sample)
                if not signature or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                instantiated = _instantiate_cypher_from_sample(sample)
                if not instantiated:
                    continue
                nodes = sample.get("nodes") or []
                rels = sample.get("relationships") or []
                entity_chain = [
                    _identity_binding(node, var_name=f"n{i}")
                    for i, node in enumerate(nodes)
                ]
                rel_chain = [str((r or {}).get("type") or "") for r in rels]
                item = {
                    "canonical_signature": signature,
                    "entity_chain": entity_chain,
                    "rel_chain": rel_chain,
                }
                if not args.skip_answer_preview:
                    item["answer_preview"] = _preview_answers(
                        db,
                        instantiated,
                        max_answers=int(max(1, args.answer_preview_limit)),
                    )
                instantiated_items.append(item)

            if not instantiated_items:
                continue

            out_templates.append(
                {
                    "template_id": idx,
                    "hops": int(template.get("hops") or 0),
                    "node_labels": template.get("node_labels") or [],
                    "rel_types": rel_types,
                    "template_signature": template.get("template_signature") or "",
                    "generic_template_cypher": generic_cypher,
                    "reconstruct_hint": "Build MATCH p=(n0)-[:REL1]-...-(nK) and pin n0/nK by entity_chain identity_key/value.",
                    "instantiated_samples": instantiated_items,
                }
            )
            print(f"[{idx}/{len(templates)}] template collected: {len(instantiated_items)} instantiated")

        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "algorithm": "ontology-template -> generic-query -> instantiate -> answer-preview",
                        "min_hops": int(args.min_hops),
                        "max_hops": int(args.max_hops),
                        "max_templates": int(args.max_templates),
                        "samples_per_template": int(args.samples_per_template),
                        "answer_preview_limit": int(args.answer_preview_limit),
                        "templates_discovered": len(templates),
                        "templates_materialized": len(out_templates),
                    },
                    "templates": out_templates,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote path catalog: {out_path}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

