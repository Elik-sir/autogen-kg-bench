"""
Export Cypher queries from path catalog as a plain JSON array of strings.

Input: path_catalog_debug.json produced by scripts/debug_path_catalog.py (ontology-first version).
Output: a JSON file containing only ["MATCH ...", "MATCH ...", ...]

Example:
  uv run python scripts/export_cypher_strings.py --in path_catalog_debug.json --out cypher_catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _safe_ident(name: str) -> str:
    raw = str(name).strip()
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
        return raw
    return f"`{raw.replace('`', '``')}`"


def _to_cypher_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def _node_fragment(binding: dict[str, Any], *, include_identity: bool) -> str:
    var_name = str(binding.get("var") or "n").strip() or "n"
    label = str(binding.get("label") or "").strip()
    label_hint = f":{_safe_ident(label)}" if label else ""
    if include_identity:
        key = str(binding.get("identity_key") or "").strip()
        if key:
            value = binding.get("identity_value")
            return f"({var_name}{label_hint} {{{_safe_ident(key)}: {_to_cypher_literal(value)}}})"
    return f"({var_name}{label_hint})"


def _should_bind_node_identity(*, node_idx: int, target_idx: int, bind_intermediate: bool) -> bool:
    """
    Binding policy for node identities:
    - n0 is always treated as anchor and should be bound (if identity exists in input).
    - target node (nK by default) must stay unbound to be returned as unknown.
    - intermediate nodes are bound only in strict mode (bind_intermediate=True).
    """
    if node_idx == target_idx:
        return False
    if node_idx == 0:
        return True
    return bind_intermediate


def _build_cypher(
    entity_chain: list[dict[str, Any]],
    rel_chain: list[str],
    *,
    bind_intermediate: bool = True,
) -> str | None:
    if not entity_chain or not rel_chain:
        return None
    if len(entity_chain) != len(rel_chain) + 1:
        return None

    # Multi-hop benchmark target should always be the final node in path.
    last_idx = len(entity_chain) - 1
    target_idx = last_idx
    parts: list[str] = []
    for i, binding in enumerate(entity_chain):
        include_identity = _should_bind_node_identity(
            node_idx=i,
            target_idx=target_idx,
            bind_intermediate=bind_intermediate,
        )
        parts.append(_node_fragment(binding, include_identity=include_identity))
        if i < len(rel_chain):
            rel_type = str(rel_chain[i] or "").strip()
            if not rel_type:
                return None
            parts.append(f"-[:{_safe_ident(rel_type)}]-")

    pattern = "".join(parts)
    return (
        "MATCH p="
        + pattern
        + "\nRETURN DISTINCT coalesce(\n"
        + f"  n{target_idx}.name,\n"
        + f"  n{target_idx}.title,\n"
        + f"  n{target_idx}.ticker,\n"
        + f"  n{target_idx}.symbol,\n"
        + f"  n{target_idx}.id,\n"
        + f"  n{target_idx}.uuid\n"
        + ") AS target_value\nLIMIT 10"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Cypher queries as JSON array of strings.")
    ap.add_argument("--in", dest="in_path", type=Path, default=ROOT / "path_catalog_debug.json")
    ap.add_argument("--out", dest="out_path", type=Path, default=ROOT / "cypher_catalog.json")
    ap.add_argument("--max", dest="max_queries", type=int, default=0, help="0 = no limit")
    ap.add_argument(
        "--include-generic",
        action="store_true",
        help="Also include template-level generic_template_cypher queries.",
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate queries while preserving first occurrence order.",
    )
    ap.add_argument(
        "--bind-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bind intermediate nodes by identity (strict mode). "
            "Use --no-bind-intermediate to keep only anchor node bound."
        ),
    )
    args = ap.parse_args()

    in_path: Path = args.in_path if args.in_path.is_absolute() else (ROOT / args.in_path)
    out_path: Path = args.out_path if args.out_path.is_absolute() else (ROOT / args.out_path)

    if not in_path.is_file():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    templates = payload.get("templates") if isinstance(payload, dict) else None
    if not isinstance(templates, list):
        print("Invalid input: expected {templates: [...]} JSON.", file=sys.stderr)
        return 2

    out: list[str] = []
    seen: set[str] = set()
    for tpl in templates:
        if not isinstance(tpl, dict):
            continue
        if args.include_generic:
            generic_q = str(tpl.get("generic_template_cypher") or "").strip()
            if generic_q:
                if not args.dedupe or generic_q not in seen:
                    out.append(generic_q)
                    if args.dedupe:
                        seen.add(generic_q)
                if args.max_queries and len(out) >= args.max_queries:
                    break
        samples = tpl.get("instantiated_samples")
        if not isinstance(samples, list):
            continue
        for s in samples:
            if not isinstance(s, dict):
                continue
            entity_chain = s.get("entity_chain")
            rel_chain = s.get("rel_chain")
            if not isinstance(entity_chain, list) or not isinstance(rel_chain, list):
                continue
            q = _build_cypher(
                entity_chain,
                rel_chain,
                bind_intermediate=bool(args.bind_intermediate),
            )
            if not q:
                continue
            if args.dedupe and q in seen:
                continue
            out.append(q)
            if args.dedupe:
                seen.add(q)
            if args.max_queries and len(out) >= args.max_queries:
                break
        if args.max_queries and len(out) >= args.max_queries:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} cypher queries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

