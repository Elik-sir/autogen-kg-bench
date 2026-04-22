"""
Extract a stress-test subgraph: supernodes (high degree) + bounded paths between them.

Environment (same as Neo4jManager):
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB_NAME (optional, default neo4j)

Cypher design notes
-------------------
1) Supernodes: `size((n)--())` uses the node's relationship count without expanding
   all neighbors into memory. Still scans all nodes; for huge graphs, prefer GDS
   degree on a projected graph (see gds_degree_supernodes).

2) Paths between supernodes: exact "longest simple path" is not practical in Cypher
   (combinatorial). This script uses:
   - APOC: `apoc.algo.allSimplePaths` with maxLevel + limit (preferred when installed).
   - Fallback: shortestPath with a hard relationship cap, plus optional
     variable-length sampling with strict LIMIT.

3) GDS: There is no general "longest path" procedure for arbitrary graphs (NP-hard).
   Use GDS for degree/streaming and for k-shortest or random-walk sampling if you
   install the library; see optional calls below.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterable

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# --- Cypher: identify top-N supernodes by total degree (in + out) ------------
# Ordering by size((n)--()) is cheap per-node compared to expanding all rels.
CYPHER_SUPERnodes = """
MATCH (n)
WITH n, size((n)--()) AS degree
WHERE degree > 0
RETURN
  elementId(n) AS element_id,
  labels(n) AS labels,
  properties(n) AS properties,
  degree
ORDER BY degree DESC
LIMIT $limit
"""

# --- Cypher: APOC bounded paths (expandConfig + terminatorNodes) --------------
# Uses NODE_GLOBAL uniqueness to keep paths simple. Cap maxLevel + limit hard.
# relationshipFilter: e.g. '<>' (any type, both directions), 'KNOWS>', 'KNOWS<|WORKS_WITH>'
# See: https://neo4j.com/labs/apoc/4.4/path-finding/

# --- Cypher: fallback — single shortest path (bounded hops) -------------------
# shortestPath guarantees simple path; upper bound on length avoids runaway.
CYPHER_SHORTEST_PATH = """
MATCH (a), (b)
WHERE elementId(a) = $src_id AND elementId(b) = $dst_id
MATCH p = shortestPath((a)-[*..$max_hops]-(b))
RETURN p AS path
LIMIT 1
"""

# --- Cypher: optional — one bounded variable-length path (not "longest") -----
# Use only with tiny limits; good for adding extra connectivity samples.
CYPHER_SAMPLE_PATH = """
MATCH (a), (b)
WHERE elementId(a) = $src_id AND elementId(b) = $dst_id
MATCH p = (a)-[*$min_hops..$max_hops]-(b)
RETURN p AS path
LIMIT $path_limit
"""

# --- GDS (optional): degree after projecting the whole graph ------------------
# Requires GDS and an appropriate projection. graph_name must be unique per run.
CYPHER_GDS_DROP = "CALL gds.graph.drop($graph_name, false) YIELD graphName"

CYPHER_GDS_PROJECT_ALL = """
CALL gds.graph.project($graph_name, '*', '*')
YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
"""

CYPHER_GDS_DEGREE_STREAM = """
CALL gds.degree.stream($graph_name)
YIELD nodeId, score
RETURN gds.util.asNode(nodeId) AS n, score AS degree
ORDER BY degree DESC
LIMIT $limit
"""


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v if v not in ("", None) else default


@dataclass
class SubgraphCollector:
    """De-duplicate nodes and relationships while streaming paths."""

    node_ids: set[str] = field(default_factory=set)
    rel_ids: set[str] = field(default_factory=set)
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)

    def ingest_path(self, path) -> None:
        for i, node in enumerate(path.nodes):
            eid = node.element_id
            if eid not in self.nodes:
                self.nodes[eid] = {
                    "element_id": eid,
                    "labels": list(node.labels),
                    "properties": dict(node),
                }
            self.node_ids.add(eid)
        for rel in path.relationships:
            rid = rel.element_id
            if rid not in self.edges:
                self.edges[rid] = {
                    "element_id": rid,
                    "type": rel.type,
                    "start_element_id": rel.start_node.element_id,
                    "end_element_id": rel.end_node.element_id,
                    "properties": dict(rel),
                }
            self.rel_ids.add(rid)


def _pairs_from_supernodes(
    supernode_ids: list[str],
    max_pairs: int,
    strategy: str,
    seed: int | None,
) -> list[tuple[str, str]]:
    if len(supernode_ids) < 2:
        return []
    pairs: list[tuple[str, str]] = []
    if strategy == "all":
        for i, a in enumerate(supernode_ids):
            for b in supernode_ids[i + 1 :]:
                pairs.append((a, b))
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs
    rng = random.Random(seed)
    attempts = 0
    max_attempts = max_pairs * 20
    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(supernode_ids, 2)
        if a > b:
            a, b = b, a
        t = (a, b)
        if t not in pairs:
            pairs.append(t)
    return pairs


def _run_paths_apoc(
    session,
    collector: SubgraphCollector,
    src: str,
    dst: str,
    rel_filter: str,
    max_level: int,
    path_limit: int,
) -> int:
    filt = rel_filter if rel_filter.strip() else "<>"
    q = """
    MATCH (a), (b)
    WHERE elementId(a) = $src_id AND elementId(b) = $dst_id
    CALL apoc.path.expandConfig(a, {
      terminatorNodes: [b],
      relationshipFilter: $rel_filter,
      minLevel: 1,
      maxLevel: $max_level,
      limit: $path_limit,
      uniqueness: 'NODE_GLOBAL'
    })
    YIELD path
    RETURN path
    """
    n = 0
    result = session.run(
        q,
        src_id=src,
        dst_id=dst,
        rel_filter=filt,
        max_level=max_level,
        path_limit=path_limit,
    )
    for record in result:
        collector.ingest_path(record["path"])
        n += 1
    return n


def _run_paths_shortest(
    session,
    collector: SubgraphCollector,
    src: str,
    dst: str,
    max_hops: int,
) -> int:
    n = 0
    result = session.run(
        CYPHER_SHORTEST_PATH,
        src_id=src,
        dst_id=dst,
        max_hops=max_hops,
    )
    for record in result:
        collector.ingest_path(record["path"])
        n += 1
    return n


def _batch_fetch_nodes(session, element_ids: Iterable[str], batch_size: int):
    ids = list(element_ids)
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        rows = session.run(
            """
            UNWIND $ids AS eid
            MATCH (n)
            WHERE elementId(n) = eid
            RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties
            """,
            ids=batch,
        )
        for r in rows:
            yield r.data()


def _batch_fetch_rels(session, element_ids: Iterable[str], batch_size: int):
    ids = list(element_ids)
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        rows = session.run(
            """
            UNWIND $ids AS rid
            MATCH ()-[r]-()
            WHERE elementId(r) = rid
            RETURN
            elementId(r) AS element_id,
            type(r) AS type,
            elementId(startNode(r)) AS start_element_id,
            elementId(endNode(r)) AS end_element_id,
            properties(r) AS properties
            """,
            ids=batch,
        )
        for r in rows:
            yield r.data()


def export_json(path: str, nodes: list[dict], edges: list[dict], meta: dict) -> None:
    payload = {"meta": meta, "nodes": nodes, "edges": edges}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def export_graphml(path: str, nodes: list[dict], edges: list[dict]) -> None:
    ns = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}graphml")
    # keys for node label and common props — dynamic keys from first N nodes
    key_id = 0
    prop_keys: dict[str, str] = {}

    def ensure_key(for_type: str, name: str, attr_type: str = "string") -> str:
        nonlocal key_id
        k = f"d{key_id}"
        key_id += 1
        el = ET.SubElement(root, f"{{{ns}}}key")
        el.set("id", k)
        el.set("for", for_type)
        el.set("attr.name", name)
        el.set("attr.type", attr_type)
        return k

    k_labels = ensure_key("node", "labels")
    graph = ET.SubElement(root, f"{{{ns}}}graph")
    graph.set("edgedefault", "directed")

    for n in nodes:
        nid = n["element_id"]
        node_el = ET.SubElement(graph, f"{{{ns}}}node")
        node_el.set("id", nid)
        d = ET.SubElement(node_el, f"{{{ns}}}data")
        d.set("key", k_labels)
        d.text = ",".join(n.get("labels") or [])
        for pk, pv in (n.get("properties") or {}).items():
            if pk not in prop_keys:
                prop_keys[pk] = ensure_key("node", pk)
            d2 = ET.SubElement(node_el, f"{{{ns}}}data")
            d2.set("key", prop_keys[pk])
            d2.text = "" if pv is None else str(pv)

    rel_key_type = ensure_key("edge", "type")
    edge_prop_keys: dict[str, str] = {}
    for e in edges:
        eid = e["element_id"]
        edge_el = ET.SubElement(graph, f"{{{ns}}}edge")
        edge_el.set("id", eid)
        edge_el.set("source", e["start_element_id"])
        edge_el.set("target", e["end_element_id"])
        t = ET.SubElement(edge_el, f"{{{ns}}}data")
        t.set("key", rel_key_type)
        t.text = e.get("type") or ""
        for pk, pv in (e.get("properties") or {}).items():
            if pk not in edge_prop_keys:
                edge_prop_keys[pk] = ensure_key("edge", pk)
            d2 = ET.SubElement(edge_el, f"{{{ns}}}data")
            d2.set("key", edge_prop_keys[pk])
            d2.text = "" if pv is None else str(pv)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def fetch_supernodes(
    session,
    limit: int,
    use_gds: bool,
    gds_graph_name: str | None,
) -> list[dict[str, Any]]:
    if use_gds and gds_graph_name:
        try:
            session.run(CYPHER_GDS_DROP, graph_name=gds_graph_name)
        except Exception:
            pass
        session.run(CYPHER_GDS_PROJECT_ALL, graph_name=gds_graph_name)
        rows = session.run(CYPHER_GDS_DEGREE_STREAM, graph_name=gds_graph_name, limit=limit)
        out = []
        for r in rows:
            n = r["n"]
            out.append(
                {
                    "element_id": n.element_id,
                    "labels": list(n.labels),
                    "properties": dict(n),
                    "degree": int(r["degree"]),
                }
            )
        return out
    rows = session.run(CYPHER_SUPERnodes, limit=limit)
    return [r.data() for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract supernode-heavy subgraph for benchmarking.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of supernodes to take.")
    parser.add_argument("--max-pairs", type=int, default=50, help="Cap path queries (pair count).")
    parser.add_argument(
        "--pair-strategy",
        choices=("sample", "all"),
        default="sample",
        help="all = first max-pairs in lexicographic order; sample = random pairs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for pair sampling.")
    parser.add_argument(
        "--path-mode",
        choices=("apoc", "shortest"),
        default="apoc",
        help="apoc uses allSimplePaths with caps; shortest uses shortestPath fallback.",
    )
    parser.add_argument("--rel-filter", type=str, default="", help="APOC relationship filter, e.g. 'KNOWS>' or ''")
    parser.add_argument("--max-level", type=int, default=5, help="APOC allSimplePaths max depth.")
    parser.add_argument("--path-limit", type=int, default=25, help="Max paths per pair.")
    parser.add_argument("--max-hops", type=int, default=15, help="shortestPath upper bound (*..max_hops).")
    parser.add_argument("--use-gds", action="store_true", help="Use GDS degree.stream (requires GDS).")
    parser.add_argument(
        "--gds-graph-name",
        type=str,
        default=None,
        help="Project name for GDS (default: auto uuid).",
    )
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for node/rel fetch.")
    parser.add_argument("--out-json", type=str, default="benchmark_subgraph.json")
    parser.add_argument("--out-graphml", type=str, default="", help="Optional GraphML path.")
    args = parser.parse_args()

    uri = _env("NEO4J_URI")
    user = _env("NEO4J_USER")
    password = _env("NEO4J_PASSWORD", "") or ""
    database = _env("NEO4J_DB_NAME", "neo4j") or "neo4j"
    if not uri or not user:
        print("NEO4J_URI and NEO4J_USER must be set.", file=sys.stderr)
        return 1

    gds_name = args.gds_graph_name
    if args.use_gds and not gds_name:
        gds_name = f"bench_{uuid.uuid4().hex[:12]}"

    driver = GraphDatabase.driver(uri, auth=(user, password))
    collector = SubgraphCollector()
    meta: dict[str, Any] = {
        "top_n": args.top_n,
        "max_pairs": args.max_pairs,
        "pair_strategy": args.pair_strategy,
        "path_mode": args.path_mode,
        "rel_filter": args.rel_filter,
        "max_level": args.max_level,
        "path_limit": args.path_limit,
        "max_hops": args.max_hops,
        "use_gds": args.use_gds,
    }

    try:
        with driver.session(database=database) as session:
            supernodes = fetch_supernodes(session, args.top_n, args.use_gds, gds_name if args.use_gds else None)
            if not supernodes:
                print("No supernodes found (empty graph or zero-degree nodes only).", file=sys.stderr)
                return 2

            for s in supernodes:
                eid = s["element_id"]
                if eid not in collector.nodes:
                    collector.nodes[eid] = {
                        "element_id": eid,
                        "labels": s.get("labels") or [],
                        "properties": dict(s.get("properties") or {}),
                    }
                    collector.node_ids.add(eid)

            ids = [s["element_id"] for s in supernodes]
            pairs = _pairs_from_supernodes(ids, args.max_pairs, args.pair_strategy, args.seed)
            paths_found = 0
            for src, dst in pairs:
                if src == dst:
                    continue
                try:
                    if args.path_mode == "apoc":
                        paths_found += _run_paths_apoc(
                            session,
                            collector,
                            src,
                            dst,
                            args.rel_filter,
                            args.max_level,
                            args.path_limit,
                        )
                    else:
                        paths_found += _run_paths_shortest(
                            session, collector, src, dst, args.max_hops
                        )
                except Exception as e:
                    print(f"[warn] pair {src}->{dst}: {e}", file=sys.stderr)

            # Enrich / ensure full properties via batched fetch
            nodes_full: dict[str, dict[str, Any]] = {}
            for row in _batch_fetch_nodes(session, collector.node_ids, args.batch_size):
                nodes_full[row["element_id"]] = {
                    "element_id": row["element_id"],
                    "labels": row["labels"],
                    "properties": row["properties"],
                }
            edges_full: dict[str, dict[str, Any]] = {}
            for row in _batch_fetch_rels(session, collector.rel_ids, args.batch_size):
                edges_full[row["element_id"]] = {
                    "element_id": row["element_id"],
                    "type": row["type"],
                    "start_element_id": row["start_element_id"],
                    "end_element_id": row["end_element_id"],
                    "properties": row["properties"],
                }

            meta.update(
                {
                    "supernode_count": len(supernodes),
                    "pairs_queried": len(pairs),
                    "paths_materialized": paths_found,
                    "unique_nodes": len(collector.node_ids),
                    "unique_rels": len(collector.rel_ids),
                }
            )

        if args.use_gds and gds_name:
            try:
                with driver.session(database=database) as s2:
                    s2.run(CYPHER_GDS_DROP, graph_name=gds_name)
            except Exception:
                pass
    finally:
        driver.close()

    node_list = list(nodes_full.values()) if nodes_full else list(collector.nodes.values())
    edge_list = list(edges_full.values()) if edges_full else list(collector.edges.values())

    export_json(args.out_json, node_list, edge_list, meta)
    print(f"Wrote JSON: {args.out_json} ({len(node_list)} nodes, {len(edge_list)} edges)")
    if args.out_graphml:
        export_graphml(args.out_graphml, node_list, edge_list)
        print(f"Wrote GraphML: {args.out_graphml}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
