"""
Импорт уже построенного графа LightRAG (NetworkX GraphML) в Neo4j.

Файл `graph_chunk_entity_relation.graphml` содержит узлы и рёбра с атрибутами
(weight, description, keywords, source_id, …). Импорт через админские загрузчики
часто теряет типы/атрибуты рёбер — здесь рёбра пишутся явным Cypher.

Режимы типа связи (см. settings.NEO4J_IMPORT_REL_TYPE_MODE):
  - lightrag: как у Neo4JStorage — все связи с типом DIRECTED и свойством keywords.
  - keywords_primary: тип связи в Neo4j = первое ключевое слово из keywords
    (нормализовано под идентификатор Cypher), плюс те же свойства на рёбрах.

Запуск (из папки light-rag):
  uv run python import_graphml_to_neo4j.py
  uv run python import_graphml_to_neo4j.py --wipe-workspace

По умолчанию узлы получают вторую метку из entity_type (NEO4J_ENTITY_TYPE_AS_LABEL /
--no-entity-type-as-label), совместимо с индексацией через Neo4JEntityTypeLabelStorage.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
from neo4j import GraphDatabase

import settings
from neo4j_entity_type_storage import entity_type_as_neo4j_label


def _escape_label(label: str) -> str:
    s = (label or "").strip()
    if not s:
        s = "base"
    return s.replace("`", "``")


def _neo4j_prop_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if hasattr(v, "item"):  # numpy scalar
        try:
            v = v.item()
        except Exception:  # noqa: BLE001
            return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return int(v)
        return float(v) if isinstance(v, float) else int(v)
    if isinstance(v, str):
        return v
    return str(v)


def _node_row(nid: str, data: dict[str, Any]) -> dict[str, Any]:
    props = {k: _neo4j_prop_value(v) for k, v in data.items() if k != "entity_id"}
    entity_id = str(data.get("entity_id", nid))
    return {"entity_id": entity_id, "props": props}


def _edge_props(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "weight",
        "description",
        "keywords",
        "source_id",
        "file_path",
        "created_at",
        "truncate",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k not in data:
            continue
        v = _neo4j_prop_value(data[k])
        if v is not None and v != "":
            out[k] = v
    return out


_REL_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _primary_rel_type(keywords: str | None) -> str:
    """Первое ключевое слово из поля keywords LightRAG → безопасный тип связи Neo4j."""
    if not keywords or not str(keywords).strip():
        return "DIRECTED"
    first = str(keywords).split(",")[0].strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", first).strip("_")
    if not slug:
        return "DIRECTED"
    slug = slug.upper()
    if slug[0].isdigit():
        slug = "R_" + slug
    if not _REL_TYPE_RE.match(slug):
        return "DIRECTED"
    return slug


def _apply_neo4j_env() -> None:
    if settings.NEO4J_URI:
        import os

        os.environ["NEO4J_URI"] = settings.NEO4J_URI
    if settings.NEO4J_USERNAME:
        import os

        os.environ["NEO4J_USERNAME"] = settings.NEO4J_USERNAME
    if settings.NEO4J_PASSWORD:
        import os

        os.environ["NEO4J_PASSWORD"] = settings.NEO4J_PASSWORD
    if settings.NEO4J_DATABASE:
        import os

        os.environ["NEO4J_DATABASE"] = settings.NEO4J_DATABASE
    if settings.NEO4J_WORKSPACE:
        import os

        os.environ["NEO4J_WORKSPACE"] = settings.NEO4J_WORKSPACE


def _resolve_graphml_path(arg: str | None) -> Path:
    if arg and str(arg).strip():
        p = Path(arg).expanduser()
        return p if p.is_absolute() else (settings.LIGHT_RAG_DIR / p).resolve()
    name = settings.GRAPHML_FILE
    p = Path(name).expanduser()
    if p.is_absolute():
        return p.resolve()
    work = Path(settings.WORKING_DIR).expanduser()
    if not work.is_absolute():
        work = (settings.LIGHT_RAG_DIR / work).resolve()
    return (work / name).resolve()


def import_graphml(
    graphml_path: Path,
    *,
    wipe_workspace: bool,
    rel_type_mode: str,
    batch_size: int,
    entity_type_as_label: bool,
) -> None:
    import os

    mode = rel_type_mode.strip().lower()
    if mode not in ("lightrag", "keywords_primary"):
        mode = "lightrag"

    _apply_neo4j_env()
    uri = os.environ.get("NEO4J_URI") or settings.NEO4J_URI
    user = os.environ.get("NEO4J_USERNAME") or settings.NEO4J_USERNAME
    password = os.environ.get("NEO4J_PASSWORD") or settings.NEO4J_PASSWORD
    database = os.environ.get("NEO4J_DATABASE") or settings.NEO4J_DATABASE
    workspace = (os.environ.get("NEO4J_WORKSPACE") or settings.NEO4J_WORKSPACE or "base").strip() or "base"
    label = _escape_label(workspace)

    if not uri or not user or password is None or password == "":
        raise RuntimeError(
            "Задайте NEO4J_URI, NEO4J_USERNAME и NEO4J_PASSWORD в .env или settings.py"
        )

    if not graphml_path.is_file():
        raise FileNotFoundError(f"GraphML не найден: {graphml_path}")

    g = nx.read_graphml(graphml_path)
    nodes = [_node_row(nid, dict(g.nodes[nid])) for nid in g.nodes()]
    edges_raw: list[tuple[str, str, dict[str, Any], str]] = []
    for u, v, edata in g.edges(data=True):
        d = dict(edata)
        props = _edge_props(d)
        if mode == "keywords_primary":
            rt = _primary_rel_type(d.get("keywords"))
        else:
            rt = "DIRECTED"
        src = str(g.nodes[u].get("entity_id", u))
        tgt = str(g.nodes[v].get("entity_id", v))
        edges_raw.append((src, tgt, props, rt))

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            if wipe_workspace:
                session.run(
                    f"MATCH (n:`{label}`) DETACH DELETE n"
                ).consume()

            if entity_type_as_label:
                by_et: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in nodes:
                    tl = _escape_label(
                        entity_type_as_neo4j_label(row["props"].get("entity_type"))
                    )
                    by_et[tl].append(row)
                for type_l, et_rows in by_et.items():
                    for i in range(0, len(et_rows), batch_size):
                        batch = et_rows[i : i + batch_size]
                        session.run(
                            f"""
                            UNWIND $rows AS row
                            MERGE (n:`{label}`:`{type_l}` {{entity_id: row.entity_id}})
                            SET n += row.props
                            """,
                            rows=batch,
                        ).consume()
            else:
                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i : i + batch_size]
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:`{label}` {{entity_id: row.entity_id}})
                        SET n += row.props
                        """,
                        rows=batch,
                    ).consume()

            if mode == "keywords_primary":
                by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for src, tgt, props, rt in edges_raw:
                    by_type[rt].append({"src": src, "tgt": tgt, "props": props})
                for rt, rows in by_type.items():
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i : i + batch_size]
                        session.run(
                            f"""
                            UNWIND $rows AS row
                            MATCH (a:`{label}` {{entity_id: row.src}})
                            MATCH (b:`{label}` {{entity_id: row.tgt}})
                            MERGE (a)-[r:`{rt}`]-(b)
                            SET r += row.props
                            """,
                            rows=batch,
                        ).consume()
            else:
                rows = [{"src": s, "tgt": t, "props": p} for s, t, p, _ in edges_raw]
                for i in range(0, len(rows), batch_size):
                    batch = rows[i : i + batch_size]
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (a:`{label}` {{entity_id: row.src}})
                        MATCH (b:`{label}` {{entity_id: row.tgt}})
                        MERGE (a)-[r:DIRECTED]-(b)
                        SET r += row.props
                        """,
                        rows=batch,
                    ).consume()

            session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.entity_id)"
            ).consume()
    finally:
        driver.close()

    print(
        f"Импорт завершён: {graphml_path.name} → Neo4j db={database!r} label=`{label}` "
        f"(узлов: {len(nodes)}, рёбер: {len(edges_raw)}, режим типа связи: {mode}, "
        f"entity_type как метка: {entity_type_as_label})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Импорт LightRAG GraphML в Neo4j")
    p.add_argument(
        "--graphml",
        default=None,
        help="Путь к .graphml (по умолчанию WORKING_DIR/GRAPHML_FILE из settings)",
    )
    p.add_argument(
        "--wipe-workspace",
        action="store_true",
        help="Удалить в Neo4j все узлы с меткой workspace перед импортом",
    )
    p.add_argument(
        "--rel-type-mode",
        choices=("lightrag", "keywords_primary"),
        default=settings.NEO4J_IMPORT_REL_TYPE_MODE,
        help="lightrag=DIRECTED+keywords в свойствах; keywords_primary=тип ребра из первого keyword",
    )
    p.add_argument("--batch-size", type=int, default=500)
    p.set_defaults(entity_type_as_label=settings.NEO4J_ENTITY_TYPE_AS_LABEL)
    p.add_argument(
        "--entity-type-as-label",
        dest="entity_type_as_label",
        action="store_true",
        help="Добавить метку Neo4j из entity_type (перекрывает settings)",
    )
    p.add_argument(
        "--no-entity-type-as-label",
        dest="entity_type_as_label",
        action="store_false",
        help="Только метка workspace (перекрывает settings)",
    )
    args = p.parse_args()
    try:
        import_graphml(
            _resolve_graphml_path(args.graphml),
            wipe_workspace=args.wipe_workspace,
            rel_type_mode=args.rel_type_mode,
            batch_size=max(1, int(args.batch_size)),
            entity_type_as_label=bool(args.entity_type_as_label),
        )
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
