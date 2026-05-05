"""
Экспорт графа из Neo4j в формат LightRAG `insert_custom_kg`.

Что делает:
1) Читает узлы/рёбра из workspace-метки (NEO4J_WORKSPACE).
2) Строит dict: {"entities": [...], "relationships": [...], "chunks": [...]}.
3) По умолчанию пишет JSON и сразу вставляет его в LightRAG.

Запуск (из папки `src/benchmarks/light-rag`):
  uv run python neo4j_to_custom_kg.py
  uv run python neo4j_to_custom_kg.py --no-insert
  uv run python neo4j_to_custom_kg.py --workspace my_ws --out-json my_kg.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

import settings


def _escape_label(label: str) -> str:
    s = (label or "").strip()
    if not s:
        s = "base"
    return s.replace("`", "``")


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _as_float(v: Any, default: float = 1.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _resolve_paths(working_dir: str, out_json: str) -> tuple[Path, Path]:
    work = Path(working_dir).expanduser()
    if not work.is_absolute():
        work = (settings.LIGHT_RAG_DIR / work).resolve()

    out = Path(out_json).expanduser()
    if not out.is_absolute():
        out = (settings.LIGHT_RAG_DIR / out).resolve()
    return work, out


def _resolve_neo4j_conn(
    *,
    uri: str | None,
    username: str | None,
    password: str | None,
    database: str | None,
) -> tuple[str, str, str, str]:
    resolved_uri = _safe_str(uri) or os.getenv("NEO4J_URI") or settings.NEO4J_URI
    resolved_user = (
        _safe_str(username)
        or os.getenv("NEO4J_USERNAME")
        or os.getenv("NEO4J_USER")
        or settings.NEO4J_USERNAME
    )
    resolved_pass = (
        _safe_str(password) or os.getenv("NEO4J_PASSWORD") or settings.NEO4J_PASSWORD
    )
    resolved_db = (
        _safe_str(database) or os.getenv("NEO4J_DATABASE") or settings.NEO4J_DATABASE
    )
    if not resolved_uri or not resolved_user or not resolved_pass:
        raise RuntimeError(
            "Нужны параметры подключения к Neo4j: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
        )
    return resolved_uri, resolved_user, resolved_pass, _safe_str(resolved_db, "neo4j")


def _build_custom_kg(
    *,
    uri: str,
    username: str,
    password: str,
    database: str,
    workspace: str,
    max_chunks: int,
) -> dict[str, list[dict[str, Any]]]:
    ws = _safe_str(workspace, settings.NEO4J_WORKSPACE or "lightrag")
    ws_label = _escape_label(ws)

    driver = GraphDatabase.driver(uri, auth=(username, password))
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    try:
        with driver.session(database=database) as session:
            node_rows = session.run(
                f"""
                MATCH (n:`{ws_label}`)
                RETURN elementId(n) AS element_id, properties(n) AS props, labels(n) AS labels
                """
            ).data()

            id_map: dict[str, str] = {}
            for row in node_rows:
                element_id = _safe_str(row.get("element_id"))
                props = row.get("props") or {}
                labels = row.get("labels") or []
                entity_id = _safe_str(
                    props.get("entity_id") or props.get("id") or props.get("name"),
                    f"node:{element_id}",
                )
                id_map[element_id] = entity_id
                type_labels = [l for l in labels if _safe_str(l) != ws]
                entity_type = _safe_str(
                    props.get("entity_type"),
                    type_labels[0] if type_labels else "Entity",
                )
                entities.append(
                    {
                        "entity_name": entity_id,
                        "entity_type": entity_type,
                        "description": _safe_str(
                            props.get("description")
                            or props.get("summary")
                            or props.get("content")
                        ),
                        "source_id": _safe_str(
                            props.get("source_id"),
                            f"neo4j:{element_id}",
                        ),
                    }
                )

                if max_chunks != 0 and len(chunks) >= max_chunks:
                    continue
                chunk_text = _safe_str(
                    props.get("content")
                    or props.get("text")
                    or props.get("chunk_content")
                )
                is_chunk_type = _safe_str(props.get("entity_type")).lower() == "chunk"
                if chunk_text and (is_chunk_type or "chunk" in [x.lower() for x in labels]):
                    chunks.append(
                        {
                            "content": chunk_text,
                            "source_id": _safe_str(
                                props.get("source_id"),
                                f"neo4j:{element_id}",
                            ),
                            "source_chunk_index": int(
                                props.get("source_chunk_index")
                                or props.get("chunk_order_index")
                                or 0
                            ),
                        }
                    )

            rel_rows = session.run(
                f"""
                MATCH (a:`{ws_label}`)-[r]->(b:`{ws_label}`)
                RETURN
                    elementId(a) AS src_el,
                    elementId(b) AS tgt_el,
                    properties(a) AS src_props,
                    properties(b) AS tgt_props,
                    type(r) AS rel_type,
                    properties(r) AS rel_props
                """
            ).data()

            for row in rel_rows:
                src_el = _safe_str(row.get("src_el"))
                tgt_el = _safe_str(row.get("tgt_el"))
                src_props = row.get("src_props") or {}
                tgt_props = row.get("tgt_props") or {}
                rel_type = _safe_str(row.get("rel_type"), "RELATED_TO")
                rel_props = row.get("rel_props") or {}

                src_id = id_map.get(src_el) or _safe_str(
                    src_props.get("entity_id"),
                    f"node:{src_el}",
                )
                tgt_id = id_map.get(tgt_el) or _safe_str(
                    tgt_props.get("entity_id"),
                    f"node:{tgt_el}",
                )

                relationships.append(
                    {
                        "src_id": src_id,
                        "tgt_id": tgt_id,
                        "description": _safe_str(
                            rel_props.get("description"),
                            f"{src_id} {rel_type} {tgt_id}",
                        ),
                        "keywords": _safe_str(
                            rel_props.get("keywords"),
                            rel_type.lower(),
                        ),
                        "weight": _as_float(rel_props.get("weight"), 1.0),
                        "source_id": _safe_str(
                            rel_props.get("source_id")
                            or src_props.get("source_id")
                            or tgt_props.get("source_id"),
                            f"neo4j-rel:{src_el}->{tgt_el}",
                        ),
                    }
                )
    finally:
        driver.close()

    return {
        "entities": entities,
        "relationships": relationships,
        "chunks": chunks,
    }


async def _insert_into_lightrag(custom_kg: dict[str, Any], working_dir: Path) -> None:
    from openrouter_lightrag import (
        apply_openrouter_env_defaults,
        build_rag,
        ensure_lightrag_available,
    )

    ensure_lightrag_available()
    apply_openrouter_env_defaults()

    rag = build_rag(working_dir)
    await rag.initialize_storages()
    try:
        if hasattr(rag, "ainsert_custom_kg"):
            await rag.ainsert_custom_kg(custom_kg)
        elif hasattr(rag, "insert_custom_kg"):
            rag.insert_custom_kg(custom_kg)
        else:
            raise RuntimeError("В установленной версии LightRAG нет insert_custom_kg.")
    finally:
        await rag.finalize_storages()


async def _run(args: argparse.Namespace) -> int:
    uri, username, password, database = _resolve_neo4j_conn(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
    )
    working_dir, out_json = _resolve_paths(args.working_dir, args.out_json)
    workspace = _safe_str(args.workspace, settings.NEO4J_WORKSPACE or "lightrag")

    custom_kg = _build_custom_kg(
        uri=uri,
        username=username,
        password=password,
        database=database,
        workspace=workspace,
        max_chunks=int(args.max_chunks),
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(custom_kg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Custom KG сохранён: {out_json} "
        f"(entities={len(custom_kg['entities'])}, "
        f"relationships={len(custom_kg['relationships'])}, "
        f"chunks={len(custom_kg['chunks'])})"
    )

    if not args.no_insert:
        working_dir.mkdir(parents=True, exist_ok=True)
        await _insert_into_lightrag(custom_kg, working_dir)
        print(f"Custom KG вставлен в LightRAG (working_dir={working_dir}).")
    else:
        print("Вставка в LightRAG пропущена (--no-insert).")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Экспорт графа из Neo4j в LightRAG custom_kg и (опционально) insert_custom_kg."
    )
    p.add_argument("--uri", default=None, help="NEO4J_URI (по умолчанию env/settings)")
    p.add_argument(
        "--username",
        default=None,
        help="NEO4J_USERNAME (по умолчанию env/settings)",
    )
    p.add_argument(
        "--password",
        default=None,
        help="NEO4J_PASSWORD (по умолчанию env/settings)",
    )
    p.add_argument(
        "--database",
        default=settings.NEO4J_DATABASE,
        help="База Neo4j (по умолчанию из settings)",
    )
    p.add_argument(
        "--workspace",
        default=settings.NEO4J_WORKSPACE or "lightrag",
        help="Метка workspace в Neo4j, из которой читать граф",
    )
    p.add_argument(
        "--working-dir",
        default=settings.WORKING_DIR,
        help="WORKING_DIR LightRAG для insert_custom_kg",
    )
    p.add_argument(
        "--out-json",
        default="neo4j_custom_kg.json",
        help="Куда сохранить собранный custom_kg JSON",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=2000,
        help="Лимит chunks (0 = без лимита), чтобы не раздувать payload",
    )
    p.add_argument(
        "--no-insert",
        action="store_true",
        help="Только экспорт в JSON, без вставки в LightRAG",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
