"""
Neo4j graph storage: вторая метка узла из свойства entity_type (Person, Organization, …).

Совместимо с запросами LightRAG: узлы по-прежнему имеют метку workspace (NEO4J_WORKSPACE),
плюс метка из entity_type. Свойство entity_type на узле сохраняется.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from neo4j import AsyncManagedTransaction
from neo4j import exceptions as neo4jExceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.utils import logger

_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def entity_type_as_neo4j_label(raw: str | None) -> str:
    """Преобразует entity_type LightRAG в безопасную метку Neo4j (PascalCase)."""
    if raw is None or not str(raw).strip():
        return "Entity"
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw).strip()).strip("_")
    if not slug:
        return "Entity"
    parts = [p for p in slug.split("_") if p]
    if not parts:
        return "Entity"
    label = "".join(
        (p[0].upper() + p[1:].lower()) if len(p) > 1 else p.upper() for p in parts
    )
    if label and label[0].isdigit():
        label = "T_" + label
    if not _LABEL_RE.match(label):
        return "Entity"
    return label


@dataclass
class Neo4JEntityTypeLabelStorage(Neo4JStorage):
    """Как Neo4JStorage, но MERGE (n:Workspace:EntityKind {entity_id: ...})."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        workspace_label = self._get_workspace_label()
        type_label = entity_type_as_neo4j_label(node_data.get("entity_type"))
        properties = node_data
        if "entity_id" not in properties:
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")

        try:
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    query = f"""
                    MERGE (n:`{workspace_label}`:`{type_label}` {{entity_id: $entity_id}})
                    SET n += $properties
                    """
                    result = await tx.run(
                        query, entity_id=node_id, properties=properties
                    )
                    await result.consume()

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def upsert_nodes_batch(
        self, nodes: list[tuple[str, dict[str, str]]]
    ) -> None:
        if not nodes:
            return
        workspace_label = self._get_workspace_label()
        by_type: dict[str, list[dict[str, str | dict[str, str]]]] = defaultdict(list)
        for node_id, node_data in nodes:
            if "entity_id" not in node_data:
                raise ValueError(
                    "Neo4j: node properties must contain an 'entity_id' field"
                )
            tl = entity_type_as_neo4j_label(node_data.get("entity_type"))
            by_type[tl].append({"entity_id": node_id, "props": node_data})

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                for type_label, nodes_data in by_type.items():

                    async def execute_batch(
                        tx: AsyncManagedTransaction,
                        *,
                        tl: str = type_label,
                        nd: list = nodes_data,
                    ):
                        query = f"""
                        UNWIND $nodes AS row
                        MERGE (n:`{workspace_label}`:`{tl}` {{entity_id: row.entity_id}})
                        SET n += row.props
                        """
                        result = await tx.run(query, nodes=nd)
                        await result.consume()

                    await session.execute_write(execute_batch)
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error during batch node upsert: {str(e)}"
            )
            raise
