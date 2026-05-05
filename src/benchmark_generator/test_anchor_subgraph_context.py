from __future__ import annotations

import unittest
from typing import Any

from benchmark_generator.utils.anchor_subgraph_context import (
    build_anchor_subgraph_context,
    build_balanced_anchor_order,
    get_stratified_anchor_pool,
)


class _FakeDbManager:
    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_query(self, query: str, params: dict[str, Any] | None = None):
        safe_params = params or {}
        self.calls.append((query, safe_params))

        if "CALL db.labels()" in query:
            return [{"label": "Company"}, {"label": "News"}]

        if "MATCH (n:`Company`)" in query:
            return [
                {
                    "node_id": 101,
                    "element_id": "4:company-101",
                    "labels": ["Company"],
                    "props": {"name": "Acme Corp", "ticker": "ACM"},
                    "degree": 12,
                    "diversity": 4,
                },
                {
                    "node_id": 102,
                    "element_id": "4:company-102",
                    "labels": ["Company"],
                    "props": {"name": "Globex", "ticker": "GBX"},
                    "degree": 10,
                    "diversity": 3,
                },
            ]

        if "MATCH (n:`News`)" in query:
            return [
                {
                    "node_id": 201,
                    "element_id": "4:news-201",
                    "labels": ["News"],
                    "props": {"title": "Acme expands operations"},
                    "degree": 8,
                    "diversity": 2,
                }
            ]

        if "MATCH p=(anchor)-[*1.." in query:
            hop_count = 0
            if "length(p) = 2" in query:
                hop_count = 2
            if hop_count == 2:
                return [
                    {
                        "nodes": [
                            {
                                "element_id": "4:company-101",
                                "node_id": 101,
                                "labels": ["Company"],
                                "props": {"name": "Acme Corp", "ticker": "ACM"},
                            },
                            {
                                "element_id": "4:event-301",
                                "node_id": 301,
                                "labels": ["Event"],
                                "props": {"title": "Supply Chain Initiative"},
                            },
                            {
                                "element_id": "4:person-401",
                                "node_id": 401,
                                "labels": ["Person"],
                                "props": {"name": "Jane Doe", "title": "COO"},
                            },
                        ],
                        "relationships": [{"type": "INVOLVED_IN"}, {"type": "LED_BY"}],
                    }
                ]
            return []

        return []


class AnchorSubgraphContextTests(unittest.TestCase):
    def test_stratified_anchor_pool_collects_each_label(self):
        db = _FakeDbManager()
        pool = get_stratified_anchor_pool(db, limit_per_label=2)

        self.assertIn("Company", pool)
        self.assertIn("News", pool)
        self.assertEqual(2, len(pool["Company"]))
        self.assertEqual(1, len(pool["News"]))

    def test_build_balanced_anchor_order_interleaves_labels(self):
        pool = {
            "Company": [
                {"label": "Company", "element_id": "c1"},
                {"label": "Company", "element_id": "c2"},
            ],
            "News": [{"label": "News", "element_id": "n1"}],
        }
        order = build_balanced_anchor_order(pool)

        labels = [item.get("label") for item in order]
        self.assertEqual(3, len(order))
        self.assertGreaterEqual(labels.count("Company"), 2)
        self.assertIn("News", labels[:2], "News anchor should appear in first round")

    def test_anchor_subgraph_context_contains_local_ontology_sections(self):
        db = _FakeDbManager()
        anchor = {
            "label": "Company",
            "labels": ["Company"],
            "node_id": 101,
            "element_id": "4:company-101",
            "props": {"name": "Acme Corp", "ticker": "ACM"},
        }

        ctx = build_anchor_subgraph_context(
            db,
            anchor=anchor,
            hop_count=2,
            max_paths_per_anchor=5,
        )

        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual("Company", ctx.get("anchor_label"))
        self.assertEqual(2, ctx.get("hop_count"))
        self.assertEqual(1, ctx.get("paths_found"))
        ontology = str(ctx.get("local_ontology", ""))
        self.assertIn("LOCAL ONTOLOGY", ontology)
        self.assertIn("UNIQUE ENTITIES", ontology)
        self.assertIn("PATH EXAMPLES", ontology)
        self.assertIn("Acme Corp", ontology)
        self.assertIn("INVOLVED_IN", ontology)


if __name__ == "__main__":
    unittest.main()
