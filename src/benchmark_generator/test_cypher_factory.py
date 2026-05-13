from __future__ import annotations

import unittest

from benchmark_generator.cypher_factory import (
    build_aggregation_candidates_from_path,
    build_multi_hop_candidate,
    build_simple_candidate_from_path,
)


class CypherFactoryTests(unittest.TestCase):
    def setUp(self):
        self.multi_hop_path = {
            "nodes": [
                {"labels": ["Company"], "props": {"name": "Acme Corp"}},
                {"labels": ["NewsArticle"], "props": {"title": "AI launch"}},
                {"labels": ["Company"], "props": {"name": "NVIDIA"}},
                {"labels": ["Technology"], "props": {"name": "Generative AI"}},
            ],
            "relationships": [
                {"type": "MENTIONED_IN"},
                {"type": "MENTIONED_IN"},
                {"type": "DEVELOPS"},
            ],
        }
        self.simple_path = {
            "nodes": [
                {"labels": ["Company"], "props": {"name": "Acme Corp"}},
                {"labels": ["Product"], "props": {"name": "Acme Cloud", "revenue": 10.5}},
            ],
            "relationships": [{"type": "PRODUCES"}],
        }

    def test_multi_hop_builder_uses_exact_hop_count(self):
        candidate = build_multi_hop_candidate(
            path=self.multi_hop_path,
            hop_count=3,
            complexity="multi-hop-3",
            path_index=4,
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("multi-hop-3", candidate.complexity)
        self.assertIn("MATCH p=(n0)-[:`MENTIONED_IN`]-(n1:`NewsArticle`)", candidate.cypher)
        self.assertIn("-[:`DEVELOPS`]-(n3:`Technology`)", candidate.cypher)
        self.assertIn("n0.`name` = 'Acme Corp'", candidate.cypher)
        self.assertEqual("path", candidate.provenance.get("source"))
        self.assertEqual(4, candidate.provenance.get("path_index"))

    def test_simple_builder_returns_direct_neighbor_query(self):
        candidate = build_simple_candidate_from_path(path=self.simple_path, path_index=1)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual("simple", candidate.complexity)
        self.assertIn("MATCH (n0:`Company`)-[:`PRODUCES`]-(n1:`Product`)", candidate.cypher)
        self.assertIn("RETURN DISTINCT", candidate.cypher)
        self.assertIn("AS target_value", candidate.cypher)
        self.assertEqual({}, candidate.params)

    def test_aggregation_builder_emits_count_and_numeric_aggregates(self):
        candidates = build_aggregation_candidates_from_path(path=self.simple_path, path_index=9)
        self.assertGreaterEqual(len(candidates), 2)
        template_ids = {str(item.provenance.get("template_id")) for item in candidates}
        self.assertIn("agg_count_distinct_neighbors", template_ids)
        self.assertTrue(any("toFloat" in item.cypher for item in candidates))
        self.assertTrue(any("count(DISTINCT n1)" in item.cypher for item in candidates))

    def test_prefers_semantic_label_over_searchable(self):
        path = {
            "nodes": [
                {"labels": ["Location"], "props": {"name": "Norwalk"}},
                {"labels": ["Company"], "props": {"name": "Example Corp"}},
                {"labels": ["Searchable", "Resource"], "props": {"name": "Cloud GPU"}},
            ],
            "relationships": [{"type": "LOCATED_IN"}, {"type": "REQUIRES"}],
        }
        candidate = build_multi_hop_candidate(
            path=path,
            hop_count=2,
            complexity="multi-hop-2",
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertIn("(n2:`Resource`)", candidate.cypher)
        self.assertNotIn("(n2:`Searchable`)", candidate.cypher)
        self.assertIn("resource", candidate.question.lower())
        self.assertNotIn("searchable", candidate.question.lower())


if __name__ == "__main__":
    unittest.main()

