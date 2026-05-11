from __future__ import annotations

import unittest
from typing import Any

from benchmark_generator.question_generation import QuestionGenerationEngine


class _FakeDbManager:
    def run_query(self, query: str, params: dict[str, Any] | None = None):
        if "CALL db.labels()" in query:
            return [{"label": "Company"}]

        if "MATCH (n:`Company`)" in query:
            return [
                {
                    "element_id": "4:company-1",
                    "labels": ["Company"],
                    "props": {"name": "Acme Corp"},
                    "degree": 5,
                    "diversity": 3,
                }
            ]

        if "MATCH p=(anchor)-[*1..1]-(target)" in query and "length(p) = 1" in query:
            return [
                {
                    "nodes": [
                        {
                            "element_id": "4:company-1",
                            "labels": ["Company"],
                            "props": {"name": "Acme Corp"},
                        },
                        {
                            "element_id": "4:product-1",
                            "labels": ["Product"],
                            "props": {"name": "Acme Cloud", "revenue": 10.0},
                        },
                    ],
                    "relationships": [{"type": "PRODUCES"}],
                }
            ]

        if "MATCH p=(anchor)-[*1..3]-(target)" in query and "length(p) = 3" in query:
            return [
                {
                    "nodes": [
                        {
                            "element_id": "4:company-1",
                            "labels": ["Company"],
                            "props": {"name": "Acme Corp"},
                        },
                        {
                            "element_id": "4:news-1",
                            "labels": ["NewsArticle"],
                            "props": {"title": "AI expansion"},
                        },
                        {
                            "element_id": "4:company-2",
                            "labels": ["Company"],
                            "props": {"name": "NVIDIA"},
                        },
                        {
                            "element_id": "4:tech-1",
                            "labels": ["Technology"],
                            "props": {"name": "Generative AI"},
                        },
                    ],
                    "relationships": [
                        {"type": "MENTIONED_IN"},
                        {"type": "MENTIONED_IN"},
                        {"type": "DEVELOPS"},
                    ],
                }
            ]
        return []


class _FakeLlm:
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        return "[]"


class ProgrammaticGenerationSmokeTests(unittest.TestCase):
    def setUp(self):
        self.engine = QuestionGenerationEngine(db=_FakeDbManager(), llm=_FakeLlm())
        self.schema = {"Company": {"type": "node", "properties": {"name": {}}}}

    def test_simple_generation_returns_programmatic_candidate(self):
        items = self.engine.generate_simple_pairs(self.schema, {}, num_questions=1)
        self.assertEqual(1, len(items))
        item = items[0]
        self.assertEqual("simple", item.get("complexity"))
        self.assertIn("MATCH (n0:`Company`)-[:`PRODUCES`]-(n1:`Product`)", item.get("cypher", ""))
        self.assertIn("provenance", item)

    def test_multi_hop_generation_returns_programmatic_candidate(self):
        items = self.engine.generate_multi_hop_x_pairs(self.schema, hop_count=3, num_questions=1)
        self.assertEqual(1, len(items))
        item = items[0]
        self.assertEqual("multi-hop-3", item.get("complexity"))
        self.assertIn("MATCH p=", item.get("cypher", ""))
        self.assertIn("target_meta", item)

    def test_aggregation_generation_returns_programmatic_candidate(self):
        items = self.engine.generate_aggregation_pairs(self.schema, {}, num_questions=1)
        self.assertEqual(1, len(items))
        item = items[0]
        self.assertEqual("aggregation", item.get("complexity"))
        self.assertIn("count(DISTINCT n1)", item.get("cypher", ""))
        self.assertEqual("path", item.get("provenance", {}).get("source"))


if __name__ == "__main__":
    unittest.main()

