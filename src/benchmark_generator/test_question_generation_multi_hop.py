from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmark_generator.question_generation import QuestionGenerationEngine


class QuestionGenerationMultiHopTests(unittest.TestCase):
    def setUp(self):
        self.engine = QuestionGenerationEngine(db=None, llm=None)

    @staticmethod
    def _sample_path_3hop():
        return {
            "nodes": [
                {
                    "element_id": "n0",
                    "labels": ["Company"],
                    "props": {"name": "NVIDIA Corp."},
                    "display_name": "NVIDIA Corp.",
                },
                {
                    "element_id": "n1",
                    "labels": ["InstitutionalInvestor"],
                    "props": {"name": "Vanguard Group"},
                    "display_name": "Vanguard Group",
                },
                {
                    "element_id": "n2",
                    "labels": ["Company"],
                    "props": {"name": "Microsoft Corp."},
                    "display_name": "Microsoft Corp.",
                },
                {
                    "element_id": "n3",
                    "labels": ["NewsArticle"],
                    "props": {"title": "AI infrastructure partnership"},
                    "display_name": "AI infrastructure partnership",
                },
            ],
            "relationships": [
                {"type": "INVESTED_IN"},
                {"type": "INVESTED_IN"},
                {"type": "MENTIONED_IN"},
            ],
        }

    def test_build_question_from_path_is_concise_and_grounded(self):
        question = self.engine._build_question_from_path(
            path=self._sample_path_3hop(),
            hop_count=3,
            complexity="multi-hop-3",
            existing_questions=[],
        )
        self.assertIn("NVIDIA Corp.", question)
        self.assertIn("invested in", question.lower())
        self.assertTrue(question.endswith("?"))
        self.assertNotIn("node", question.lower())
        self.assertLessEqual(len(question), 240)

    def test_quality_gate_rejects_story_like_prompt(self):
        ok = self.engine._passes_multi_hop_question_quality_gate(
            question=(
                "Which company is indirectly linked to NVIDIA Corp. through a series of "
                "signals and shared strategic interest alongside multiple entities?"
            ),
            path=self._sample_path_3hop(),
            hop_count=3,
        )
        self.assertFalse(ok)

    def test_generate_multi_hop_supports_hop3(self):
        anchor = {
            "label": "Company",
            "labels": ["Company"],
            "element_id": "n0",
            "props": {"name": "NVIDIA Corp."},
        }
        self.engine._next_anchor = lambda: anchor  # type: ignore[method-assign]
        local_context = {"paths": [self._sample_path_3hop()]}
        with patch(
            "benchmark_generator.question_generation.build_anchor_subgraph_context",
            return_value=local_context,
        ):
            out = self.engine.generate_multi_hop_x_pairs(
                schema={},
                hop_count=3,
                num_questions=1,
                existing_questions=[],
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].get("complexity"), "multi-hop-3")
        self.assertIn("MATCH p=", out[0].get("cypher", ""))


if __name__ == "__main__":
    unittest.main()
