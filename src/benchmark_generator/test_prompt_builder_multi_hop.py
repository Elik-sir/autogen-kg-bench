from __future__ import annotations

import unittest

from benchmark_generator.utils.prompt_builder import (
    build_multi_hop_2_prompts,
    build_multi_hop_3_prompts,
    build_multi_hop_4_prompts,
)


class MultiHopPromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "Company": {"type": "node", "properties": {"name": {}, "ticker": {}}},
            "Person": {"type": "node", "properties": {"name": {}, "title": {}}},
            "LED_BY": {"type": "relationship", "properties": {}},
        }
        self.local_context = {
            "anchor_label": "Company",
            "anchor_element_id": "4:company-101",
            "paths_found": 7,
            "local_ontology": (
                "LOCAL ONTOLOGY (only from extracted paths):\n"
                "- Node labels: Company, Event, Person\n"
                "- Relationship types: INVOLVED_IN, LED_BY\n"
                "PATH EXAMPLES:\n"
                "- (Company: Acme Corp) -[INVOLVED_IN]-> (Event: Initiative) -[LED_BY]-> (Person: Jane Doe)\n"
            ),
        }

    def test_multi_hop_2_prompt_has_strict_hop_rules(self):
        _, user_prompt = build_multi_hop_2_prompts(self.schema, self.local_context, 1)
        prompt_l = user_prompt.lower()
        self.assertIn('type "multi-hop-2"', user_prompt)
        self.assertIn("exactly 2 logical steps/hops", prompt_l)
        self.assertIn("Use ONLY the local ontology", user_prompt)
        self.assertIn("Do NOT invent entities", user_prompt)
        self.assertIn('"complexity": "multi-hop-2"', user_prompt)

    def test_multi_hop_3_prompt_has_strict_hop_rules(self):
        _, user_prompt = build_multi_hop_3_prompts(self.schema, self.local_context, 1)
        prompt_l = user_prompt.lower()
        self.assertIn('type "multi-hop-3"', user_prompt)
        self.assertIn("exactly 3 logical steps/hops", prompt_l)
        self.assertIn("cypher must reflect exactly 3 hops", prompt_l)
        self.assertIn('"complexity": "multi-hop-3"', user_prompt)

    def test_multi_hop_4_prompt_has_strict_hop_rules(self):
        _, user_prompt = build_multi_hop_4_prompts(self.schema, self.local_context, 1)
        prompt_l = user_prompt.lower()
        self.assertIn('type "multi-hop-4"', user_prompt)
        self.assertIn("exactly 4 logical steps/hops", prompt_l)
        self.assertIn("output an empty json array []", prompt_l)
        self.assertIn('"complexity": "multi-hop-4"', user_prompt)


if __name__ == "__main__":
    unittest.main()
