from __future__ import annotations

import unittest

from benchmark_generator.validation import _is_balanced_multi_hop_question


class ValidationMultiHopQualityTests(unittest.TestCase):
    def test_accepts_balanced_question(self):
        ok, reason = _is_balanced_multi_hop_question(
            "Which companies are connected to NVIDIA Corp. through invested in, then mentioned in?"
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_rejects_graph_jargon(self):
        ok, reason = _is_balanced_multi_hop_question(
            "Which node is reachable from NVIDIA Corp. in this graph path?"
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "graph_jargon")


if __name__ == "__main__":
    unittest.main()
