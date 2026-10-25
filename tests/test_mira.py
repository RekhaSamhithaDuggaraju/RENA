import unittest
from core.llm.mira import score_memory_items, WEIGHTS


class MIRATests(unittest.TestCase):
    def row(self, item_id, **overrides):
        values = {"id": item_id, "semantic_relevance": .8, "historical_usefulness": .7,
                  "query_context_usefulness": .9, "redundancy": .1, "confidence": .95}
        values.update(overrides)
        return values

    def test_weighted_score_and_rank_are_deterministic(self):
        result = score_memory_items([self.row("low", semantic_relevance=.1), self.row("high")])
        self.assertEqual(result["items"][0]["id"], "high")
        self.assertEqual([x["rank"] for x in result["items"]], [1, 2])
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_redundancy_reduces_importance(self):
        a = score_memory_items([self.row("a", redundancy=0)])["items"][0]["score"]
        b = score_memory_items([self.row("b", redundancy=1)]) ["items"][0]["score"]
        self.assertAlmostEqual(a - b, WEIGHTS["non_redundancy"])

    def test_invalid_signal_rejected(self):
        with self.assertRaises(ValueError):
            score_memory_items([self.row("bad", confidence=1.2)])

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            score_memory_items([self.row("x"), self.row("x")])

    def test_empty_items_rejected(self):
        with self.assertRaises(ValueError):
            score_memory_items([])


if __name__ == "__main__":
    unittest.main()
