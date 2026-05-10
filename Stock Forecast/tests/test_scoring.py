import unittest

from core.scoring import CompositeScorer
from core.types import ModuleScore, Rating


class CompositeScorerTests(unittest.TestCase):
    def test_redistributes_missing_module_weights(self):
        scorer = CompositeScorer(
            weights={
                "valuation": 0.50,
                "fundamental": 0.30,
                "risk": 0.20,
            }
        )

        result = scorer.compute(
            [
                ModuleScore(name="valuation", score=80, confidence=0.90),
                ModuleScore(name="risk", score=40, confidence=0.50),
            ]
        )

        self.assertAlmostEqual(result.weights_used["valuation"], 0.50 / 0.70)
        self.assertAlmostEqual(result.weights_used["risk"], 0.20 / 0.70)
        self.assertAlmostEqual(result.overall_score, 68.6)
        self.assertEqual(result.rating, Rating.BUY)

    def test_handles_no_scores(self):
        result = CompositeScorer().compute([])

        self.assertEqual(result.overall_score, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.rating, Rating.STRONG_SELL)
        self.assertEqual(result.module_scores, [])


if __name__ == "__main__":
    unittest.main()

