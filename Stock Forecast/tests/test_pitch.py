import os
import unittest
from unittest.mock import patch

from analysis.pitch import generate_pitch
from core.types import CompanyInfo, CompositeResult, ModuleScore, Rating


class PitchFallbackTests(unittest.TestCase):
    def test_generates_template_pitch_without_api_keys(self):
        composite = CompositeResult(
            overall_score=72.0,
            confidence=0.8,
            rating=Rating.BUY,
            confidence_band=(69.0, 75.0),
            possible_ratings=[Rating.BUY],
            weights_used={"valuation": 1.0},
            component_scores={"valuation": 72.0},
            module_scores=[ModuleScore(name="valuation", score=72.0, confidence=0.8)],
        )
        info = CompanyInfo(
            ticker="TST",
            name="Test Systems",
            sector="Technology",
            industry="Software",
            market_cap=1_000_000_000,
            shares_outstanding=10_000_000,
            country="United States",
            description="Test Systems builds software used in automated financial analysis.",
        )

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "GROQ_API_KEY": ""}):
            result = generate_pitch(
                ticker="TST",
                info=info,
                composite=composite,
                current_price=100.0,
            )

        self.assertEqual(result.model_used, "template (no API key)")
        self.assertIn("## Company Overview", result.pitch_text)
        self.assertIn("## Conclusion", result.pitch_text)
        self.assertEqual(result.ticker, "TST")


if __name__ == "__main__":
    unittest.main()

