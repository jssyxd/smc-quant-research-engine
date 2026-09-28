"""
Unit tests for Credal Uncertainty Engine (NeurIPS 2025 Credal Transformer framework).

Demonstrates:
1. High conviction bullish setup -> low uncertainty, strong trade signal.
2. Conflicted setup (Bullish OB + Bearish HTF) -> high uncertainty/conflict, abstains.
3. No evidence setup (flat market) -> high vacuity/uncertainty, abstains.
4. Mathematical invariants of Subjective Logic & Dirichlet belief distributions.
5. High conviction bearish setup.
6. Weak/isolated factor setup -> vacuity triggers abstention.
7. Indicator alias robustness.
"""

import math
import os
import sys
import unittest

# Ensure project root and src/ are discoverable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.credal_engine import (
    CredalDecision,
    CredalSMCEngine,
    CredalSignal,
    DEFAULT_BULL_FACTORS,
    DEFAULT_BEAR_FACTORS,
)


class TestCredalEngine(unittest.TestCase):
    """Test suite validating CredalSMCEngine against task requirements and math invariants."""

    def setUp(self) -> None:
        self.engine = CredalSMCEngine(u_max=0.35, delta=0.15)

    def test_high_conviction_bullish_setup(self) -> None:
        """1. High conviction bullish setup -> low uncertainty, strong trade signal."""
        indicators = {
            "htf_bull_bias": True,
            "bull_ob": True,
            "ssl_swept": True,
            "in_discount": True,
            "htf_bear_bias": False,
            "bear_ob": False,
            "bsl_swept": False,
            "in_premium": False,
        }
        decision = self.engine.evaluate(indicators)

        # 1. Epistemic uncertainty must be strictly below u_max (0.35)
        self.assertLess(decision.uncertainty, self.engine.u_max,
                        f"Expected low uncertainty < {self.engine.u_max}, got {decision.uncertainty}")

        # 2. Model must NOT abstain
        self.assertFalse(decision.should_abstain, "Model should not abstain on high conviction setup")
        self.assertIsNone(decision.abstain_reason)

        # 3. Output signal must be BULLISH with strong conviction
        self.assertEqual(decision.credal_signal, CredalSignal.BULLISH)
        self.assertEqual(decision.credal_signal, "BULLISH")
        self.assertTrue(decision.is_bullish)
        self.assertFalse(decision.is_bearish)
        self.assertGreater(decision.confidence, 0.5,
                           f"Expected high confidence > 0.5, got {decision.confidence}")

        # 4. Dirichlet probability p_bull must dominate
        self.assertGreater(decision.p_bull, decision.p_bear)
        self.assertGreater(decision.p_bull, 0.70)
        self.assertGreater(decision.b_bull, 0.60)
        self.assertEqual(decision.b_bear, 0.0)

    def test_conflicted_setup_abstains(self) -> None:
        """2. Conflicted setup (Bullish OB + Bearish HTF) -> high uncertainty, abstains."""
        indicators = {
            "bull_ob": True,        # Bullish signal (OB)
            "htf_bear_bias": True,  # Opposing macro trend (Bearish EMA/HTF)
            "ssl_swept": False,
            "bsl_swept": False,
            "in_discount": False,
            "in_premium": False,
        }
        decision = self.engine.evaluate(indicators)

        # 1. Model MUST abstain to prevent trade hallucination
        self.assertTrue(decision.should_abstain, "Conflicted setup MUST trigger abstention")
        self.assertEqual(decision.credal_signal, CredalSignal.NEUTRAL)
        self.assertEqual(decision.confidence, 0.0, "Abstained signal must have 0.0 confidence")
        self.assertTrue(decision.is_neutral)

        # 2. Conflicting evidence exists between bull and bear
        self.assertGreater(decision.e_bull, 0.0)
        self.assertGreater(decision.e_bear, 0.0)
        belief_diff = abs(decision.b_bull - decision.b_bear)
        self.assertLess(belief_diff, self.engine.delta,
                        f"Expected conflict diff < {self.engine.delta}, got {belief_diff}")

        # 3. Epistemic uncertainty exceeds u_max
        self.assertGreater(decision.uncertainty, self.engine.u_max)
        self.assertIn("CONFLICT", str(decision.abstain_reason))

    def test_no_evidence_flat_market_abstains(self) -> None:
        """3. No evidence setup (flat market) -> high vacuity/uncertainty, abstains."""
        # All indicators False / 0 / empty
        indicators = {}
        decision = self.engine.evaluate(indicators)

        # 1. Vacuity / Epistemic uncertainty must be at maximum (1.0 = total ignorance)
        self.assertAlmostEqual(decision.uncertainty, 1.0, places=6,
                               msg=f"Expected vacuity u = 1.0 on zero evidence, got {decision.uncertainty}")

        # 2. Model MUST abstain
        self.assertTrue(decision.should_abstain, "Zero evidence must trigger abstention")
        self.assertEqual(decision.credal_signal, CredalSignal.NEUTRAL)
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.abstain_reason, "HIGH_UNCERTAINTY")

        # 3. Expected probabilities reflect uninformative Dirichlet prior (uniform 1/3)
        self.assertAlmostEqual(decision.p_bull, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(decision.p_bear, 1.0 / 3.0, places=4)
        self.assertAlmostEqual(decision.p_neutral, 1.0 / 3.0, places=4)

        # 4. Beliefs must be exactly 0
        self.assertEqual(decision.b_bull, 0.0)
        self.assertEqual(decision.b_bear, 0.0)
        self.assertEqual(decision.b_neutral, 0.0)

    def test_high_conviction_bearish_setup(self) -> None:
        """High conviction bearish setup -> low uncertainty, strong trade signal."""
        indicators = {
            "htf_bear_bias": True,
            "bear_ob": True,
            "bsl_swept": True,
            "in_premium": True,
        }
        decision = self.engine.evaluate(indicators)

        self.assertLess(decision.uncertainty, self.engine.u_max)
        self.assertFalse(decision.should_abstain)
        self.assertEqual(decision.credal_signal, CredalSignal.BEARISH)
        self.assertTrue(decision.is_bearish)
        self.assertGreater(decision.confidence, 0.5)
        self.assertGreater(decision.p_bear, 0.70)
        self.assertEqual(decision.b_bull, 0.0)

    def test_isolated_weak_factor_abstains(self) -> None:
        """Single isolated factor (e.g. Bull OB alone) lacks confluence -> vacuity triggers abstention."""
        # Even without conflict, a single factor should NOT fool the system into a high-risk trade
        indicators = {"bull_ob": True}
        decision = self.engine.evaluate(indicators)

        self.assertTrue(decision.should_abstain, "Isolated factor lacks confluence, must abstain")
        self.assertGreater(decision.uncertainty, self.engine.u_max)
        self.assertEqual(decision.credal_signal, CredalSignal.NEUTRAL)
        self.assertEqual(decision.confidence, 0.0)

    def test_subjective_logic_invariants(self) -> None:
        """Verify mathematical invariants of Dirichlet Evidential / Subjective Logic theory."""
        test_cases = [
            (0.0, 0.0, 0.0),
            (6.5, 0.0, 0.0),
            (0.0, 6.5, 0.0),
            (2.0, 2.0, 0.0),
            (4.0, 1.5, 0.5),
            (10.0, 2.0, 1.0),
        ]

        for e_bull, e_bear, e_neutral in test_cases:
            dec = self.engine.compute_from_evidence(e_bull, e_bear, e_neutral)

            # Invariant 1: sum(b_k) + u == 1.0
            sum_b_and_u = dec.b_bull + dec.b_bear + dec.b_neutral + dec.uncertainty
            self.assertAlmostEqual(sum_b_and_u, 1.0, places=6,
                                   msg=f"Failed sum(b) + u == 1: got {sum_b_and_u}")

            # Invariant 2: sum(p_k) == 1.0
            sum_p = dec.p_bull + dec.p_bear + dec.p_neutral
            self.assertAlmostEqual(sum_p, 1.0, places=6,
                                   msg=f"Failed sum(p) == 1: got {sum_p}")

            # Invariant 3: p_k == b_k + u / K
            expected_p_bull = dec.b_bull + dec.uncertainty / 3.0
            expected_p_bear = dec.b_bear + dec.uncertainty / 3.0
            expected_p_neutral = dec.b_neutral + dec.uncertainty / 3.0
            self.assertAlmostEqual(dec.p_bull, expected_p_bull, places=6)
            self.assertAlmostEqual(dec.p_bear, expected_p_bear, places=6)
            self.assertAlmostEqual(dec.p_neutral, expected_p_neutral, places=6)

            # Invariant 4: uncertainty in (0, 1]
            self.assertGreater(dec.uncertainty, 0.0)
            self.assertLessEqual(dec.uncertainty, 1.0)

    def test_indicator_aliases(self) -> None:
        """Verify indicator alias resolution across varied naming conventions."""
        # PascalCase from smc_optimized codebase
        pascal_indicators = {
            "HTFBullBias": True,
            "BullOBSignal": True,
            "SSLSwept": True,
            "InDiscount": True,
        }
        dec_pascal = self.engine.evaluate(pascal_indicators)

        # snake_case canonical
        snake_indicators = {
            "htf_bull_bias": True,
            "bull_ob": True,
            "ssl_swept": True,
            "in_discount": True,
        }
        dec_snake = self.engine.evaluate(snake_indicators)

        self.assertAlmostEqual(dec_pascal.e_bull, dec_snake.e_bull, places=6)
        self.assertAlmostEqual(dec_pascal.uncertainty, dec_snake.uncertainty, places=6)
        self.assertEqual(dec_pascal.credal_signal, dec_snake.credal_signal)

    def test_decision_container_ergonomics(self) -> None:
        """Verify CredalDecision provides attribute, dict, and indexing access."""
        decision = self.engine.evaluate(htf_bull_bias=True, bull_ob=True, ssl_swept=True, in_discount=True)

        # Attribute access
        self.assertIsInstance(decision.p_bull, float)
        self.assertIsInstance(decision.p_bear, float)
        self.assertIsInstance(decision.uncertainty, float)
        self.assertIsInstance(decision.should_abstain, bool)
        self.assertIsInstance(decision.credal_signal, CredalSignal)

        # Dict subscript access
        self.assertEqual(decision["p_bull"], decision.p_bull)
        self.assertEqual(decision["uncertainty"], decision.uncertainty)
        self.assertEqual(decision["credal_signal"], decision.credal_signal)

        # to_dict conversion
        as_dict = decision.to_dict()
        self.assertIn("p_bull", as_dict)
        self.assertIn("p_bear", as_dict)
        self.assertIn("uncertainty", as_dict)
        self.assertIn("should_abstain", as_dict)
        self.assertIn("credal_signal", as_dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
