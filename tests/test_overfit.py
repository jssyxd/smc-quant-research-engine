"""
Comprehensive test suite for Anti-Overfitting and Backtest Audit Engine.

Verifies:
1. Synthetic strategy with true alpha -> high DSR, passes.
2. Overfitted random walk strategies (N=500 trials) -> DSR drops, high PBO, fails.
3. Analytical formula validation and multiple-testing monotonicity.
4. Interface consistency for functions and classes.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
import numpy as np

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from anti_overfit_suite import (
    DeflatedSharpeRatio,
    ProbabilityOfBacktestOverfitting,
    HaircutSharpeRatio,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    haircut_sharpe,
    expected_maximum_sr,
    AntiOverfitAuditor,
    EULER_MASCHERONI,
)


class TestAntiOverfitSuite(unittest.TestCase):
    """Test suite verifying DSR, PBO, and Haircut Sharpe ratio."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_synthetic_strategy_with_true_alpha_passes(self) -> None:
        """
        1. Synthetic strategy with true alpha -> high DSR, passes.
        """
        T = 1000
        # True alpha: steady positive drift with modest volatility
        # Annualized SR ~ 2.5 - 3.0
        alpha_drift = 0.04 / math.sqrt(252)
        alpha_vol = 0.15 / math.sqrt(252)
        true_alpha_returns = np.random.normal(loc=alpha_drift, scale=alpha_vol, size=T)

        # 9 competing noise variations (N=10 total trials)
        noise_returns = np.random.normal(loc=0.0, scale=alpha_vol, size=(T, 9))
        trials_matrix = np.column_stack([true_alpha_returns, noise_returns])

        # 1. Deflated Sharpe Ratio
        dsr_res = DeflatedSharpeRatio.compute(
            returns=true_alpha_returns,
            trials=np.mean(trials_matrix, axis=0) / np.std(trials_matrix, axis=0, ddof=1),
            significance_level=0.05,
        )

        self.assertGreater(dsr_res.dsr, 0.95, f"DSR should be > 0.95 for true alpha, got {dsr_res.dsr}")
        self.assertLess(dsr_res.p_value, 0.05, f"p-value should be < 0.05, got {dsr_res.p_value}")
        self.assertTrue(dsr_res.passes, "True alpha strategy should pass DSR audit.")
        self.assertTrue(dsr_res.is_significant, "True alpha should be statistically significant.")

        # Class instantiation check
        dsr_obj = DeflatedSharpeRatio(
            returns=true_alpha_returns,
            trials=np.mean(trials_matrix, axis=0) / np.std(trials_matrix, axis=0, ddof=1),
        )
        self.assertAlmostEqual(float(dsr_obj), dsr_res.dsr, places=6)

        # 2. Probability of Backtest Overfitting (PBO) via CSCV
        pbo_res = ProbabilityOfBacktestOverfitting.compute(
            returns_matrix=trials_matrix,
            n_partitions=8,
            pbo_threshold=0.40,
        )

        self.assertLessEqual(pbo_res.pbo, 0.15, f"PBO should be near zero for true alpha, got {pbo_res.pbo}")
        self.assertFalse(pbo_res.is_overfitted, "True alpha should not be classified as overfitted.")
        self.assertTrue(pbo_res.passes, "True alpha should pass PBO audit.")
        self.assertAlmostEqual(float(pbo_res), pbo_res.pbo, places=6)

        # 3. Haircut Sharpe Ratio (Harvey & Liu)
        cand_sr_ann = float(np.mean(true_alpha_returns) / np.std(true_alpha_returns, ddof=1) * math.sqrt(252))
        haircut_res = haircut_sharpe(
            sr=cand_sr_ann,
            n_trials=10,
            t=T,
            method="bonferroni",
        )

        self.assertTrue(haircut_res.passes, "True alpha should pass Haircut Sharpe check.")
        self.assertGreater(haircut_res.haircut_sr, 1.0, "Haircut Sharpe should remain strong and positive.")
        self.assertLess(haircut_res.p_adj, 0.05, "Multiple-testing adjusted p-value should remain < 0.05.")

        # End-to-end Auditor check
        auditor = AntiOverfitAuditor(pbo_threshold=0.40, min_dsr=0.95)
        report = auditor.audit(returns_matrix=trials_matrix, candidate_idx=0)
        self.assertTrue(report.overall_pass, "True alpha should pass end-to-end anti-overfitting audit.")

    def test_overfitted_random_walk_strategies_fails(self) -> None:
        """
        2. Overfitted random walk strategies (N=500 trials) -> DSR drops, high PBO, fails.
        """
        T = 1000
        N = 500

        # Generate N=500 pure random walk strategies (zero alpha, independent Gaussian noise)
        rw_returns = np.random.normal(loc=0.0, scale=0.01, size=(T, N))

        # In-sample Sharpe ratio for all 500 trials
        means = np.mean(rw_returns, axis=0)
        stds = np.std(rw_returns, axis=0, ddof=1)
        trial_srs = means / stds

        # Pick the "best" strategy in-sample (pure data snooping / alpha hallucination)
        best_idx = int(np.argmax(trial_srs))
        best_sr = float(trial_srs[best_idx])
        best_sr_annualized = best_sr * math.sqrt(252)

        # Naive backtesting hallucination: apparent annualized Sharpe looks positive (> 1.0)
        self.assertGreater(best_sr, 0.0, "Best random trial should have positive sample Sharpe.")

        # 1. Deflated Sharpe Ratio
        var_trials = float(np.var(trial_srs, ddof=1))
        dsr_res = DeflatedSharpeRatio.compute(
            returns=rw_returns[:, best_idx],
            trials=trial_srs,
            significance_level=0.05,
        )

        # DSR drops significantly under multiple testing (N=500)
        self.assertLess(dsr_res.dsr, 0.50, f"DSR should drop below 0.50 for overfitted trials, got {dsr_res.dsr}")
        self.assertGreater(dsr_res.p_value, 0.50, f"p-value should be > 0.50 under null of no skill, got {dsr_res.p_value}")
        self.assertFalse(dsr_res.passes, "Overfitted random walk strategy must fail DSR audit.")
        self.assertFalse(dsr_res.is_significant, "Overfitted random walk must not be significant.")

        # Expected maximum Sharpe should exceed or match the observed best Sharpe
        self.assertGreater(
            dsr_res.expected_max_sr,
            best_sr * 0.9,
            "Expected maximum SR under H0 should be comparable to or exceed observed snooped SR.",
        )

        # 2. Probability of Backtest Overfitting (PBO) via CSCV
        pbo_res = ProbabilityOfBacktestOverfitting.compute(
            returns_matrix=rw_returns,
            n_partitions=8,
            pbo_threshold=0.40,
        )

        # Under pure noise, the IS-best strategy is symmetrically distributed OOS -> PBO ~ 0.50
        self.assertGreaterEqual(
            pbo_res.pbo,
            0.40,
            f"PBO should be high (>= 0.40) for 500 noise trials, got {pbo_res.pbo}",
        )
        self.assertTrue(pbo_res.is_overfitted, "Search over 500 noise trials must be flagged as overfitted.")
        self.assertFalse(pbo_res.passes, "Overfitted search must fail PBO audit.")
        # Performance degradation should be substantial
        self.assertGreater(pbo_res.degradation, 0.50, "Performance degradation should exceed 50%.")

        # 3. Haircut Sharpe Ratio (Harvey & Liu)
        haircut_res = haircut_sharpe(
            sr=best_sr_annualized,
            n_trials=N,
            t=T,
            method="bonferroni",
            annualized=True,
        )
        self.assertFalse(haircut_res.passes, "Overfitted random walk must fail Haircut Sharpe test.")
        self.assertGreater(
            haircut_res.haircut_pct,
            0.70,
            f"Haircut penalty should be severe (> 70%), got {haircut_res.haircut_pct:.1%}",
        )

        # End-to-end Auditor check
        auditor = AntiOverfitAuditor(pbo_threshold=0.40, min_dsr=0.95)
        report = auditor.audit(returns_matrix=rw_returns, candidate_idx=best_idx)
        self.assertFalse(report.overall_pass, "Overfitted search must fail overall audit.")

    def test_expected_maximum_sr_analytical_formula(self) -> None:
        """Verify analytical EVT formula: E[max_N] ≈ sqrt(2 ln N) + gamma / sqrt(2 ln N)."""
        N = 100
        expected_std = math.sqrt(2.0 * math.log(N)) + EULER_MASCHERONI / math.sqrt(2.0 * math.log(N))
        calc_std = expected_maximum_sr(n_trials=N, var_trials=None)
        self.assertAlmostEqual(expected_std, calc_std, places=9)

        # N=1 edge case should yield 0.0
        self.assertEqual(expected_maximum_sr(n_trials=1), 0.0)

        # Variance scaling
        var_val = 0.25
        calc_scaled = expected_maximum_sr(n_trials=N, var_trials=var_val)
        self.assertAlmostEqual(calc_scaled, math.sqrt(var_val) * expected_std, places=9)

    def test_dsr_n_trials_monotonicity(self) -> None:
        """Verify that increasing trial count N monotonically increases E[max_N] and lowers DSR."""
        sr = 0.25
        t = 252
        var_trials = 0.005

        dsr_values = []
        emax_values = []
        z_values = []
        for n in [2, 5, 20, 100, 500]:
            res = deflated_sharpe_ratio(
                sr=sr,
                t=t,
                skew=0.0,
                kurtosis=3.0,
                n_trials=n,
                var_trials=var_trials,
            )
            dsr_values.append(res.dsr)
            emax_values.append(res.expected_max_sr)
            z_values.append(res.z_stat)

        # E[max_N] should be strictly increasing with N
        for i in range(len(emax_values) - 1):
            self.assertGreater(emax_values[i + 1], emax_values[i])

        # z-stat and DSR should be strictly decreasing with N
        for i in range(len(z_values) - 1):
            self.assertLess(z_values[i + 1], z_values[i])
            self.assertLess(dsr_values[i + 1], dsr_values[i])
    def test_pbo_cscv_validation_and_partitions(self) -> None:
        """Verify CSCV validates even partitions and handles small/large combinations."""
        # Odd partition count should raise ValueError
        dummy = np.random.normal(size=(100, 5))
        with self.assertRaises(ValueError):
            ProbabilityOfBacktestOverfitting.compute(dummy, n_partitions=7)

        # Minimum trial count validation
        single_col = np.random.normal(size=(100, 1))
        with self.assertRaises(ValueError):
            ProbabilityOfBacktestOverfitting.compute(single_col, n_partitions=4)

        # CSCV with S=6 -> C(6, 3) = 20 combinations
        res = probability_of_backtest_overfitting(dummy, n_partitions=6)
        self.assertEqual(res.n_combinations, 20)
        self.assertEqual(len(res.rank_distribution), 20)
        self.assertEqual(len(res.logits), 20)

    def test_haircut_sharpe_multiple_methods_and_correlation(self) -> None:
        """Verify Bonferroni, BHY, Holm, and correlation discount behaviors."""
        sr = 2.0
        n = 50
        t = 252

        res_bonf = haircut_sharpe(sr=sr, n_trials=n, t=t, method="bonferroni", correlation=0.0)
        res_corr = haircut_sharpe(sr=sr, n_trials=n, t=t, method="bonferroni", correlation=0.8)

        # Higher correlation should reduce effective trials -> lower haircut penalty
        self.assertLess(
            res_corr.haircut_pct,
            res_bonf.haircut_pct,
            "Correlated trials should suffer less haircut than independent trials.",
        )
        self.assertGreater(
            res_corr.haircut_sr,
            res_bonf.haircut_sr,
            "Haircut Sharpe should be higher with correlated trials.",
        )

        # Single trial (N=1) should have zero haircut
        res_single = haircut_sharpe(sr=sr, n_trials=1, t=t)
        self.assertEqual(res_single.haircut_pct, 0.0)
        self.assertEqual(res_single.haircut_sr, sr)

        # Class interface check
        hc_obj = HaircutSharpeRatio(sr=sr, n_trials=n, t=t)
        self.assertAlmostEqual(float(hc_obj), res_bonf.haircut_sr, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
