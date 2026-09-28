"""
Comprehensive unit tests for Qlib-inspired Micro-Structure SMC Alpha Pipeline.

Verifies:
1. Strict Causality:
   - Causal Swing High/Low detection: swing at p = t - swing_len confirmed at bar t.
   - Prefix Invariance: truncating dataset at bar T yields identical features and signals.
2. Feature Non-Degeneracy:
   - orderflow_imbalance strictly bounded in [0.0, 1.0], handles zero-range candles safely.
   - vol_spike strictly positive and finite.
   - ob_level equals 50% equilibrium (Mean Threshold) of validated Order Block candle.
   - fvg_level equals exact midpoint of Fair Value Gap.
   - atr_ratio = ATR(14) / ATR(100) and volatility regime filtering [0.55, 1.80].
   - Multi-scale EMA trend alignment (EMA20 > EMA50).
3. Signal Consistency & Maker Execution:
   - score_long, score_short strictly bounded in [0, 100].
   - signal in {-1, 0, 1}.
   - limit_entry_px in discount zone (<= Close) for LONG, premium zone (>= Close) for SHORT.
   - Dirichlet Credal uncertainty abstention (u <= 0.35, non-conflicting).
4. Interface & Robustness:
   - Case-insensitive column matching (e.g. Open vs open).
   - Functional API vs Class Pipeline equivalence.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is discoverable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.qlib_smc_alpha import (
    QlibSMCConfig,
    QlibSMCAlphaPipeline,
    compute_wilder_atr,
    compute_causal_swings,
    compute_orderflow_imbalance,
    compute_vol_spike,
    compute_atr_ratio,
    compute_ob_levels,
    compute_fvg_levels,
    compute_ema_trend,
    extract_qlib_smc_features,
    generate_qlib_smc_signals,
    compute_qlib_smc_alpha,
)


class TestQlibSMCAlpha(unittest.TestCase):
    """Test suite for Qlib SMC Alpha Feature Extraction and Signal Generation."""

    def setUp(self) -> None:
        np.random.seed(42)

    def test_causal_swing_high_low_timing(self) -> None:
        """1A. Verify causal swing high/low confirmation at p = t - swing_len strictly at bar t."""
        swing_len = 5
        n = 30
        high = np.full(n, 100.0)
        low = np.full(n, 90.0)
        close = np.full(n, 95.0)
        open_ = np.full(n, 95.0)
        volume = np.full(n, 1000.0)

        # Place a clear swing high peak at bar p = 10
        high[10] = 115.0
        # Place a clear swing low trough at bar p = 18
        low[18] = 80.0

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        swings = compute_causal_swings(df, swing_len=swing_len)

        # 1. At bar p = 10, the swing high MUST NOT be confirmed yet
        self.assertFalse(swings["is_swing_high_confirmed"].iloc[10])

        # 2. Between bars 10 and 14, confirmation MUST be False
        for t in range(10, 10 + swing_len):
            self.assertFalse(
                swings["is_swing_high_confirmed"].iloc[t],
                f"Swing high at p=10 was prematurely confirmed at t={t}",
            )

        # 3. At exactly bar t = 10 + swing_len = 15, swing high MUST be confirmed
        self.assertTrue(
            swings["is_swing_high_confirmed"].iloc[15],
            "Swing high at p=10 must be confirmed at t = p + swing_len = 15",
        )
        self.assertEqual(swings["swing_high"].iloc[15], 115.0)

        # 4. Similarly for swing low at p = 18, confirmation must occur at t = 18 + 5 = 23
        for t in range(18, 18 + swing_len):
            self.assertFalse(
                swings["is_swing_low_confirmed"].iloc[t],
                f"Swing low at p=18 was prematurely confirmed at t={t}",
            )
        self.assertTrue(
            swings["is_swing_low_confirmed"].iloc[23],
            "Swing low at p=18 must be confirmed at t = 23",
        )
        self.assertEqual(swings["swing_low"].iloc[23], 80.0)

        # 5. Future invariance: modifying bars > 15 must not alter confirmation at t=15
        df_mod = df.copy()
        df_mod.loc[16:, "high"] = 120.0
        swings_mod = compute_causal_swings(df_mod, swing_len=swing_len)
        self.assertTrue(swings_mod["is_swing_high_confirmed"].iloc[15])
        self.assertEqual(swings_mod["swing_high"].iloc[15], 115.0)

    def test_prefix_invariance_strict_causality(self) -> None:
        """1B. Prefix Invariance: truncating at bar T yields identical features and signals."""
        n = 200
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.4)
        high = close + np.random.uniform(0.1, 1.2, n)
        low = close - np.random.uniform(0.1, 1.2, n)
        open_ = low + np.random.uniform(0.2, 0.8, n) * (high - low)
        volume = np.random.uniform(200, 2000, n)

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        pipeline = QlibSMCAlphaPipeline(QlibSMCConfig(swing_len=5))
        df_full = pipeline.run(df)

        numeric_cols = [
            "orderflow_imbalance",
            "vol_spike",
            "atr14",
            "atr100",
            "atr_ratio",
            "ema20",
            "ema50",
            "score_long",
            "score_short",
            "credal_uncertainty",
        ]
        bool_cols = [
            "is_swing_high_confirmed",
            "is_swing_low_confirmed",
            "volatility_regime_ok",
            "trend_up",
            "trend_down",
            "bull_ob",
            "bear_ob",
            "bull_fvg",
            "bear_fvg",
            "should_abstain",
        ]

        # Test across multiple truncation cutoffs
        for T in [50, 75, 100, 140, 180, 200]:
            df_sub = pipeline.run(df.iloc[:T])
            row_sub = df_sub.iloc[-1]
            row_full = df_full.iloc[T - 1]

            # Numerical precision check
            for col in numeric_cols:
                val_sub = row_sub[col]
                val_full = row_full[col]
                self.assertAlmostEqual(
                    float(val_sub),
                    float(val_full),
                    places=5,
                    msg=f"Causality leak detected in numeric column '{col}' at cutoff T={T}",
                )

            # Boolean check
            for col in bool_cols:
                self.assertEqual(
                    bool(row_sub[col]),
                    bool(row_full[col]),
                    msg=f"Causality leak detected in boolean column '{col}' at cutoff T={T}",
                )

            # Signal check
            self.assertEqual(
                int(row_sub["signal"]),
                int(row_full["signal"]),
                msg=f"Causality leak detected in 'signal' at cutoff T={T}",
            )

    def test_orderflow_imbalance_properties(self) -> None:
        """2A. Verify orderflow_imbalance formula and bounds."""
        df = pd.DataFrame({
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [12.0, 12.0, 12.0, 10.0],
            "low": [8.0, 8.0, 8.0, 10.0],
            "close": [12.0, 8.0, 10.0, 10.0],  # Close at High, Close at Low, Close at Mid, Flat bar
            "volume": [100.0, 100.0, 100.0, 100.0],
        })
        ofi = compute_orderflow_imbalance(df)

        # Close == High -> 1.0
        self.assertAlmostEqual(ofi.iloc[0], 1.0, places=5)
        # Close == Low -> 0.0
        self.assertAlmostEqual(ofi.iloc[1], 0.0, places=5)
        # Close == Mid -> 0.5
        self.assertAlmostEqual(ofi.iloc[2], 0.5, places=5)
        # Flat candle (High == Low) -> handled safely without ZeroDivisionError
        self.assertTrue(0.0 <= ofi.iloc[3] <= 1.0)

    def test_vol_spike_properties(self) -> None:
        """2B. Verify volume spike factor calculation."""
        vols = np.full(50, 100.0)
        vols[30] = 300.0  # 3x volume spike

        df = pd.DataFrame({
            "open": np.full(50, 10.0),
            "high": np.full(50, 11.0),
            "low": np.full(50, 9.0),
            "close": np.full(50, 10.0),
            "volume": vols,
        })
        v_spike = compute_vol_spike(df, period=20)

        # Baseline volume spike should be ~1.0
        self.assertAlmostEqual(v_spike.iloc[20], 1.0, places=4)
        # Spike bar should be significantly elevated
        self.assertGreater(v_spike.iloc[30], 2.0)
        # No NaNs or infs
        self.assertTrue(np.all(np.isfinite(v_spike)))
        self.assertTrue(np.all(v_spike > 0))

    def test_ob_level_50pct_equilibrium(self) -> None:
        """2C. Verify validated Order Block detection and 50% equilibrium level."""
        # Candle 0: Down candle (Open 102, High 103, Low 99, Close 100) -> 50% eq = (103 + 99) / 2 = 101.0
        # Candle 1: Bullish displacement (Open 100, High 108, Low 99.5, Close 107)
        df = pd.DataFrame({
            "open": [102.0, 100.0],
            "high": [103.0, 108.0],
            "low": [99.0, 99.5],
            "close": [100.0, 107.0],
            "volume": [100.0, 500.0],
        })
        ob_res = compute_ob_levels(df, vol_spike_threshold=1.2)

        self.assertTrue(ob_res["bull_ob"].iloc[1])
        self.assertFalse(ob_res["bear_ob"].iloc[1])
        # 50% equilibrium of candle 0 must be (103.0 + 99.0) / 2 = 101.0
        self.assertAlmostEqual(ob_res["ob_level"].iloc[1], 101.0, places=5)
        self.assertAlmostEqual(ob_res["bull_ob_level"].iloc[1], 101.0, places=5)

        # Bearish OB:
        # Candle 0: Up candle (Open 100, High 104, Low 99, Close 103) -> 50% eq = 101.5
        # Candle 1: Bearish displacement (Open 103, High 103.5, Low 95, Close 96)
        df_bear = pd.DataFrame({
            "open": [100.0, 103.0],
            "high": [104.0, 103.5],
            "low": [99.0, 95.0],
            "close": [103.0, 96.0],
            "volume": [100.0, 500.0],
        })
        ob_bear = compute_ob_levels(df_bear, vol_spike_threshold=1.2)
        self.assertTrue(ob_bear["bear_ob"].iloc[1])
        self.assertAlmostEqual(ob_bear["ob_level"].iloc[1], 101.5, places=5)

    def test_fvg_level_midpoint(self) -> None:
        """2D. Verify Fair Value Gap detection and midpoint computation."""
        # 3 candles for Bullish FVG:
        # Candle 0: High = 100.0
        # Candle 1: Big surge
        # Candle 2: Low = 104.0 (Low[2] > High[0], gap = [100, 104], midpoint = 102.0)
        df = pd.DataFrame({
            "open": [98.0, 101.0, 105.0],
            "high": [100.0, 107.0, 108.0],
            "low": [97.0, 100.5, 104.0],
            "close": [99.0, 106.0, 107.0],
            "volume": [100.0, 300.0, 200.0],
        })
        fvg = compute_fvg_levels(df, min_atr_mult=0.05)

        self.assertTrue(fvg["bull_fvg"].iloc[2])
        self.assertFalse(fvg["bear_fvg"].iloc[2])
        # Midpoint of (100.0 + 104.0) / 2.0 = 102.0
        self.assertAlmostEqual(fvg["fvg_level"].iloc[2], 102.0, places=5)

        # Bearish FVG:
        # Candle 0: Low = 105.0
        # Candle 2: High = 101.0 (High[2] < Low[0], gap = [101, 105], midpoint = 103.0)
        df_bear = pd.DataFrame({
            "open": [107.0, 103.0, 99.0],
            "high": [108.0, 104.0, 101.0],
            "low": [105.0, 98.0, 97.0],
            "close": [106.0, 99.0, 98.0],
            "volume": [100.0, 300.0, 200.0],
        })
        fvg_bear = compute_fvg_levels(df_bear, min_atr_mult=0.05)
        self.assertTrue(fvg_bear["bear_fvg"].iloc[2])
        self.assertAlmostEqual(fvg_bear["fvg_level"].iloc[2], 103.0, places=5)

    def test_atr_ratio_and_volatility_regime(self) -> None:
        """2E. Verify ATR ratio and volatility regime filtering [0.55, 1.80]."""
        n = 150
        # Steady baseline
        high = np.linspace(101, 130, n)
        low = high - 2.0
        close = high - 1.0
        open_ = low + 1.0
        volume = np.full(n, 100.0)

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        atr14, atr100, ratio = compute_atr_ratio(df, fast_period=14, slow_period=100)

        self.assertTrue(np.all(atr14 > 0))
        self.assertTrue(np.all(atr100 > 0))
        self.assertTrue(np.all(np.isfinite(ratio)))

        # In steady state, ratio is around 1.0
        self.assertTrue(0.55 <= ratio.iloc[-1] <= 1.80)

        # Inject sudden extreme blowout at the end
        df_blowout = df.copy()
        df_blowout.loc[140:149, "high"] = df_blowout.loc[140:149, "high"] + 20.0
        _, _, ratio_blowout = compute_atr_ratio(df_blowout, fast_period=14, slow_period=100)
        self.assertGreater(ratio_blowout.iloc[-1], 1.80)

    def test_multi_scale_ema_trend_alignment(self) -> None:
        """2F. Verify multi-scale EMA trend alignment (EMA20 > EMA50)."""
        n = 100
        uptrend = np.linspace(50, 150, n)
        df_up = pd.DataFrame({
            "open": uptrend - 0.5,
            "high": uptrend + 1.0,
            "low": uptrend - 1.0,
            "close": uptrend,
            "volume": 100.0,
        })
        trend_up_df = compute_ema_trend(df_up, fast_period=20, slow_period=50)
        self.assertTrue(trend_up_df["trend_up"].iloc[-1])
        self.assertFalse(trend_up_df["trend_down"].iloc[-1])

        downtrend = np.linspace(150, 50, n)
        df_down = pd.DataFrame({
            "open": downtrend + 0.5,
            "high": downtrend + 1.0,
            "low": downtrend - 1.0,
            "close": downtrend,
            "volume": 100.0,
        })
        trend_down_df = compute_ema_trend(df_down, fast_period=20, slow_period=50)
        self.assertTrue(trend_down_df["trend_down"].iloc[-1])
        self.assertFalse(trend_down_df["trend_up"].iloc[-1])

    def test_score_and_signal_bounds(self) -> None:
        """3A. Verify scores in [0, 100] and signals in {-1, 0, 1}."""
        n = 300
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.random.uniform(0.1, 1.5, n)
        low = close - np.random.uniform(0.1, 1.5, n)
        open_ = low + np.random.uniform(0.1, 0.9, n) * (high - low)
        volume = np.random.uniform(100, 1000, n)

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        pipeline = QlibSMCAlphaPipeline()
        res = pipeline.run(df)

        self.assertTrue(np.all(res["score_long"] >= 0.0))
        self.assertTrue(np.all(res["score_long"] <= 100.0))
        self.assertTrue(np.all(res["score_short"] >= 0.0))
        self.assertTrue(np.all(res["score_short"] <= 100.0))
        self.assertTrue(set(res["signal"].unique()).issubset({-1, 0, 1}))

    def test_maker_limit_entry_px_discount_premium(self) -> None:
        """3B. Verify Maker limit order prices: discount (<= Close) for LONG, premium (>= Close) for SHORT."""
        # Construct synthetic data designed to trigger trade signals
        n = 120
        drift = np.linspace(100, 130, n)
        high = drift + 0.8
        low = drift - 0.8
        close = drift + 0.2
        open_ = drift - 0.2
        volume = np.full(n, 100.0)

        # Form confirmed swing low at bar 30
        low[30] = 95.0
        # Liquidity sweep at bar 45
        low[45] = 94.0
        close[45] = 102.0
        # Bullish OB displacement at bar 46-47
        open_[46] = 102.0
        close[46] = 101.5
        high[46] = 102.5
        low[46] = 101.0

        open_[47] = 101.5
        close[47] = 106.0
        high[47] = 106.5
        low[47] = 101.2
        volume[47] = 500.0

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
        pipeline = QlibSMCAlphaPipeline(QlibSMCConfig(min_score_threshold=50.0))
        res = pipeline.run(df)

        # Verify limit entry price contract across all rows
        for i in range(len(res)):
            sig = res["signal"].iloc[i]
            c = res["close"].iloc[i]
            px = res["limit_entry_px"].iloc[i]

            if sig == 1:
                # Maker resting bid MUST be in discount zone (<= Close)
                self.assertFalse(np.isnan(px), f"limit_entry_px must be valid float for LONG at bar {i}")
                self.assertGreater(px, 0.0)
                self.assertLessEqual(
                    px,
                    c + 1e-6,
                    f"LONG limit_entry_px ({px}) must be <= Close ({c}) for Maker discount execution",
                )
            elif sig == -1:
                # Maker resting ask MUST be in premium zone (>= Close)
                self.assertFalse(np.isnan(px), f"limit_entry_px must be valid float for SHORT at bar {i}")
                self.assertGreater(px, 0.0)
                self.assertGreaterEqual(
                    px,
                    c - 1e-6,
                    f"SHORT limit_entry_px ({px}) must be >= Close ({c}) for Maker premium execution",
                )
            else:
                # Neutral signal must have NaN limit price
                self.assertTrue(
                    np.isnan(px),
                    f"limit_entry_px must be NaN when signal == 0 at bar {i}, got {px}",
                )

    def test_credal_uncertainty_abstention_behavior(self) -> None:
        """3C. Verify Credal uncertainty prevents trades under conflicting or high-vacuity states."""
        n = 100
        # Flat choppy market (high vacuity, no trend, no volume)
        df_flat = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.1),
            "low": np.full(n, 99.9),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 10.0),
        })
        pipeline = QlibSMCAlphaPipeline(QlibSMCConfig(use_credal_filter=True))
        res_flat = pipeline.run(df_flat)

        # High uncertainty should force abstention and 0 signals
        self.assertTrue(np.all(res_flat["signal"] == 0))
        self.assertTrue(np.all(res_flat["credal_uncertainty"] > 0.35))
        self.assertTrue(np.all(res_flat["should_abstain"]))

    def test_column_name_case_insensitivity(self) -> None:
        """4A. Verify pipeline accepts capitalized or lowercase column names transparently."""
        df = pd.DataFrame({
            "Open": [10.0, 11.0, 12.0],
            "High": [12.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [11.5, 12.5, 13.5],
            "Volume": [100.0, 200.0, 150.0],
        })
        feats = extract_qlib_smc_features(df)
        self.assertIn("orderflow_imbalance", feats.columns)
        self.assertIn("vol_spike", feats.columns)
        self.assertIn("atr_ratio", feats.columns)

    def test_functional_and_class_api_equivalence(self) -> None:
        """4B. Verify functional compute_qlib_smc_alpha matches QlibSMCAlphaPipeline."""
        n = 60
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.2)
        df = pd.DataFrame({
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.uniform(50, 500, n),
        })
        config = QlibSMCConfig(swing_len=4)
        res_fn = compute_qlib_smc_alpha(df, config=config)
        res_class = QlibSMCAlphaPipeline(config=config).run(df)

        pd.testing.assert_frame_equal(res_fn, res_class)


if __name__ == "__main__":
    unittest.main()
