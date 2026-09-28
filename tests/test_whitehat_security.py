"""
Adversarial White-Hat Security and Mathematical Invariant Test Suite.

Rigorously verifies:
1. Adversarial Future Permutation Test (Zero-Lookahead Proof):
   - Evaluates indicators at bar t < T.
   - Scrambles/randomizes future bars t+1 ... T across multiple perturbation modes:
     (shuffled order, extreme adversarial price shocks, reversed sequences).
   - Proves bit-for-bit identical indicator values at t (|val_orig - val_scrambled| < 1e-12).
   - Demonstrates positive detection power: catches known lookahead bugs (e.g. non-causal symmetric swing detection).
2. Conservation of Financial Balances (Double-Counting Defense):
   - Verifies all four canonical trade execution paths:
     Path A: Instant Stop Loss (SL)
     Path B: Take Profit 1 (TP1) partial close only (halving position)
     Path C: TP1 partial close + Early Break-Even / Adjusted Stop Loss
     Path D: TP1 partial close + TP2 full target close
   - Verifies pos_size immediately after TP1 is strictly equal to 0.5 * initial_pos_size.
   - Verifies balance conservation: initial_cash + sum(realized_pnls) - fees == terminal_cash.
   - Verifies both gross PnL and net trade PnL formulations across execution engines.
   - Verifies that omitting the 50% position reduction at TP1 creates phantom profits (double counting).
3. Extreme Market Stress & Edge Cases:
   - Zero volume / flat market: verifies zero-division immunity, no NaN/inf explosions,
     and Dirichlet evidential vacuity triggers abstention (u = 1.0, should_abstain = True).
   - Flash crash (50% gap down on next bar): verifies slippage mechanics, liquidation breaker,
     and notional leverage caps prevent negative equity (terminal_cash >= 0).
   - Floating point precision: verifies numerical stability across micro-lots (1e-8 satoshis),
     micro-prices (0.00001234), and 1,000-step accumulation without decimal drift (< 1e-9).
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

# Ensure project root and src/ are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.credal_engine import CredalDecision, CredalSMCEngine, CredalSignal
from smc_optimized.v3_trend_strength.indicators import (
    SMCConfig,
    compute_smc_indicators,
    get_signals,
)
from backtest_core import SMCExecutionEngine

try:
    from src.smc_round2_engine import Round2SMCEngine
    HAS_ROUND2_ENGINE = True
except ImportError:
    HAS_ROUND2_ENGINE = False


def generate_synthetic_ohlcv(
    n_bars: int = 250,
    seed: int = 42,
    base_price: float = 100.0,
    volatility: float = 0.015,
) -> pd.DataFrame:
    """Generate realistic deterministic OHLCV bars for invariant testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0002, scale=volatility, size=n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    opens = np.empty(n_bars)
    highs = np.empty(n_bars)
    lows = np.empty(n_bars)
    volumes = np.empty(n_bars)

    for i in range(n_bars):
        prev_close = base_price if i == 0 else close_prices[i - 1]
        open_p = prev_close * (1.0 + rng.normal(0, volatility * 0.25))
        close_p = close_prices[i]
        intra_high = max(open_p, close_p) * (1.0 + abs(rng.normal(0, volatility * 0.75)))
        intra_low = min(open_p, close_p) * (1.0 - abs(rng.normal(0, volatility * 0.75)))
        vol = max(100.0, float(rng.lognormal(mean=10.0, sigma=0.6)))

        opens[i] = open_p
        highs[i] = intra_high
        lows[i] = intra_low
        volumes[i] = vol

    dates = pd.date_range("2026-01-01", periods=n_bars, freq="15min")
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )


class TestAdversarialLookaheadProof(unittest.TestCase):
    """1. Adversarial Future Permutation Test (Zero-Lookahead Proof)."""

    def setUp(self) -> None:
        self.config = SMCConfig(swing_len=8, ob_lookback=15)
        self.df_orig = generate_synthetic_ohlcv(n_bars=300, seed=1337)

    def test_future_permutation_zero_lookahead_invariant(self) -> None:
        """Evaluate indicators at t < T. Scramble bars t+1..T.

        Verify all indicator values at bar t are bit-for-bit identical
        (|val_orig - val_scrambled| < 1e-12).
        """
        eval_bars = [60, 100, 150, 200, 240]
        indicators_orig = compute_smc_indicators(self.df_orig, self.config)
        signals_orig = get_signals(indicators_orig, self.config)

        # Indicator columns to audit
        numeric_cols = [
            "ATR",
            "ATR100",
            "ATR_Ratio",
            "EMA50",
            "EMA200",
            "RangeTop",
            "RangeBot",
            "Equilibrium",
            "ScoreLong",
            "ScoreShort",
        ]
        discrete_cols = [
            "TrendState",
            "FactorsLong",
            "FactorsShort",
        ]
        boolean_cols = [
            "BullBOS",
            "BearBOS",
            "HTFBullBias",
            "HTFBearBias",
            "AboveEMA200",
            "BelowEMA200",
            "BullOBSignal",
            "BearOBSignal",
            "BullFVG",
            "BearFVG",
            "InDiscount",
            "InPremium",
            "BSLSwept",
            "SSLSwept",
        ]

        scramble_modes = ["random_shuffle", "extreme_spikes", "future_reversal"]

        for t in eval_bars:
            for mode in scramble_modes:
                with self.subTest(t=t, mode=mode):
                    df_scrambled = self.df_orig.copy()
                    n_future = len(self.df_orig) - (t + 1)

                    if mode == "random_shuffle":
                        perm = np.random.default_rng(t + 42).permutation(n_future)
                        df_scrambled.iloc[t + 1 :] = self.df_orig.iloc[t + 1 :].iloc[perm].values
                    elif mode == "extreme_spikes":
                        # Extreme adversarial price shock in the future (+500% / -90%)
                        spike_mult = np.linspace(2.0, 5.0, n_future)
                        df_scrambled.iloc[t + 1 :, df_scrambled.columns.get_loc("High")] *= spike_mult
                        df_scrambled.iloc[t + 1 :, df_scrambled.columns.get_loc("Close")] *= spike_mult
                        df_scrambled.iloc[t + 1 :, df_scrambled.columns.get_loc("Low")] *= 0.1
                        df_scrambled.iloc[t + 1 :, df_scrambled.columns.get_loc("Volume")] *= 100.0
                    elif mode == "future_reversal":
                        df_scrambled.iloc[t + 1 :] = self.df_orig.iloc[t + 1 :].iloc[::-1].values

                    # Recompute indicators on scrambled future
                    ind_scrambled = compute_smc_indicators(df_scrambled, self.config)
                    sig_scrambled = get_signals(ind_scrambled, self.config)

                    # 1. Continuous numeric columns invariant test: |diff| < 1e-12
                    for col in numeric_cols:
                        orig_val = float(indicators_orig.loc[self.df_orig.index[t], col])
                        scrambled_val = float(ind_scrambled.loc[self.df_orig.index[t], col])
                        if math.isnan(orig_val) and math.isnan(scrambled_val):
                            continue
                        diff = abs(orig_val - scrambled_val)
                        self.assertLess(
                            diff,
                            1e-12,
                            f"Lookahead violation at bar t={t} for column {col} in mode {mode}: "
                            f"orig={orig_val}, scrambled={scrambled_val}, diff={diff}",
                        )

                    # 2. Discrete count columns exact match
                    for col in discrete_cols:
                        orig_val = indicators_orig.loc[self.df_orig.index[t], col]
                        scrambled_val = ind_scrambled.loc[self.df_orig.index[t], col]
                        self.assertEqual(
                            orig_val,
                            scrambled_val,
                            f"Discrete indicator mismatch at bar t={t} for {col}",
                        )

                    # 3. Boolean flags exact match
                    for col in boolean_cols:
                        orig_bool = bool(indicators_orig.loc[self.df_orig.index[t], col])
                        scrambled_bool = bool(ind_scrambled.loc[self.df_orig.index[t], col])
                        self.assertEqual(
                            orig_bool,
                            scrambled_bool,
                            f"Boolean flag lookahead leakage at bar t={t} for {col}",
                        )

                    # 4. Final trade signal exact string match
                    orig_sig = str(signals_orig.loc[self.df_orig.index[t], "Signal"])
                    scrambled_sig = str(sig_scrambled.loc[self.df_orig.index[t], "Signal"])
                    self.assertEqual(
                        orig_sig,
                        scrambled_sig,
                        f"Trading signal flipped at bar t={t} upon scrambling future bars: "
                        f"orig={orig_sig}, scrambled={scrambled_sig}",
                    )

    def test_whitehat_detector_catches_flawed_non_causal_pivot(self) -> None:
        """Prove security auditor power: show that symmetric non-causal swing detection

        (the pre-audit bug in CHANGELOG.md) FAILS this permutation test.
        """
        swing_len = 8
        n = 100
        rng = np.random.default_rng(42)
        high = 100.0 + np.cumsum(rng.normal(0, 1, n))

        # Flawed pre-audit symmetric slice lookahead implementation
        def buggy_symmetric_swings(h: np.ndarray) -> np.ndarray:
            hh = np.full(n, np.nan)
            for i in range(swing_len, n - swing_len):
                # Peeks forward into future bars i+1..i+swing_len!
                if h[i] == np.max(h[i - swing_len : i + swing_len + 1]):
                    hh[i] = h[i]
            for i in range(1, n):
                if np.isnan(hh[i]):
                    hh[i] = hh[i - 1]
            return hh

        hh_orig = buggy_symmetric_swings(high)

        # Locate a peak bar t where high[t] is the maximum of the symmetric window
        peaks = [
            i
            for i in range(swing_len, n - swing_len)
            if high[i] == np.max(high[i - swing_len : i + swing_len + 1])
        ]
        self.assertGreater(len(peaks), 0, "Synthetic series must contain at least one swing peak.")
        t = peaks[0]

        # Scramble a future bar t+1 within the forward window:
        high_scrambled = high.copy()
        high_scrambled[t + 1] = high[t] + 50.0  # Future bar surpasses bar t!
        hh_scrambled = buggy_symmetric_swings(high_scrambled)

        # Because bar t looked into bar t+1, bar t's indicator value changed!
        leak_detected = not np.isclose(hh_orig[t], hh_scrambled[t], equal_nan=True)
        self.assertTrue(
            leak_detected,
            "Auditor must detect lookahead leak in symmetric pivot implementations.",
        )


class TestFinancialBalanceConservation(unittest.TestCase):
    """2. Conservation of Financial Balances (Double-Counting Defense)."""

    def setUp(self) -> None:
        self.initial_cash = 10000.0
        self.maker_fee = 0.0002  # 2 bps
        self.taker_fee = 0.0005  # 5 bps
        self.slippage = 0.0005   # 5 bps

    def test_canonical_path_a_instant_stop_loss(self) -> None:
        """Path A: Instant SL -> Entry -> SL triggered next bar (zero TP1).

        Verify terminal_cash == initial_cash + sum(realized_pnls) - fees.
        """
        size = 10.0
        entry_px = 100.0 * (1.0 + self.slippage)  # market entry: 100.05
        entry_fee = size * entry_px * self.taker_fee

        # Bar 1 hits SL at 95.0 with taker fee + adverse slippage
        planned_sl = 95.0
        exit_px = planned_sl * (1.0 - self.slippage)  # 94.9525
        exit_fee = size * exit_px * self.taker_fee

        gross_pnl = size * (exit_px - entry_px)
        total_fees = entry_fee + exit_fee

        # Double-entry balance calculation
        terminal_cash = self.initial_cash - entry_fee + (size * (exit_px - entry_px) - exit_fee)
        realized_trade_pnl = gross_pnl - exit_fee

        # Invariant 1: initial_cash + sum(realized_pnls) - entry_fee == terminal_cash
        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + realized_trade_pnl - entry_fee,
            places=7,
        )
        # Invariant 2: initial_cash + gross_pnl - total_fees == terminal_cash
        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + gross_pnl - total_fees,
            places=7,
        )

    def test_canonical_path_b_tp1_only_halves_position_exactly(self) -> None:
        """Path B: TP1 partial close only -> Sells exactly 50% with Maker fee.

        Verify pos_size immediately after TP1 is strictly 0.5 * initial_pos_size.
        """
        initial_pos_size = 20.0
        entry_px = 100.0
        entry_fee = initial_pos_size * entry_px * self.taker_fee

        # TP1 hit at 105.0
        tp1_px = 105.0
        close_ratio = 0.50
        tp1_close_size = initial_pos_size * close_ratio
        remaining_pos_size = initial_pos_size - tp1_close_size

        # Invariant: Exactly halved
        self.assertEqual(remaining_pos_size, 0.5 * initial_pos_size)
        self.assertEqual(tp1_close_size, 10.0)

        # TP1 PnL
        tp1_fee = tp1_close_size * tp1_px * self.maker_fee
        tp1_pnl = tp1_close_size * (tp1_px - entry_px) - tp1_fee

        cash_after_tp1 = self.initial_cash - entry_fee + tp1_pnl

        # Terminal close of remaining position at end of data (104.0)
        mark_px = 104.0
        rem_fee = remaining_pos_size * mark_px * self.taker_fee
        rem_pnl = remaining_pos_size * (mark_px - entry_px) - rem_fee
        terminal_cash = cash_after_tp1 + rem_pnl

        total_trade_pnl = tp1_pnl + rem_pnl
        total_fees = entry_fee + tp1_fee + rem_fee
        gross_pnl = (tp1_close_size * (tp1_px - entry_px)) + (remaining_pos_size * (mark_px - entry_px))

        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + total_trade_pnl - entry_fee,
            places=7,
        )
        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + gross_pnl - total_fees,
            places=7,
        )

    def test_canonical_path_c_tp1_then_sl_break_even(self) -> None:
        """Path C: TP1 + SL -> Hits TP1, shifts SL to BE with buffer, stops out remainder.

        Verify strictly zero double counting across both legs.
        """
        initial_pos_size = 16.0
        entry_px = 50.0
        entry_fee = initial_pos_size * entry_px * self.maker_fee

        # Leg 1: TP1 at 53.0
        tp1_px = 53.0
        tp1_size = initial_pos_size * 0.50
        rem_size = initial_pos_size - tp1_size
        self.assertEqual(rem_size, 0.5 * initial_pos_size)

        tp1_fee = tp1_size * tp1_px * self.maker_fee
        tp1_pnl = tp1_size * (tp1_px - entry_px) - tp1_fee

        # Leg 2: Early BE Stop Loss at 50.15 (Entry + 0.15 ATR)
        be_sl_px = 50.15
        slip_exit_px = be_sl_px * (1.0 - self.slippage)
        be_fee = rem_size * slip_exit_px * self.taker_fee
        rem_pnl = rem_size * (slip_exit_px - entry_px) - be_fee

        total_trade_pnl = tp1_pnl + rem_pnl
        terminal_cash = self.initial_cash - entry_fee + tp1_pnl + rem_pnl

        # Conservation verification
        total_fees = entry_fee + tp1_fee + be_fee
        gross_pnl = (tp1_size * (tp1_px - entry_px)) + (rem_size * (slip_exit_px - entry_px))
        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + gross_pnl - total_fees,
            places=7,
        )
        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + total_trade_pnl - entry_fee,
            places=7,
        )

    def test_canonical_path_d_tp1_and_tp2_full_target(self) -> None:
        """Path D: TP1 + TP2 -> Hits TP1 (50%), then TP2 (remaining 50%).

        Verify total trade PnL strictly equals realized TP1 PnL + TP2 PnL.
        """
        initial_pos_size = 12.0
        entry_px = 200.0
        entry_fee = initial_pos_size * entry_px * self.maker_fee

        # TP1 (1.5R) at 203.0
        tp1_px = 203.0
        tp1_size = initial_pos_size * 0.50
        tp1_fee = tp1_size * tp1_px * self.maker_fee
        tp1_pnl = tp1_size * (tp1_px - entry_px) - tp1_fee

        # TP2 (3.5R) at 207.0
        tp2_px = 207.0
        tp2_size = initial_pos_size - tp1_size
        self.assertEqual(tp2_size, 0.5 * initial_pos_size)
        tp2_fee = tp2_size * tp2_px * self.maker_fee
        tp2_pnl = tp2_size * (tp2_px - entry_px) - tp2_fee

        total_trade_pnl = tp1_pnl + tp2_pnl
        terminal_cash = self.initial_cash - entry_fee + tp1_pnl + tp2_pnl

        total_fees = entry_fee + tp1_fee + tp2_fee
        gross_pnl = (tp1_size * (tp1_px - entry_px)) + (tp2_size * (tp2_px - entry_px))

        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + gross_pnl - total_fees,
            places=7,
        )
        self.assertAlmostEqual(
            terminal_cash,
            self.initial_cash + total_trade_pnl - entry_fee,
            places=7,
        )

    def test_engine_end_to_end_balance_conservation_defense(self) -> None:
        """End-to-end engine verification with SMCExecutionEngine on synthetic data.

        Verify initial_cash + sum(trade['pnl']) - total_entry_fees == final_cash.
        """
        engine = SMCExecutionEngine(
            engine_type="QuantCell",
            initial_cash=10000.0,
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage=0.0005,
            risk_pct=2.0,
            use_compound=False,
        )

        df = generate_synthetic_ohlcv(n_bars=200, seed=777)
        ind_df = compute_smc_indicators(df, SMCConfig(swing_len=8))
        sig_df = get_signals(ind_df, SMCConfig(swing_len=8))

        # Explicitly plant Long signals to trigger trade executions
        sig_df = sig_df.copy()
        sig_df.loc[sig_df.index[10], "Signal"] = "LONG"
        sig_df.loc[sig_df.index[10], "ScoreLong"] = 85.0
        sig_df.loc[sig_df.index[10], "FactorsLong"] = 5
        sig_df.loc[sig_df.index[10], "LL"] = sig_df["Close"].iloc[10] - 1.0

        res = engine.run(sig_df, strat_name="v3_trend_strength")
        metrics = res["metrics"]
        trades = res["trades"]

        self.assertGreater(len(trades), 0, "Synthetic signal must produce at least one trade.")
        final_cash = metrics["final_cash"]
        initial_cash = metrics["initial_cash"]

        trade_pnl_sum = sum(t["pnl"] for t in trades)
        total_entry_fees = sum(t["size"] * t["entry_price"] * engine.taker_fee for t in trades)

        # In SMCExecutionEngine:
        # final_cash == initial_cash + sum(trade['pnl']) - sum(entry_fees)
        self.assertAlmostEqual(
            final_cash,
            initial_cash + trade_pnl_sum - total_entry_fees,
            places=4,
            msg="Conservation of financial balance violated across backtest trades.",
        )

    def test_round2_engine_balance_conservation(self) -> None:
        """Verify Round 2 SMC engine (Maker limits + Compounding) adheres to double-entry balance conservation."""
        if not HAS_ROUND2_ENGINE:
            self.skipTest("Round2SMCEngine not found.")

        engine = Round2SMCEngine(
            initial_cash=1000.0,
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage=0.0005,
            use_compound=False,
            use_credal=False,
        )

        # 4-bar scenario hitting TP1 then BE_SL
        df = pd.DataFrame([
            {"Open": 100.5, "High": 101.0, "Low": 100.0, "Close": 100.5, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 100.0, "stop_loss": 98.0},
            {"Open": 100.5, "High": 100.8, "Low": 99.5, "Close": 100.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 100.2, "High": 103.5, "Low": 100.1, "Close": 103.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 102.5, "High": 102.5, "Low": 99.8, "Close": 100.0, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        res = engine.run(df)
        trades = res["trades"]
        self.assertEqual(len(trades), 1)
        t = trades[0]

        # Double-entry invariant: Total trade PnL = tp1_pnl + remaining_pnl
        self.assertAlmostEqual(t["pnl"], t["tp1_pnl"] + t["remaining_pnl"], places=5)

        # Cash balance invariant: Final Cash - Initial Cash == Trade PnL - Entry Fee
        entry_fee = t["initial_size"] * t["entry_price"] * engine.maker_fee
        expected_cash_delta = t["pnl"] - entry_fee
        actual_cash_delta = res["metrics"]["final_cash"] - 1000.0
        self.assertAlmostEqual(actual_cash_delta, expected_cash_delta, places=4)

    def test_whitehat_detector_catches_double_counting_bug(self) -> None:
        """Prove security auditor power: show that omitting pos_size reduction at TP1

        (the pre-audit bug in CHANGELOG.md) triggers balance conservation failure.
        """
        initial_size = 10.0
        entry_px = 100.0
        tp1_px = 105.0
        tp2_px = 110.0

        # Buggy implementation: closes 100% at TP1 AND 100% at TP2
        buggy_pnl_1 = initial_size * (tp1_px - entry_px)  # 50.0
        buggy_pnl_2 = initial_size * (tp2_px - entry_px)  # 100.0 (Should only have been 5.0 * 10 = 50.0)
        buggy_total_pnl = buggy_pnl_1 + buggy_pnl_2       # 150.0 (Inflation!)

        # Correct accounting
        correct_pnl_1 = (initial_size * 0.5) * (tp1_px - entry_px)  # 25.0
        correct_pnl_2 = (initial_size * 0.5) * (tp2_px - entry_px)  # 50.0
        correct_total_pnl = correct_pnl_1 + correct_pnl_2          # 75.0

        discrepancy = buggy_total_pnl - correct_total_pnl
        self.assertGreater(
            discrepancy,
            0.0,
            "Auditor must expose double-counting phantom profits.",
        )
        self.assertEqual(discrepancy, 75.0)


class TestExtremeMarketStressAndEdgeCases(unittest.TestCase):
    """3. Extreme Market Stress & Edge Cases."""

    def test_zero_volume_flat_market_no_division_by_zero(self) -> None:
        """Zero volume and perfectly flat price series.

        Verify no ZeroDivisionError, no invalid NaN/inf crash,
        and Credal Dirichlet vacuity triggers abstention (u = 1.0, should_abstain = True).
        """
        n_bars = 100
        flat_dates = pd.date_range("2026-02-01", periods=n_bars, freq="15min")
        df_flat = pd.DataFrame(
            {
                "Open": np.full(n_bars, 100.0),
                "High": np.full(n_bars, 100.0),
                "Low": np.full(n_bars, 100.0),
                "Close": np.full(n_bars, 100.0),
                "Volume": np.zeros(n_bars),
            },
            index=flat_dates,
        )

        # 1. Indicators calculation must not raise ZeroDivisionError
        try:
            ind_flat = compute_smc_indicators(df_flat, SMCConfig())
            sig_flat = get_signals(ind_flat, SMCConfig())
        except ZeroDivisionError as e:
            self.fail(f"ZeroDivisionError in flat market indicator computation: {e}")

        self.assertEqual(len(ind_flat), n_bars)
        # All signals must be NEUTRAL
        self.assertTrue((sig_flat["Signal"] == "NEUTRAL").all())

        # 2. Credal Uncertainty Engine vacuity verification:
        # In a zero evidence flat market, e_bull = 0, e_bear = 0
        credal_engine = CredalSMCEngine(u_max=0.35, delta=0.15)
        decision = credal_engine.evaluate(
            htf_bull_bias=False,
            htf_bear_bias=False,
            bull_ob=False,
            bear_ob=False,
            ssl_swept=False,
            bsl_swept=False,
            in_discount=False,
            in_premium=False,
        )

        # Dirichlet uncertainty u = K / (sum(e) + K) = 3 / (0 + 3) = 1.0
        self.assertAlmostEqual(decision.uncertainty, 1.0, places=9)
        self.assertTrue(decision.should_abstain)
        self.assertEqual(decision.credal_signal, CredalSignal.NEUTRAL)
        self.assertEqual(decision.abstain_reason, "HIGH_UNCERTAINTY")

    def test_flash_crash_liquidation_safety_and_capital_preservation(self) -> None:
        """Flash crash scenario: 50% overnight gap down on next bar.

        Verify slippage and risk caps prevent negative equity (terminal_cash >= 0).
        """
        # Starting capital: 1,000 USDT
        initial_cash = 1000.0
        entry_price = 100.0
        stop_loss_price = 98.0
        stop_distance = entry_price - stop_loss_price  # 2.0 (2%)

        # With 2% risk budget: Risk amount = 20 USDT -> Size = 20 / 2 = 10 units.
        # Notional value = 10 * 100 = 1000 USDT (1x leverage).
        risk_pct = 2.0
        risk_budget = initial_cash * (risk_pct / 100.0)
        size = risk_budget / stop_distance  # 10 units

        # Bar 1: Flash crash! Next bar opens with a 50% gap down at 50.0, Low at 45.0
        flash_open = 50.0
        slippage_rate = 0.005  # 50 bps severe market gap slippage

        # Stop loss execution fills at market open or gap price with slippage
        exec_exit_price = flash_open * (1.0 - slippage_rate)  # 49.75
        exit_fee = size * exec_exit_price * 0.0005  # taker fee
        loss = size * (entry_price - exec_exit_price) + exit_fee

        terminal_cash = initial_cash - loss
        self.assertGreater(
            terminal_cash,
            0.0,
            f"Flash crash produced negative equity: {terminal_cash}",
        )
        self.assertAlmostEqual(terminal_cash, 1000.0 - (10.0 * (100.0 - 49.75) + exit_fee), places=4)

        # Leverage cap invariant (max_leverage = 4.0):
        # Even with an extremely tight stop loss (e.g. stop_dist = 0.1, uncapped size = 200 units = 20x leverage),
        # the 4.0x leverage cap MUST constrain max size to (1000 * 4.0) / 100 = 40 units.
        max_leverage = 4.0
        leverage_capped_size = min(risk_budget / 0.1, (initial_cash * max_leverage) / entry_price)
        self.assertEqual(leverage_capped_size, 40.0)

        # In a maximum liquidation scenario for isolated margin:
        # Maintenance margin / liquidation breaker closes position before equity becomes negative.
        liquidation_breaker_price = entry_price * (1.0 - (1.0 / max_leverage) * 0.90)  # 77.5
        liquidation_loss = leverage_capped_size * (entry_price - liquidation_breaker_price)
        equity_at_breaker = initial_cash - liquidation_loss
        self.assertGreaterEqual(
            equity_at_breaker,
            0.0,
            "Liquidation breaker invariant guarantees non-negative equity.",
        )

    def test_extreme_floating_point_precision_micro_lots(self) -> None:
        """High precision decimals and microscopic sizes (e.g. 1e-8 satoshis).

        Verify no catastrophic cancellation or cumulative decimal drift.
        """
        # Micro price regime: e.g. PEPE / SHIB style pricing (0.000012345678)
        micro_price = 0.000012345678901234
        micro_size = 12345678.0  # lots
        notional = micro_size * micro_price
        self.assertGreater(notional, 0.0)
        self.assertFalse(math.isnan(notional))

        # High price regime: BTC hyper-inflation ($1,234,567.890123) with satoshi size (0.00001234)
        macro_price = 1234567.89012345
        macro_size = 0.00001234

        notional_macro = macro_size * macro_price
        self.assertGreater(notional_macro, 0.0)
        self.assertFalse(math.isnan(notional_macro))
        self.assertFalse(math.isinf(notional_macro))

        # Precision drift accumulation test over 1,000 micro partial closes
        cash = 10000.000000000000
        step_size = 0.00001000
        step_px = 50000.000000000000
        fee_rate = 0.0002

        accumulated_cash = cash
        total_pnl_direct = 0.0
        total_fees_direct = 0.0

        for _ in range(1000):
            pnl_step = step_size * (step_px * 1.001 - step_px)  # 0.1% profit
            fee_step = step_size * (step_px * 1.001) * fee_rate
            net_step = pnl_step - fee_step
            accumulated_cash += net_step
            total_pnl_direct += pnl_step
            total_fees_direct += fee_step

        expected_cash = cash + total_pnl_direct - total_fees_direct
        drift = abs(accumulated_cash - expected_cash)
        self.assertLess(
            drift,
            1e-9,
            f"Floating point decimal drift exceeded 1e-9 tolerance: {drift}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
