"""
Unit tests for Round 2 High-Frequency SMC Execution Engine.

Validates:
1. Maker limit order entry:
   - Order placed at limit_entry_px.
   - Fills with Maker fee (0.02%) and 0.00% slippage on subsequent bar.
   - Order expiry: cancelled if not filled within 3 bars.
2. Accurate double-entry bookkeeping:
   - TP1 (1.5R) partial close: halves position (close_size = initial_pos_size * 0.5) with Maker fee.
   - Early BE buffer: shifts SL to entry_price +- 0.15 * atr.
   - TP2 (3.5R): closes remaining 50% with Maker fee.
   - SL: closes remaining position with Taker fee (0.05%) + 0.05% slippage.
   - Balance conservation / Zero double counting:
     Total Trade PnL strictly equals realized TP1 PnL + remaining exit PnL.
     Final Cash - Initial Cash == Sum of all trade PnLs.
3. Compounding & Risk Control:
   - Dynamic sizing scales with real-time equity: Size = (Equity * 2%) / StopDistance.
   - Position sizing bounded by 4x max notional leverage.
4. Credal Uncertainty Gate:
   - Abstains when u > 0.35 or when factors conflict.
   - Allows limit orders when conviction is high and epistemic uncertainty is low.
"""

import math
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.credal_engine import CredalSMCEngine, CredalSignal
from src.smc_round2_engine import (
    Round2SMCEngine,
    OrderSide,
    PositionSide,
    PendingLimitOrder,
    ActivePosition,
)


class TestRound2Engine(unittest.TestCase):
    """Test suite validating Round 2 SMC execution engine contracts."""

    def setUp(self) -> None:
        self.engine = Round2SMCEngine(
            initial_cash=1000.0,
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage=0.0005,
            risk_pct=2.0,
            use_compound=True,
            max_leverage=4.0,
            rr_tp1=1.5,
            rr_tp2=3.5,
            early_be_buffer_atr=0.15,
            limit_order_expiry_bars=3,
            use_credal=False,  # default off for direct order tests, tested explicitly
        )

    def test_maker_limit_order_fill_and_fees(self) -> None:
        """1. Verify Maker limit order fills with 0.02% maker fee and 0.00% slippage."""
        # Create a 3-bar scenario:
        # Bar 0: Signal generated at Close=100, Limit order placed at 99.0
        # Bar 1: Low drops to 98.5 (<= 99.0), so limit order fills at exactly 99.0
        # Bar 2: Exits at end of data
        df = pd.DataFrame([
            {"Open": 100.0, "High": 101.0, "Low": 99.5, "Close": 100.0, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 99.0, "stop_loss": 97.0},
            {"Open": 100.0, "High": 100.5, "Low": 98.5, "Close": 99.5, "ATR": 1.0, "Signal": "NEUTRAL", "limit_entry_px": 99.0, "stop_loss": 97.0},
            {"Open": 99.5, "High": 100.0, "Low": 99.0, "Close": 99.5, "ATR": 1.0, "Signal": "NEUTRAL", "limit_entry_px": 99.0, "stop_loss": 97.0},
        ])

        result = self.engine.run(df)
        trades = result["trades"]
        self.assertEqual(len(trades), 1)

        trade = trades[0]
        # Maker fill price must be exactly limit_entry_px (no slippage!)
        self.assertEqual(trade["entry_price"], 99.0)

        # Expected size: Risk budget = 1000 * 2% = 20. Stop distance = 99 - 97 = 2.0.
        # Size = 20 / 2.0 = 10.0 units.
        # Max leverage cap: 1000 * 4 / 99 = 40.4 units. 10.0 < 40.4.
        self.assertAlmostEqual(trade["initial_size"], 10.0, places=4)

        # Entry fee = 10.0 * 99.0 * 0.0002 = 0.198
        expected_entry_fee = 10.0 * 99.0 * 0.0002
        self.assertAlmostEqual(trade["total_fees"], expected_entry_fee + (10.0 * 99.5 * 0.0005), places=4)

    def test_maker_limit_order_cancellation_after_expiry(self) -> None:
        """Verify limit order expires and cancels after 3 bars without filling."""
        # Limit order placed at 95.0, but market stays between 98 and 102 for 5 bars
        df = pd.DataFrame([
            {"Open": 100.0, "High": 102.0, "Low": 99.0, "Close": 100.0, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 95.0, "stop_loss": 93.0},
            {"Open": 100.0, "High": 101.0, "Low": 98.0, "Close": 99.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 99.0, "High": 100.0, "Low": 98.0, "Close": 99.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 99.0, "High": 101.0, "Low": 98.0, "Close": 100.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 4: price plunges to 94.0, but order should have expired at bar 3 (idx 0 + 3 = 3)
            {"Open": 100.0, "High": 100.0, "Low": 94.0, "Close": 94.5, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 94.5, "High": 95.0, "Low": 94.0, "Close": 94.8, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        result = self.engine.run(df)
        trades = result["trades"]
        # Order should NOT have executed because it expired before bar 4
        self.assertEqual(len(trades), 0)
        self.assertEqual(result["metrics"]["final_cash"], 1000.0)

    def test_partial_close_tp1_and_tp2_double_entry_bookkeeping(self) -> None:
        """2. Verify TP1 halves position, moves SL to BE, and TP2 closes remainder with zero double-counting."""
        # Entry = 100.0, SL = 98.0 -> Stop dist = 2.0
        # TP1 (1.5R) = 100.0 + 1.5 * 2.0 = 103.0
        # TP2 (3.5R) = 100.0 + 3.5 * 2.0 = 107.0
        # ATR = 1.0
        # BE SL = 100.0 + 0.15 * 1.0 = 100.15
        df = pd.DataFrame([
            # Bar 0: Long signal, limit placed at 100.0
            {"Open": 100.5, "High": 101.0, "Low": 100.0, "Close": 100.5, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 100.0, "stop_loss": 98.0},
            # Bar 1: Fills at 100.0
            {"Open": 100.5, "High": 100.8, "Low": 99.8, "Close": 100.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 2: High reaches 103.5 (>= TP1 103.0). TP1 hit! Sells 50%
            {"Open": 100.2, "High": 103.5, "Low": 100.1, "Close": 103.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 3: High reaches 107.5 (>= TP2 107.0). TP2 hit! Sells remaining 50%
            {"Open": 103.0, "High": 107.5, "Low": 102.5, "Close": 107.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 4: Quiet bar
            {"Open": 107.2, "High": 108.0, "Low": 107.0, "Close": 107.5, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        result = self.engine.run(df)
        trades = result["trades"]
        self.assertEqual(len(trades), 1)

        trade = trades[0]
        self.assertTrue(trade["tp1_hit"])
        self.assertEqual(trade["exit_reason"], "TP2")
        self.assertEqual(trade["entry_price"], 100.0)
        self.assertEqual(trade["exit_price"], 107.0)

        # Initial size = (1000 * 0.02) / 2.0 = 10.0 units
        initial_size = 10.0
        self.assertAlmostEqual(trade["initial_size"], initial_size, places=4)
        self.assertAlmostEqual(trade["exit_size"], 5.0, places=4)  # remaining 50%

        # Double-entry check:
        # TP1 PnL: 5.0 units * (103.0 - 100.0) - (5.0 * 103.0 * 0.0002 maker_fee)
        tp1_fee = 5.0 * 103.0 * 0.0002
        expected_tp1_pnl = 5.0 * 3.0 - tp1_fee
        self.assertAlmostEqual(trade["tp1_pnl"], expected_tp1_pnl, places=4)

        # TP2 PnL: 5.0 units * (107.0 - 100.0) - (5.0 * 107.0 * 0.0002 maker_fee)
        tp2_fee = 5.0 * 107.0 * 0.0002
        expected_rem_pnl = 5.0 * 7.0 - tp2_fee
        self.assertAlmostEqual(trade["remaining_pnl"], expected_rem_pnl, places=4)

        # Total trade PnL strictly equals realized TP1 PnL + remaining exit PnL
        self.assertAlmostEqual(trade["pnl"], expected_tp1_pnl + expected_rem_pnl, places=4)

        # Balance conservation check: Final Cash - Initial Cash == Trade PnL - Entry Fee
        # (Entry fee was charged on fill: 10.0 * 100 * 0.0002 = 0.20)
        entry_fee = 10.0 * 100.0 * 0.0002
        expected_net_cash_delta = (expected_tp1_pnl + expected_rem_pnl) - entry_fee
        actual_net_cash_delta = result["metrics"]["final_cash"] - 1000.0
        self.assertAlmostEqual(actual_net_cash_delta, expected_net_cash_delta, places=4)

    def test_tp1_then_early_be_sl_bookkeeping(self) -> None:
        """Verify TP1 hit followed by early BE stop out has zero double-counting."""
        # Entry = 100.0, SL = 98.0 -> Stop dist = 2.0
        # TP1 (1.5R) = 103.0
        # ATR = 1.0 -> BE SL = 100.0 + 0.15 * 1.0 = 100.15
        df = pd.DataFrame([
            {"Open": 100.5, "High": 101.0, "Low": 100.0, "Close": 100.5, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 100.0, "stop_loss": 98.0},
            {"Open": 100.5, "High": 100.8, "Low": 99.5, "Close": 100.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 2: Hits TP1 at 103.0
            {"Open": 100.2, "High": 103.5, "Low": 100.1, "Close": 103.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 3: Drops to 100.0 (hits BE SL 100.15!)
            {"Open": 102.5, "High": 102.5, "Low": 99.8, "Close": 100.0, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        result = self.engine.run(df)
        trades = result["trades"]
        self.assertEqual(len(trades), 1)

        trade = trades[0]
        self.assertTrue(trade["tp1_hit"])
        self.assertEqual(trade["exit_reason"], "BE_SL")

        # Invariant: Total trade PnL must strictly equal tp1_pnl + remaining_pnl
        self.assertAlmostEqual(trade["pnl"], trade["tp1_pnl"] + trade["remaining_pnl"], places=5)

        # Cash balance invariant
        total_pnl_from_metrics = result["metrics"]["total_pnl"]
        entry_fee = 10.0 * 100.0 * 0.0002
        self.assertAlmostEqual(total_pnl_from_metrics, trade["pnl"] - entry_fee, places=4)

    def test_compounding_position_sizing(self) -> None:
        """3. Verify compounding dynamic sizing adjusts with real-time equity."""
        # Test with use_compound=True vs use_compound=False
        engine_comp = Round2SMCEngine(initial_cash=1000.0, use_compound=True, risk_pct=2.0)
        engine_fixed = Round2SMCEngine(initial_cash=1000.0, use_compound=False, risk_pct=2.0)

        # Capital doubled to 2000.0:
        # Size = (2000 * 0.02) / 2.0 = 20.0 units for compounding
        # Size = (1000 * 0.02) / 2.0 = 10.0 units for fixed
        size_comp = engine_comp.calculate_position_size(equity=2000.0, entry_price=100.0, stop_loss_price=98.0)
        size_fixed = engine_fixed.calculate_position_size(equity=2000.0, entry_price=100.0, stop_loss_price=98.0)

        self.assertAlmostEqual(size_comp, 20.0, places=4)
        self.assertAlmostEqual(size_fixed, 10.0, places=4)

    def test_leverage_cap_safety(self) -> None:
        """Verify position size is capped at 4x notional leverage."""
        engine = Round2SMCEngine(initial_cash=1000.0, max_leverage=4.0, risk_pct=2.0)

        # Extremely tight stop: Stop distance = 0.1 on a 100.0 asset
        # Uncapped size by risk: (1000 * 0.02) / 0.1 = 200 units (Notional = 20,000 = 20x leverage!)
        # Leverage cap: (1000 * 4.0) / 100.0 = 40.0 units (Notional = 4,000 = 4x leverage)
        size = engine.calculate_position_size(equity=1000.0, entry_price=100.0, stop_loss_price=99.9)
        self.assertAlmostEqual(size, 40.0, places=4)

    def test_credal_uncertainty_gate_abstention(self) -> None:
        """4. Verify Credal Uncertainty Gate abstains when uncertainty > 0.35 or conflict exists."""
        engine_credal = Round2SMCEngine(initial_cash=1000.0, use_credal=True, credal_u_max=0.35)

        # Case A: Low evidence / flat market -> vacuity u close to 1.0 -> Abstains!
        df_vacuous = pd.DataFrame([
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 99.5, "stop_loss": 98.0},
            {"Open": 100.0, "High": 100.5, "Low": 99.0, "Close": 99.5, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 99.5, "High": 102.0, "Low": 99.0, "Close": 101.5, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        res_vacuous = engine_credal.run(df_vacuous)
        # Should abstain: zero trades executed
        self.assertEqual(len(res_vacuous["trades"]), 0)

        # Case B: Conflicting evidence (Bullish OB + Bearish HTF) -> Conflict trigger -> Abstains!
        df_conflict = pd.DataFrame([
            {
                "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "ATR": 1.0,
                "Signal": "LONG", "limit_entry_px": 99.5, "stop_loss": 98.0,
                "bull_ob": 1.0, "htf_bear_bias": 1.0, "in_discount": 1.0, "in_premium": 1.0,
            },
            {"Open": 100.0, "High": 100.5, "Low": 99.0, "Close": 99.5, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 99.5, "High": 102.0, "Low": 99.0, "Close": 101.5, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        res_conflict = engine_credal.run(df_conflict)
        self.assertEqual(len(res_conflict["trades"]), 0)

        # Case C: High conviction confluence (HTF Bull Bias + Bull OB + SSL Swept + In Discount) -> Allowed!
        df_conviction = pd.DataFrame([
            {
                "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "ATR": 1.0,
                "Signal": "LONG", "limit_entry_px": 99.5, "stop_loss": 98.0,
                "htf_bull_bias": 1.0, "bull_ob": 1.0, "ssl_swept": 1.0, "in_discount": 1.0, "bull_fvg": 1.0,
            },
            # Bar 1 reaches 99.2 (<= 99.5 limit), order fills
            {"Open": 100.0, "High": 100.5, "Low": 99.2, "Close": 99.5, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 99.5, "High": 102.0, "Low": 99.0, "Close": 101.5, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        res_conviction = engine_credal.run(df_conviction)
        self.assertEqual(len(res_conviction["trades"]), 1)
        self.assertLess(res_conviction["trades"][0]["credal_uncertainty"], 0.35)

    def test_short_trade_full_cycle(self) -> None:
        """Verify Short trade execution, Maker limit fill, and TP execution."""
        # Short signal at 100.0, limit_entry at 101.0, SL = 103.0 (dist = 2.0)
        # TP1 = 101.0 - 1.5 * 2.0 = 98.0
        # TP2 = 101.0 - 3.5 * 2.0 = 94.0
        df = pd.DataFrame([
            {"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0, "ATR": 1.0, "Signal": "SHORT", "limit_entry_px": 101.0, "stop_loss": 103.0},
            # Bar 1: High reaches 101.5 (>= 101.0), short fills as Maker
            {"Open": 100.2, "High": 101.5, "Low": 100.0, "Close": 100.8, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 2: Low reaches 97.5 (<= TP1 98.0), TP1 hit. High kept below BE SL (101.0 - 0.15 = 100.85)
            {"Open": 100.5, "High": 100.6, "Low": 97.5, "Close": 98.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Bar 3: Low reaches 93.5 (<= TP2 94.0), TP2 hit
            {"Open": 98.0, "High": 98.2, "Low": 93.5, "Close": 94.0, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        result = self.engine.run(df)
        trades = result["trades"]
        self.assertEqual(len(trades), 1)

        trade = trades[0]
        self.assertEqual(trade["direction"], "SHORT")
        self.assertEqual(trade["entry_price"], 101.0)
        self.assertEqual(trade["exit_price"], 94.0)
        self.assertEqual(trade["exit_reason"], "TP2")
        self.assertTrue(trade["tp1_hit"])
        self.assertGreater(trade["pnl"], 0.0)

    def test_multi_trade_balance_conservation_invariant(self) -> None:
        """Verify total balance conservation across multiple long & short trades with compounding."""
        # Series of alternating signals and outcomes
        df = pd.DataFrame([
            # Trade 1: Long at 100.0, fills, hits TP1, hits TP2
            {"Open": 100.5, "High": 101.0, "Low": 100.0, "Close": 100.5, "ATR": 1.0, "Signal": "LONG", "limit_entry_px": 100.0, "stop_loss": 98.0},
            {"Open": 100.5, "High": 100.8, "Low": 99.8, "Close": 100.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 100.2, "High": 103.5, "Low": 100.1, "Close": 103.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 103.0, "High": 107.5, "Low": 102.5, "Close": 107.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Flat bar
            {"Open": 107.2, "High": 107.5, "Low": 106.8, "Close": 107.0, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Trade 2: Short at 107.0, fills, hits SL
            {"Open": 107.0, "High": 107.5, "Low": 106.5, "Close": 107.0, "ATR": 1.0, "Signal": "SHORT", "limit_entry_px": 107.0, "stop_loss": 109.0},
            {"Open": 106.8, "High": 107.2, "Low": 106.5, "Close": 106.9, "ATR": 1.0, "Signal": "NEUTRAL"},
            {"Open": 107.0, "High": 109.5, "Low": 106.8, "Close": 109.2, "ATR": 1.0, "Signal": "NEUTRAL"},
            # Flat bar
            {"Open": 109.2, "High": 109.5, "Low": 109.0, "Close": 109.3, "ATR": 1.0, "Signal": "NEUTRAL"},
        ])

        result = self.engine.run(df)
        trades = result["trades"]
        self.assertEqual(len(trades), 2)

        # Total PnL across trades minus entry fees must exactly match change in cash
        sum_trade_pnl = sum(t["pnl"] for t in trades)
        sum_entry_fees = sum(t["initial_size"] * t["entry_price"] * self.engine.maker_fee for t in trades)
        expected_net_change = sum_trade_pnl - sum_entry_fees
        actual_net_change = result["metrics"]["final_cash"] - self.engine.initial_cash

        self.assertAlmostEqual(actual_net_change, expected_net_change, places=4)
        self.assertAlmostEqual(result["metrics"]["total_pnl"], actual_net_change, places=4)

        # Invariant for each trade: pnl == tp1_pnl + remaining_pnl
        for t in trades:
            self.assertAlmostEqual(t["pnl"], t["tp1_pnl"] + t["remaining_pnl"], places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
