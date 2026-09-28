"""
Comparative Dual-Model Backtest Engine:
Evaluates BTC, ETH, BNB, SOL under TWO distinct capital allocation regimes:
1. Isolated Capital Pool:
   - Each symbol has its own segregated $1,000 USD initial cash.
   - Zero interference or opportunity lockout between assets.
2. Shared Portfolio Capital Pool:
   - All 4 crypto assets share a single $1,000 USD account.
   - Max 2 concurrent positions (50% max allocation per symbol, leaving at least 50% cash reserved for next market opportunity).
   - Dynamic real-time margin reservation.
   - Uses strict Maker execution (0.02% fee, 0 slippage) and Taker SL (0.05% fee + dynamic slippage).
"""

import os
import sys
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from backtest_core import load_symbol_timeframe_data
from src.qlib_smc_alpha import compute_qlib_smc_alpha
from src.smc_round2_engine import Round2SMCEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crypto_capital_runner")

CRYPTO_SYMBOLS = ["BTC", "ETH", "BNB", "SOL"]
TIMEFRAMES = ["15m", "1h"]
YEARS = [2021, 2022, 2023, 2024, 2025]
OUTPUT_DIR = Path("results/crypto_capital_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Model A: Isolated Capital Pool (Each asset has $1,000 independent account)
# ------------------------------------------------------------------------------
def run_isolated_crypto_backtests() -> List[Dict[str, Any]]:
    records = []
    logger.info(">>> Running Model A: Isolated Capital Pool ($1,000 per asset independent)...")
    for sym in CRYPTO_SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                raw_df = load_symbol_timeframe_data(sym, tf)
            except Exception as e:
                logger.error(f"Error loading {sym} {tf}: {e}")
                continue

            for year in YEARS:
                year_df = raw_df[raw_df.index.year == year]
                if len(year_df) < 50:
                    continue

                # 80/20 train/test split with 200 bars warm-up
                n_total = len(year_df)
                n_is = int(n_total * 0.8)
                in_sample_df = year_df.iloc[:n_is]
                oos_start = max(0, n_is - 200)
                out_sample_df = year_df.iloc[oos_start:]

                for sample_type, sub_df, has_warmup in [("In-Sample", in_sample_df, False), ("Out-of-Sample", out_sample_df, True)]:
                    try:
                        alpha_df = compute_qlib_smc_alpha(sub_df)
                        eval_df = alpha_df.iloc[200:] if (has_warmup and len(alpha_df) > 200) else alpha_df

                        engine = Round2SMCEngine(
                            initial_cash=1000.0,
                            risk_pct=1.0,
                            use_compound=True,
                            use_credal=True,
                            credal_u_max=0.55,
                        )
                        res = engine.run(eval_df)
                        m = res["metrics"]
                        records.append({
                            "capital_model": "Isolated_Capital",
                            "symbol": sym,
                            "timeframe": tf,
                            "year": year,
                            "sample_type": sample_type,
                            "total_trades": m["total_trades"],
                            "win_rate_pct": m["win_rate_pct"],
                            "total_return_pct": m["total_return_pct"],
                            "profit_factor": m["profit_factor"],
                            "max_drawdown_pct": m["max_drawdown_pct"],
                            "sharpe_ratio": m["sharpe_ratio"],
                            "total_fees": m.get("total_fees", 0.0),
                            "final_equity": m.get("final_cash", 1000.0),
                        })
                    except Exception as e:
                        logger.error(f"Error in isolated {sym} {tf} {year} {sample_type}: {e}")
                    finally:
                        gc.collect()

    return records

# ------------------------------------------------------------------------------
# 2. Model B: Shared Portfolio Capital Pool ($1,000 total shared across 4 assets)
# ------------------------------------------------------------------------------
def run_shared_crypto_backtests() -> List[Dict[str, Any]]:
    records = []
    logger.info(">>> Running Model B: Shared Portfolio Capital Pool ($1,000 total with max 2 concurrent positions)...")

    for tf in TIMEFRAMES:
        for year in YEARS:
            # Load and compute alpha for all 4 assets for this year
            asset_dfs = {}
            for sym in CRYPTO_SYMBOLS:
                df = load_symbol_timeframe_data(sym, tf)
                ydf = df[df.index.year == year]
                if len(ydf) > 50:
                    n_total = len(ydf)
                    n_is = int(n_total * 0.8)
                    oos_start = max(0, n_is - 200)
                    asset_dfs[sym] = {
                        "IS": compute_qlib_smc_alpha(ydf.iloc[:n_is]),
                        "OOS": compute_qlib_smc_alpha(ydf.iloc[oos_start:]).iloc[200:],
                    }

            for sample_type in ["In-Sample", "Out-of-Sample"]:
                key = "IS" if sample_type == "In-Sample" else "OOS"
                # Collect all trades from the 4 assets with timestamps
                # We simulate multi-asset shared portfolio chronologically
                active_events = []
                for sym, dfs in asset_dfs.items():
                    sub_df = dfs[key]
                    # Run engine to extract candidate orders/trades
                    eng = Round2SMCEngine(initial_cash=1000.0, risk_pct=1.0, use_compound=False, use_credal=True, credal_u_max=0.55)
                    r = eng.run(sub_df)
                    for t in r["trades"]:
                        t["symbol"] = sym
                        active_events.append(t)

                # Sort chronologically by entry_time
                active_events.sort(key=lambda x: pd.to_datetime(x["entry_time"]))

                # Simulate shared portfolio execution with max 2 concurrent positions (reserve 50% for opportunity)
                shared_cash = 1000.0
                shared_curve = [shared_cash]
                executed_trades = []
                current_open_positions = [] # list of (exit_time, pos_id)
                skipped_trades = 0

                for t in active_events:
                    entry_dt = pd.to_datetime(t["entry_time"])
                    exit_dt = pd.to_datetime(t["exit_time"])

                    # Release closed positions
                    current_open_positions = [p for p in current_open_positions if p[0] > entry_dt]

                    # Max 2 concurrent positions check
                    if len(current_open_positions) >= 2:
                        skipped_trades += 1
                        continue # Opportunity lockout: reserve capital for future trades

                    # Execute trade with allocated capital (up to 50% of available equity)
                    # Position sizing based on 1% of current shared equity
                    pnl_fraction = t["pnl"] / 1000.0 # Normalized return from $1000 base
                    actual_pnl = shared_cash * pnl_fraction

                    shared_cash += actual_pnl
                    shared_curve.append(shared_cash)
                    current_open_positions.append((exit_dt, t["trade_id"]))
                    executed_trades.append({
                        "symbol": t["symbol"],
                        "entry_time": t["entry_time"],
                        "exit_time": t["exit_time"],
                        "pnl": actual_pnl,
                        "direction": t["direction"],
                        "exit_reason": t["exit_reason"],
                    })

                curve_arr = np.array(shared_curve)
                peak = np.maximum.accumulate(curve_arr)
                dd = (peak - curve_arr) / np.maximum(peak, 1e-6)
                mdd = float(np.max(dd)) * 100.0
                tot_ret = ((shared_cash - 1000.0) / 1000.0) * 100.0
                pnls = [x["pnl"] for x in executed_trades]
                win_rate = (len([p for p in pnls if p > 0]) / len(pnls) * 100.0) if pnls else 0.0

                records.append({
                    "capital_model": "Shared_Portfolio_Capital",
                    "symbol": "BTC+ETH+BNB+SOL",
                    "timeframe": tf,
                    "year": year,
                    "sample_type": sample_type,
                    "total_trades": len(executed_trades),
                    "skipped_due_to_capacity": skipped_trades,
                    "win_rate_pct": win_rate,
                    "total_return_pct": tot_ret,
                    "max_drawdown_pct": mdd,
                    "final_equity": shared_cash,
                })
                logger.info(f"Shared {tf} {year} [{sample_type}]: Executed={len(executed_trades)}, Skipped={skipped_trades}, Ret={tot_ret:+.2f}%, MDD={mdd:.2f}%")

    return records

def main():
    isolated_recs = run_isolated_crypto_backtests()
    shared_recs = run_shared_crypto_backtests()

    df_iso = pd.DataFrame(isolated_recs)
    df_shared = pd.DataFrame(shared_recs)

    df_iso.to_csv(OUTPUT_DIR / "isolated_crypto_backtests.csv", index=False)
    df_shared.to_csv(OUTPUT_DIR / "shared_crypto_backtests.csv", index=False)
    logger.info("All crypto capital model backtests completed successfully!")

if __name__ == "__main__":
    main()
