import os
import sys
import gc
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from backtest_core import load_symbol_timeframe_data
from src.qlib_smc_alpha import compute_qlib_smc_alpha
from src.smc_round2_engine import Round2SMCEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_round2_runner")

SYMBOLS = ["BTC", "BNB", "SOL", "UNIUSDT", "NEAR", "XAUUSD", "USOUSD", "GBPUSD"]
TIMEFRAMES = ["15m", "1h"]
YEARS = [2021, 2022, 2023, 2024, 2025]
OUTPUT_DIR = Path("results/round2_backtests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "trades").mkdir(parents=True, exist_ok=True)

def run_single_walkforward(sym: str, tf: str, year: int) -> List[Dict[str, Any]]:
    records = []
    try:
        df = load_symbol_timeframe_data(sym, tf)
    except Exception as e:
        logger.error(f"Error loading {sym} {tf}: {e}")
        return records

    year_df = df[df.index.year == year]
    if len(year_df) < 50:
        return records

    # 80/20 train/test split with 200 bars warm-up
    n_total = len(year_df)
    n_in_sample = int(n_total * 0.8)

    # In-Sample slice
    in_sample_df = year_df.iloc[:n_in_sample]
    # Out-of-Sample slice with 200 bars warm-up
    start_oos_idx = max(0, n_in_sample - 200)
    out_sample_df = year_df.iloc[start_oos_idx:]

    slices = [
        ("In-Sample", in_sample_df, False),
        ("Out-of-Sample", out_sample_df, True),
    ]

    for sample_type, sub_df, has_warmup in slices:
        try:
            alpha_df = compute_qlib_smc_alpha(sub_df)
            if has_warmup and len(alpha_df) > 200:
                eval_df = alpha_df.iloc[200:]
            else:
                eval_df = alpha_df

            engine = Round2SMCEngine(
                initial_cash=1000.0,
                maker_fee=0.0002,
                taker_fee=0.0005,
                slippage=0.0005,
                risk_pct=1.0,
                use_compound=True,
                max_leverage=4.0,
                use_credal=True,
                credal_u_max=0.55,
            )
            res = engine.run(eval_df)
            m = res["metrics"]
            trades = res["trades"]

            # Save individual trades
            trade_file = OUTPUT_DIR / "trades" / f"{sym}_{tf}_{year}_{sample_type.lower().replace('-', '_')}_trades.json"
            with open(trade_file, "w") as fp:
                json.dump(trades, fp, indent=2, default=str)

            rec = {
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
                "total_slippage": m.get("total_slippage_cost", 0.0),
                "maker_fill_ratio": m.get("maker_fill_ratio", 0.0),
                "ending_equity": m.get("final_cash", 1000.0),
                "trade_file": str(trade_file),
            }
            records.append(rec)
            logger.info(f"{sym} {tf} {year} [{sample_type}]: Trades={m['total_trades']}, Ret={m['total_return_pct']:+.2f}%, WinRate={m['win_rate_pct']:.1f}%, MDD={m['max_drawdown_pct']:.2f}%")
        except Exception as e:
            logger.error(f"Error running {sym} {tf} {year} {sample_type}: {e}")
        finally:
            gc.collect()

    return records

def main():
    all_records = []
    logger.info("Starting Round 2 High-Frequency SMC Walk-Forward Backtests across 8 symbols, 2 timeframes, 5 years...")
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            for year in YEARS:
                recs = run_single_walkforward(sym, tf, year)
                all_records.extend(recs)

    df_summary = pd.DataFrame(all_records)
    summary_file = OUTPUT_DIR / "round2_backtest_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    logger.info(f"All backtests completed! Summary saved to {summary_file}. Total rows: {len(df_summary)}")

if __name__ == "__main__":
    main()
