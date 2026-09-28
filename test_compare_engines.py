import sys, os
import pandas as pd
import numpy as np

# Load 1 month of BTC 1h
df = pd.read_parquet('data/raw_extracted/BTC/BTCUSDT_1h_ohlcv_2024-01.parquet')

# Test original SMCStrategy
from smc_optimized.v1_internal_factors.indicators import SMCConfig, compute_smc_indicators, get_signals
from smc_optimized.v1_internal_factors.smc_strategy import SMCStrategy

strat = SMCStrategy(config=SMCConfig(), initial_cash=1000.0, commission=0.0005)
strat.prepare_data(df)
res = strat.run_backtest()
print("Original Backtest Summary:", res.summary())
print("Total Trades:", len(res.trades))
for t in res.trades[:5]:
    print("  Trade:", t.direction.name, t.entry_time, "->", t.exit_time, "pnl:", t.pnl, "reason:", t.exit_reason)
