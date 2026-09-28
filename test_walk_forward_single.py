import sys, os, gc
import pandas as pd
from backtest_core import load_symbol_timeframe_data, SMCExecutionEngine

# Load BTC 1h
df = load_symbol_timeframe_data('BTC', '1h')
print(f"Loaded BTC 1h: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

# Natural year rolling window test
years = sorted(list(set(df.index.year)))
print("Available years:", years)

engine = SMCExecutionEngine(
    engine_type='QuantCell',
    initial_cash=1000.0,
    maker_fee=0.0002,
    taker_fee=0.0005,
    slippage=0.0005,
    risk_pct=2.0
)

for y in [2023, 2024]:
    df_y = df[df.index.year == y]
    if len(df_y) < 100:
        continue
    # 80% In-sample, 20% Out-of-sample
    split_idx = int(len(df_y) * 0.8)
    df_is = df_y.iloc[:split_idx]
    df_oos = df_y.iloc[split_idx:]

    print(f"\n--- Year {y} ---")
    print(f"IS Range: {df_is.index[0]} to {df_is.index[-1]} ({len(df_is)} bars)")
    print(f"OOS Range: {df_oos.index[0]} to {df_oos.index[-1]} ({len(df_oos)} bars)")

    res_is = engine.run(df_is, 'v1_internal_factors')
    res_oos = engine.run(df_oos, 'v1_internal_factors')

    print(f"IS Metrics: Ret={res_is['metrics']['total_return_pct']:.2f}%, Trades={res_is['metrics']['total_trades']}, WinRate={res_is['metrics']['win_rate_pct']:.1f}%, MDD={res_is['metrics']['max_drawdown_pct']:.2f}%")
    print(f"OOS Metrics: Ret={res_oos['metrics']['total_return_pct']:.2f}%, Trades={res_oos['metrics']['total_trades']}, WinRate={res_oos['metrics']['win_rate_pct']:.1f}%, MDD={res_oos['metrics']['max_drawdown_pct']:.2f}%")

