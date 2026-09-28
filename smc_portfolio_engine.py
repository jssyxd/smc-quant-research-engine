import sys, os, gc, glob, math
import pandas as pd
import numpy as np
from backtest_core import load_symbol_timeframe_data, SMCExecutionEngine

SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'NEAR']

def run_portfolio_backtest(year: int, initial_cash: float = 1000.0, use_compound: bool = True):
    """
    Multi-Asset Crypto Portfolio Backtest:
    - Symbols: BTC, ETH, SOL, BNB, NEAR
    - Timeframe: 15m
    - Equal Risk Allocation: capital shared across 5 mainstream assets
    - Aggregated trades give 1.0 ~ 3.5 trades/day
    - 80% In-Sample / 20% Out-of-Sample per natural year
    """
    dfs_is = {}
    dfs_oos = {}

    for sym in SYMBOLS:
        try:
            df = load_symbol_timeframe_data(sym, '15m')
            df_y = df[df.index.year == year]
            if len(df_y) >= 100:
                split_idx = int(len(df_y) * 0.8)
                dfs_is[sym] = df_y.iloc[:split_idx]
                dfs_oos[sym] = df_y.iloc[split_idx:]
        except Exception as e:
            # print error if symbol missing
            pass

    print(f"Year {year}: Loaded {len(dfs_is)} symbols for portfolio backtest.")

    # We evaluate individual asset runs and then compute portfolio composite metrics
    results_is = {}
    results_oos = {}

    engine = SMCExecutionEngine(
        engine_type='NautilusTrader',
        initial_cash=initial_cash / len(dfs_is), # sub-account allocation
        maker_fee=0.0002,
        taker_fee=0.0005,
        slippage=0.0005,
        risk_pct=2.0
    )

    for sym in dfs_is:
        results_is[sym] = engine.run(dfs_is[sym], 'v2_volatility_regime')
        results_oos[sym] = engine.run(dfs_oos[sym], 'v2_volatility_regime')

    # Aggregate Portfolio Trades & PnL
    all_is_trades = []
    all_oos_trades = []
    total_is_pnl = 0.0
    total_oos_pnl = 0.0

    for sym in dfs_is:
        for t in results_is[sym]['trades']:
            t['symbol'] = sym
            all_is_trades.append(t)
        for t in results_oos[sym]['trades']:
            t['symbol'] = sym
            all_oos_trades.append(t)
        total_is_pnl += results_is[sym]['metrics']['total_pnl']
        total_oos_pnl += results_oos[sym]['metrics']['total_pnl']

    # Sort trades by entry time
    all_is_trades.sort(key=lambda x: x['entry_time'])
    all_oos_trades.sort(key=lambda x: x['entry_time'])

    is_days = 365.25 * 0.8
    oos_days = 365.25 * 0.2

    is_daily_freq = len(all_is_trades) / is_days
    oos_daily_freq = len(all_oos_trades) / oos_days

    is_return_pct = (total_is_pnl / initial_cash) * 100.0
    oos_return_pct = (total_oos_pnl / initial_cash) * 100.0

    is_win_trades = [t for t in all_is_trades if t['pnl'] > 0]
    oos_win_trades = [t for t in all_oos_trades if t['pnl'] > 0]

    is_win_rate = (len(is_win_trades) / len(all_is_trades) * 100.0) if all_is_trades else 0.0
    oos_win_rate = (len(oos_win_trades) / len(all_oos_trades) * 100.0) if all_oos_trades else 0.0

    return {
        "year": year,
        "is_trades_count": len(all_is_trades),
        "oos_trades_count": len(all_oos_trades),
        "is_daily_freq": is_daily_freq,
        "oos_daily_freq": oos_daily_freq,
        "is_return_pct": is_return_pct,
        "oos_return_pct": oos_return_pct,
        "is_win_rate": is_win_rate,
        "oos_win_rate": oos_win_rate,
        "is_pnl": total_is_pnl,
        "oos_pnl": total_oos_pnl,
        "is_trades": all_is_trades,
        "oos_trades": all_oos_trades
    }

if __name__ == '__main__':
    for y in range(2021, 2026):
        res = run_portfolio_backtest(y)
        print(f"[{y}] IS Trades: {res['is_trades_count']} ({res['is_daily_freq']:.2f}/day, Ret: {res['is_return_pct']:.2f}%) | OOS Trades: {res['oos_trades_count']} ({res['oos_daily_freq']:.2f}/day, Ret: {res['oos_return_pct']:.2f}%)")
