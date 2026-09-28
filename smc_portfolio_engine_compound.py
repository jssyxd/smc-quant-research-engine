import sys, os, gc, glob, math, json
import pandas as pd
import numpy as np
from backtest_core import load_symbol_timeframe_data, SMCExecutionEngine

SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'NEAR']

def run_portfolio_compound():
    os.makedirs('results/portfolio/trades', exist_ok=True)
    os.makedirs('results/portfolio/reports', exist_ok=True)

    summary = []
    all_trades_corpus = []

    # Portfolio pool model: Single $1,000 account trading the multi-asset crypto portfolio
    # Each trade risks 0.5% ~ 1.0% of current equity (total portfolio risk controlled)
    initial_cash = 1000.0

    for year in range(2021, 2026):
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
                pass

        # Sub-account sizing: each asset has initial $200 allocated out of $1,000 total pool
        sub_initial = initial_cash / len(dfs_is)

        engine = SMCExecutionEngine(
            engine_type='NautilusTrader',
            initial_cash=sub_initial,
            maker_fee=0.0002,
            taker_fee=0.0005,
            slippage=0.0005,
            risk_pct=2.0
        )

        year_is_trades = []
        year_oos_trades = []
        year_is_pnl = 0.0
        year_oos_pnl = 0.0
        year_is_fees = 0.0
        year_oos_fees = 0.0
        year_is_slip = 0.0
        year_oos_slip = 0.0

        for sym in dfs_is:
            res_is = engine.run(dfs_is[sym], 'v2_volatility_regime')
            res_oos = engine.run(dfs_oos[sym], 'v2_volatility_regime')

            for t in res_is['trades']:
                t['symbol'] = sym
                t['period'] = 'IS'
                t['year'] = year
                year_is_trades.append(t)
                all_trades_corpus.append(t)

            for t in res_oos['trades']:
                t['symbol'] = sym
                t['period'] = 'OOS'
                t['year'] = year
                year_oos_trades.append(t)
                all_trades_corpus.append(t)

            year_is_pnl += res_is['metrics']['total_pnl']
            year_oos_pnl += res_oos['metrics']['total_pnl']
            year_is_fees += res_is['metrics']['total_fees']
            year_oos_fees += res_oos['metrics']['total_fees']
            year_is_slip += res_is['metrics']['total_slippage_cost']
            year_oos_slip += res_oos['metrics']['total_slippage_cost']

        year_is_trades.sort(key=lambda x: x['entry_time'])
        year_oos_trades.sort(key=lambda x: x['entry_time'])

        is_days = 365.25 * 0.8
        oos_days = 365.25 * 0.2

        is_daily_freq = len(year_is_trades) / is_days
        oos_daily_freq = len(year_oos_trades) / oos_days

        is_ret_pct = (year_is_pnl / initial_cash) * 100.0
        oos_ret_pct = (year_oos_pnl / initial_cash) * 100.0

        is_win = len([t for t in year_is_trades if t['pnl'] > 0])
        oos_win = len([t for t in year_oos_trades if t['pnl'] > 0])
        is_win_rate = (is_win / len(year_is_trades) * 100.0) if year_is_trades else 0.0
        oos_win_rate = (oos_win / len(year_oos_trades) * 100.0) if year_oos_trades else 0.0

        rec = {
            "year": year,
            "symbols_count": len(dfs_is),
            "is_trades": len(year_is_trades),
            "oos_trades": len(year_oos_trades),
            "is_daily_freq": is_daily_freq,
            "oos_daily_freq": oos_daily_freq,
            "is_return_pct": is_ret_pct,
            "oos_return_pct": oos_ret_pct,
            "is_win_rate": is_win_rate,
            "oos_win_rate": oos_win_rate,
            "is_pnl": year_is_pnl,
            "oos_pnl": year_oos_pnl,
            "is_fees": year_is_fees,
            "oos_fees": year_oos_fees,
            "is_slippage": year_is_slip,
            "oos_slippage": year_oos_slip
        }
        summary.append(rec)
        print(f"[{year}] IS: {len(year_is_trades)} ({is_daily_freq:.2f}/d, +{is_ret_pct:.1f}%) | OOS: {len(year_oos_trades)} ({oos_daily_freq:.2f}/d, +{oos_ret_pct:.1f}%) | Fees: ${year_oos_fees:.1f}")

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv("results/portfolio/portfolio_summary.csv", index=False)
    print("\nPortfolio Summary Saved!")
    print(df_summary[['year', 'is_daily_freq', 'oos_daily_freq', 'is_return_pct', 'oos_return_pct', 'oos_win_rate']])

    with open("results/portfolio/all_trades_corpus.json", "w") as fp:
        json.dump(all_trades_corpus, fp)

if __name__ == '__main__':
    run_portfolio_compound()
