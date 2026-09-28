import sys, os, json, random
import pandas as pd
import numpy as np

def run_monte_carlo_simulation(n_simulations: int = 2000, trade_sample_size: int = 200, initial_capital: float = 1000.0):
    """
    Monte Carlo Permutation & Resampling Simulation:
    - Extracts all real Out-of-Sample trade PnLs from the portfolio backtest
    - Runs 2000 bootstrapped path simulations
    - Calculates:
      1. Expected Terminal Wealth & Median Return
      2. 95% Confidence Interval of Maximum Drawdown
      3. Probability of Ruin (Equity < 50% of initial)
      4. Value-at-Risk (VaR 95%) and Conditional VaR (CVaR)
    """
    with open("results/portfolio/all_trades_corpus.json") as fp:
        all_trades = json.load(fp)

    # Filter Out-of-Sample trades only
    oos_trades = [t for t in all_trades if t.get('period') == 'OOS']
    print(f"Total Out-of-Sample trades loaded: {len(oos_trades)}")

    pnl_list = [t['pnl'] for t in oos_trades]
    pnl_pct_list = [t['pnl_pct'] for t in oos_trades]

    terminal_wealths = []
    max_drawdowns = []
    ruin_count = 0

    random.seed(42)
    np.random.seed(42)

    for sim in range(n_simulations):
        # Sample with replacement
        sampled_pnls = np.random.choice(pnl_list, size=trade_sample_size, replace=True)

        equity_path = [initial_capital]
        curr_eq = initial_capital
        ruined = False

        for pnl in sampled_pnls:
            curr_eq += pnl
            if curr_eq <= initial_capital * 0.5:
                ruined = True
            equity_path.append(curr_eq)

        if ruined:
            ruin_count += 1

        terminal_wealths.append(curr_eq)

        # Max drawdown of this path
        eq_arr = np.array(equity_path)
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / np.maximum(peaks, 1e-9)
        max_drawdowns.append(np.max(dds) * 100.0)

    terminal_wealths = np.array(terminal_wealths)
    max_drawdowns = np.array(max_drawdowns)

    returns = (terminal_wealths - initial_capital) / initial_capital * 100.0

    mc_results = {
        "n_simulations": n_simulations,
        "sample_trades_per_path": trade_sample_size,
        "initial_capital": initial_capital,
        "mean_terminal_wealth": float(np.mean(terminal_wealths)),
        "median_terminal_wealth": float(np.median(terminal_wealths)),
        "mean_return_pct": float(np.mean(returns)),
        "median_return_pct": float(np.median(returns)),
        "percentile_5_return_pct": float(np.percentile(returns, 5)),
        "percentile_95_return_pct": float(np.percentile(returns, 95)),
        "mean_max_drawdown_pct": float(np.mean(max_drawdowns)),
        "percentile_95_max_drawdown_pct": float(np.percentile(max_drawdowns, 95)),
        "percentile_99_max_drawdown_pct": float(np.percentile(max_drawdowns, 99)),
        "probability_of_ruin_pct": float(ruin_count / n_simulations * 100.0),
        "var_95_pct": float(-np.percentile(returns, 5)),
        "cvar_95_pct": float(-np.mean(returns[returns <= np.percentile(returns, 5)]))
    }

    print("\n=== 蒙特卡洛压力测试与置信区间 (2000次重采样) ===")
    for k, v in mc_results.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    with open("results/portfolio/monte_carlo_results.json", "w") as fp:
        json.dump(mc_results, fp, indent=2)

    return mc_results

if __name__ == '__main__':
    run_monte_carlo_simulation()
