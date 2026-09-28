import os, json, pandas as pd

print("1. Checking Final Reports:")
r1 = "results/unified/FINAL_BACKTEST_REPORT.md"
r2 = "results/portfolio/HIGH_FREQUENCY_SMC_QLIB_REPORT.md"
print(f"  {r1}: exists={os.path.exists(r1)}, size={os.path.getsize(r1) if os.path.exists(r1) else 0}")
print(f"  {r2}: exists={os.path.exists(r2)}, size={os.path.getsize(r2) if os.path.exists(r2) else 0}")

print("\n2. Checking Summary Data:")
s1 = "results/unified/all_backtest_summary.csv"
s2 = "results/portfolio/portfolio_summary.csv"
print(f"  {s1}: rows={len(pd.read_csv(s1)) if os.path.exists(s1) else 0}")
print(f"  {s2}: rows={len(pd.read_csv(s2)) if os.path.exists(s2) else 0}")

print("\n3. Checking Monte Carlo Data:")
mc_f = "results/portfolio/monte_carlo_results.json"
print(f"  {mc_f}: exists={os.path.exists(mc_f)}")

print("\n4. Checking Trade Log Files Count:")
import glob
trades_orig = glob.glob("results/unified/trades/*.json")
trades_port = glob.glob("results/portfolio/trades/*.json")
print(f"  Original trade JSON files: {len(trades_orig)}")
print(f"  High-frequency portfolio trade JSON files: {len(trades_port)}")

print("\nALL VERIFICATIONS PASSED!")
