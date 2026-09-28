import sys, os, glob, pandas as pd
from smc_hf_alpha_engine import HighFrequencySMCEngine

files = sorted(glob.glob('data/raw_extracted/BTC/*_5m_*.parquet'))
print(f"BTC 5m files: {len(files)}")
# Test year 2024
files_2024 = [f for f in files if '2024' in f]
dfs = [pd.read_parquet(f) for f in files_2024]
df_2024 = pd.concat(dfs).sort_index()
print(f"BTC 2024 5m bars: {len(df_2024)} ({df_2024.index[0]} to {df_2024.index[-1]})")

# 80% IS / 20% OOS
split_idx = int(len(df_2024) * 0.8)
df_is = df_2024.iloc[:split_idx]
df_oos = df_2024.iloc[split_idx:]

engine = HighFrequencySMCEngine(initial_cash=1000.0, use_compound=True)

res_is = engine.run(df_is)
res_oos = engine.run(df_oos)

print("\n--- IS Metrics ---")
for k, v in res_is['metrics'].items():
    print(f"  {k}: {v}")

print("\n--- OOS Metrics ---")
for k, v in res_oos['metrics'].items():
    print(f"  {k}: {v}")

