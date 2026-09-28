import sys, os, glob, pandas as pd
from smc_hf_alpha_engine_v5 import HighFrequencySMCEngineV5

files = sorted(glob.glob('data/raw_extracted/BTC/*_15m_*.parquet'))
files_2024 = [f for f in files if '2024' in f]
dfs = [pd.read_parquet(f) for f in files_2024]
df_2024 = pd.concat(dfs).sort_index()

split_idx = int(len(df_2024) * 0.8)
df_is = df_2024.iloc[:split_idx]
df_oos = df_2024.iloc[split_idx:]

engine = HighFrequencySMCEngineV5(initial_cash=1000.0, use_compound=True)

# Test swing_len=5, entry_thresh=60
res_is = engine.run(df_is, swing_len=5, entry_thresh=60.0)
res_oos = engine.run(df_oos, swing_len=5, entry_thresh=60.0)

print("\n--- IS Metrics (V5 BTC 2024 15m) ---")
for k, v in res_is['metrics'].items():
    print(f"  {k}: {v}")

print("\n--- OOS Metrics (V5 BTC 2024 15m) ---")
for k, v in res_oos['metrics'].items():
    print(f"  {k}: {v}")
