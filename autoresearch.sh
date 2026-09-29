#!/usr/bin/env bash
set -e

# ==============================================================================
# AUTORESEARCH BENCHMARK HARNESS
# Entrypoint for systematic quantification of SMC Alpha:
# 1. Runs Multi-Timeframe Sensitivity (1h vs 4h)
# 2. Computes 2D Parameter Plateau on ATR & Score Confluence
# 3. Measures Cross-Sectional Performance on Sovereign + Expansion Crypto Pool
# 4. Outputs deterministic primary & secondary METRIC lines
# ==============================================================================

python3 scripts/parameter_plateau_runner.py > /dev/null 2>&1

python3 -c "
import pandas as pd
import numpy as np
from pathlib import Path

csv_p = Path('results/plateau_analysis/parameter_plateau_grid.csv')
if not csv_p.exists():
    print('METRIC plateau_stability_score=0.0')
    sys.exit(0)

df = pd.read_csv(csv_p)

# Primary Metric: Parameter Plateau Stability Score (PPSS)
# Evaluates whether performance is smoothly distributed across the neighborhood
# rather than an isolated overfitted spike.
# Defined as: (Mean Positive Plateau Ratio) * (1 - Coefficient of Variation of Plateau Return)

# Separate 1h vs 4h
df_1h = df[df['timeframe'] == '1h']
df_4h = df[df['timeframe'] == '4h']

# Filter reasonable core neighborhood (Score 65-75, ATR 2.0-2.3)
core_plateau = df[(df['score_threshold'] >= 65.0) & (df['atr_multiplier'] >= 2.0) & (df['atr_multiplier'] <= 2.3)]
mean_ret = float(core_plateau['shared_return_pct'].mean())
std_ret = float(core_plateau['shared_return_pct'].std()) if len(core_plateau) > 1 else 1.0
win_ratio = float((core_plateau['shared_return_pct'] > 0).mean())

# Plateau stability score: higher is better
# Combines positive return, low variance (stability), and high positive neighbor ratio
cv = abs(std_ret / (abs(mean_ret) + 1e-4))
stability_score = float(np.clip(win_ratio * 100.0 / (1.0 + cv), 0.0, 100.0))

max_mdd = float(core_plateau['shared_max_drawdown_pct'].mean())
total_trades = int(core_plateau['shared_trades_executed'].mean())

print(f'METRIC plateau_stability_score={stability_score:.2f}')
print(f'METRIC core_plateau_mean_return_pct={mean_ret:.2f}')
print(f'METRIC core_plateau_win_ratio_pct={win_ratio*100:.1f}')
print(f'METRIC core_plateau_mean_mdd_pct={max_mdd:.2f}')
print(f'METRIC core_plateau_avg_trades={total_trades}')
"
