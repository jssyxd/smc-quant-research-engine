# SMC Strategy Optimized Variants

基于 DeFiers-SMC Strategy v0.4.0 的三版优化策略。

原始回测区间实际为 **2024-01-01 ~ 2025-01-31**（约 1.08 年），非 README 宣传的 2017-2026。

## 版本说明

| 版本 | 目录 | 核心改动 | 适用场景 |
|------|------|----------|----------|
| **V1 内部因子优化** | `v1_internal_factors/` | Score≥70 + Factors≥4；做多更严格；Early BE 后加缓冲 | 最优先推荐，直接替换原策略即可 |
| **V2 波动率 Regime** | `v2_volatility_regime/` | 在 V1 基础上增加 ATR 相对波动率过滤 | 减少极端波动和死寂行情的噪音交易 |
| **V3 趋势强度确认** | `v3_trend_strength/` | 在 V1 基础上增加 EMA200 慢趋势过滤 | 进一步减少逆大趋势交易 |

建议测试顺序：V1 → V2 → V3（或 V1+V2 组合）。

## 共同改动点（相对原版）

### 1. 入场阈值提高
- `entry_threshold`: 65 → **70**
- `require_factors`: 3 → **4**

### 2. 做多更严格（因为历史 Long 表现明显优于 Short）
- 做多额外要求：Score ≥ **72** 或 Factors ≥ 4 且 HTF Bias 强确认
- 做空保持 Score ≥ 70 + Factors ≥ 4

### 3. Early Break-Even 缓冲
- 原版：浮盈达到 0.7R 后，止损直接移到入场价（保本）
- 优化：移到 **入场价 ± 0.15 ATR**（做多上移、做空下移），锁定少量利润，减少刚保本就被扫的情况
- 约 55% 的 STOP_LOSS 原本只亏手续费（–0.05%），加缓冲后这部分有望变成小盈利或更少被扫

### 4. V2 额外：波动率 Regime
- `atr_ratio = ATR(14) / ATR(100)`
- `atr_ratio > 1.8`（极端波动）→ 暂停开仓
- `atr_ratio < 0.55`（死寂）→ 暂停开仓
- 已有持仓正常管理，只过滤新开仓

### 5. V3 额外：慢趋势过滤
- 增加 EMA200
- 做多要求：Close > EMA200
- 做空要求：Close < EMA200
- 与原有 EMA50 HTF Bias 叠加，形成双层趋势确认

## 文件结构

```
smc_optimized/
├── README.md
├── v1_internal_factors/
│   ├── __init__.py
│   ├── indicators.py      # 指标 + 信号（含非对称做多/做空）
│   └── smc_strategy.py    # 回测引擎（含 BE 缓冲）
├── v2_volatility_regime/
│   ├── __init__.py
│   ├── indicators.py
│   └── smc_strategy.py
└── v3_trend_strength/
    ├── __init__.py
    ├── indicators.py
    └── smc_strategy.py
```

## 使用方式

与原版相同，把对应版本的 `indicators.py` + `smc_strategy.py` 替换到你的 `strategies/` 目录即可。

```python
from strategies.indicators import SMCConfig, compute_smc_indicators, get_signals
from strategies.smc_strategy import SMCStrategy, run_smc_backtest

# V1 默认已是优化参数
config = SMCConfig()  # entry_threshold=70, require_factors=4, ...

result = run_smc_backtest(df, config=config, initial_cash=10000.0)
print(result.summary())
```

## 参数速查（各版本默认）

| 参数 | 原版 | V1 | V2 | V3 |
|------|------|----|----|----|
| entry_threshold | 65 | 70 | 70 | 70 |
| require_factors | 3 | 4 | 4 | 4 |
| long_score_extra | - | 72 | 72 | 72 |
| early_be_buffer_atr | 0 | 0.15 | 0.15 | 0.15 |
| regime_filter | 弱 | 关 | **开** | 关 |
| regime_hi | 1.5 | - | 1.8 | - |
| regime_lo | 0.7 | - | 0.55 | - |
| use_ema200_filter | 否 | 否 | 否 | **是** |

## 注意事项

1. 这些改动基于 2024-01~2025-01 的回测观察，属于**样本内优化**，实盘前务必用更新数据验证。
2. 提高阈值会显著减少交易次数（大约减半），属于用频率换质量。
3. 1 分钟周期仍然不推荐，请优先用 1h / 15m。
4. 手续费、滑点、资金费率未完全模拟，实盘表现会打折。
