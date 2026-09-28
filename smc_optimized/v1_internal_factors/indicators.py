"""
SMC 核心指标模块 - V1 内部因子优化版

改动相对原版:
1. entry_threshold 默认 70，require_factors 默认 4
2. 做多更严格: Long 需要 Score >= long_entry_threshold (默认 72)
3. 信号生成支持非对称多空条件
4. Early BE 缓冲参数暴露到 Config

基于 DeFiers-SMC Strategy v0.4.0 Pine Script 翻译
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TrendState(Enum):
    UNKNOWN = 0
    BULLISH = 1
    BEARISH = -1


@dataclass
class SMCConfig:
    """SMC 策略配置 - V1 优化默认值"""
    # 枢轴检测
    swing_len: int = 10
    mss_disp_mult: float = 1.5

    # Order Blocks
    vol_mult: float = 1.5
    fvg_min_atr: float = 0.5
    ob_max_age: int = 60
    fvg_max_age: int = 80
    max_obs: int = 15
    max_fvgs: int = 20
    ob_lookback: int = 20

    # Liquidity
    liq_range_pct: float = 0.15
    max_liq_levels: int = 15
    sweep_recency_bars: int = 15

    # Confluence & Entry  ★ V1 核心改动
    entry_threshold: float = 70.0          # 原 65 → 70
    long_entry_threshold: float = 72.0     # 做多额外提高（原无此参数）
    require_factors: int = 4               # 原 3 → 4
    htf_bias_filter: bool = True

    # Risk
    risk_pct: float = 2.0
    sl_buffer_atr: float = 0.5
    rr_tp1: float = 1.5
    rr_tp2: float = 3.0
    tp1_qty_pct: float = 50.0
    use_trailing: bool = True
    use_early_be: bool = True
    early_be_rr: float = 0.7
    # ★ V1: Early BE 后不是精确保本，而是锁定一点利润
    early_be_buffer_atr: float = 0.15      # 原 0，移到 entry ± 0.15 ATR

    # Regime (V1 默认关闭，留给 V2)
    enable_regime: bool = False
    regime_hi: float = 1.8
    regime_lo: float = 0.55
    regime_adj_pct: float = 20.0

    # Trend strength (V1 默认关闭，留给 V3)
    use_ema200_filter: bool = False


def compute_smc_indicators(df: pd.DataFrame, config: Optional[SMCConfig] = None) -> pd.DataFrame:
    """向量化计算 SMC 指标"""
    config = config or SMCConfig()

    df = df.copy()
    if 'open' in df.columns:
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })

    n = len(df)
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    open_ = df['Open'].values
    volume = df['Volume'].values

    # 1. ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    df['ATR'] = atr

    vol_sma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    atr100 = pd.Series(atr).rolling(100, min_periods=50).mean().ffill().fillna(atr[0]).values
    df['ATR100'] = atr100
    df['ATR_Ratio'] = atr / np.maximum(atr100, 1e-9)

    # 2. 枢轴 (因果严格检测：在时间 t 仅确认 t - swing_len 处的极值，绝无未来函数)
    swing_len = config.swing_len
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    curr_hh = np.nan
    curr_ll = np.nan
    for t in range(2 * swing_len, n):
        p = t - swing_len
        if high[p] == np.max(high[p - swing_len : t + 1]):
            curr_hh = high[p]
        if low[p] == np.min(low[p - swing_len : t + 1]):
            curr_ll = low[p]
        hh[t] = curr_hh
        ll[t] = curr_ll
    df['HH'] = hh
    df['LL'] = ll

    # 3. 趋势 (EMA9/21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    trend_state = np.zeros(n, dtype=int)
    for i in range(1, n):
        if ema9[i] > ema21[i]:
            trend_state[i] = 1
        elif ema9[i] < ema21[i]:
            trend_state[i] = -1
        else:
            trend_state[i] = trend_state[i-1]
    df['TrendState'] = trend_state

    # 4. BOS
    bull_bos = (close > np.roll(hh, 1)) & (trend_state >= 0)
    bear_bos = (close < np.roll(ll, 1)) & (trend_state <= 0)
    bull_bos[0] = False
    bear_bos[0] = False
    df['BullBOS'] = bull_bos
    df['BearBOS'] = bear_bos

    # 5. HTF Bias (EMA50)
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    htf_bull_bias = close > ema50
    htf_bear_bias = close < ema50
    df['HTFBullBias'] = htf_bull_bias
    df['HTFBearBias'] = htf_bear_bias
    df['EMA50'] = ema50

    # EMA200 (为 V3 预留，V1 也算出来方便对比)
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
    df['EMA200'] = ema200
    df['AboveEMA200'] = close > ema200
    df['BelowEMA200'] = close < ema200

    # 6. Order Blocks (简化)
    is_bearish = close < open_
    vol_ratio = volume / np.maximum(vol_sma20, 1e-9)
    bull_ob_signal = np.concatenate([[False], is_bearish[1:] & (vol_ratio[1:] >= config.vol_mult)])
    is_bullish = close > open_
    bear_ob_signal = np.concatenate([[False], is_bullish[1:] & (vol_ratio[1:] >= config.vol_mult)])
    df['BullOBSignal'] = bull_ob_signal
    df['BearOBSignal'] = bear_ob_signal

    # 7. FVG
    bull_fvg = (low > np.roll(high, 2)) & ((low - np.roll(high, 2)) >= config.fvg_min_atr * atr)
    bull_fvg[0] = False
    bull_fvg[1] = False
    bear_fvg = (high < np.roll(low, 2)) & ((np.roll(low, 2) - high) >= config.fvg_min_atr * atr)
    bear_fvg[0] = False
    bear_fvg[1] = False
    df['BullFVG'] = bull_fvg
    df['BearFVG'] = bear_fvg

    # 8. Premium/Discount
    range_top = np.maximum.accumulate(np.where(np.isnan(hh), -np.inf, hh))
    range_bot = np.minimum.accumulate(np.where(np.isnan(ll), np.inf, ll))
    range_top = np.where(np.isinf(range_top), high, range_top)
    range_bot = np.where(np.isinf(range_bot), low, range_bot)
    equilibrium = (range_top + range_bot) / 2
    df['RangeTop'] = range_top
    df['RangeBot'] = range_bot
    df['Equilibrium'] = equilibrium
    in_discount = close <= equilibrium
    in_premium = close >= equilibrium
    df['InDiscount'] = in_discount
    df['InPremium'] = in_premium

    # 9. Liquidity Sweep
    bsl_swept = np.zeros(n, dtype=bool)
    ssl_swept = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(hh[i-1]) and high[i] > hh[i-1] and close[i] < hh[i-1]:
            bsl_swept[i] = True
        if not np.isnan(ll[i-1]) and low[i] < ll[i-1] and close[i] > ll[i-1]:
            ssl_swept[i] = True
    df['BSLSwept'] = bsl_swept
    df['SSLSwept'] = ssl_swept

    # 10. Smart Money Score
    f1_long = htf_bull_bias.astype(float)
    f1_short = htf_bear_bias.astype(float)
    f2_long = (trend_state == 1).astype(float)
    f2_short = (trend_state == -1).astype(float)
    f3_long = ssl_swept.astype(float)
    f3_short = bsl_swept.astype(float)
    f4_long = np.minimum(bull_ob_signal.astype(float) * 1.0 + bull_fvg.astype(float) * 0.7, 1.0)
    f4_short = np.minimum(bear_ob_signal.astype(float) * 1.0 + bear_fvg.astype(float) * 0.7, 1.0)
    f5_long = in_discount.astype(float)
    f5_short = in_premium.astype(float)

    score_long = 100 * (0.25 * f1_long + 0.20 * f2_long + 0.20 * f3_long + 0.20 * f4_long + 0.15 * f5_long)
    score_short = 100 * (0.25 * f1_short + 0.20 * f2_short + 0.20 * f3_short + 0.20 * f4_short + 0.15 * f5_short)
    df['ScoreLong'] = score_long
    df['ScoreShort'] = score_short

    factors_long = (
        (f1_long >= 1).astype(int) + (f2_long >= 1).astype(int) +
        (f3_long >= 1).astype(int) + (f4_long >= 0.5).astype(int) + (f5_long >= 1).astype(int)
    )
    factors_short = (
        (f1_short >= 1).astype(int) + (f2_short >= 1).astype(int) +
        (f3_short >= 1).astype(int) + (f4_short >= 0.5).astype(int) + (f5_short >= 1).astype(int)
    )
    df['FactorsLong'] = factors_long
    df['FactorsShort'] = factors_short

    return df


def get_signals(df: pd.DataFrame, config: Optional[SMCConfig] = None) -> pd.DataFrame:
    """
    生成交易信号 - V1 非对称多空

    Long:  Score >= long_entry_threshold (72) 且 Factors >= 4
    Short: Score >= entry_threshold (70) 且 Factors >= 4
    """
    config = config or SMCConfig()
    df = df.copy()

    long_cond = (
        (df['ScoreLong'] >= config.long_entry_threshold) &
        (df['FactorsLong'] >= config.require_factors)
    )
    short_cond = (
        (df['ScoreShort'] >= config.entry_threshold) &
        (df['FactorsShort'] >= config.require_factors)
    )

    # 可选: EMA200 过滤 (V1 默认关)
    if config.use_ema200_filter:
        long_cond = long_cond & df['AboveEMA200']
        short_cond = short_cond & df['BelowEMA200']

    # 可选: 波动率 Regime (V1 默认关)
    if config.enable_regime:
        vol_ok = (df['ATR_Ratio'] >= config.regime_lo) & (df['ATR_Ratio'] <= config.regime_hi)
        long_cond = long_cond & vol_ok
        short_cond = short_cond & vol_ok

    df['Signal'] = 'NEUTRAL'
    df.loc[long_cond, 'Signal'] = 'LONG'
    df.loc[short_cond, 'Signal'] = 'SHORT'
    return df


class SMCIndicators:
    def __init__(self, df: pd.DataFrame, config: Optional[SMCConfig] = None):
        self.df = df
        self.config = config or SMCConfig()

    def run(self) -> pd.DataFrame:
        return compute_smc_indicators(self.df, self.config)
