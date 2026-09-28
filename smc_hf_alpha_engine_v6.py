import sys, os, gc, math
import pandas as pd
import numpy as np

def compute_multi_tf_smc_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-Scale SMC Structure + Micro Reversal:
    - Higher Timeframe Context: EMA50 and EMA200 for market macro regime
    - Major Swings: swing_len=8 to establish true Liquidity levels (BSL / SSL)
    - Confluence Score: High selective threshold (Score >= 70, Factors >= 4)
    - Execution on 15m bars with dynamic risk sizing
    """
    df = df.copy()
    if 'open' in df.columns:
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

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
    df['ATR_Ratio'] = atr / np.maximum(atr100, 1e-9)

    # 2. Major Swings (swing_len = 8)
    swing_len = 8
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    for i in range(swing_len, n - swing_len):
        if high[i] == np.max(high[i - swing_len:i + swing_len + 1]):
            is_high[i] = True
        if low[i] == np.min(low[i - swing_len:i + swing_len + 1]):
            is_low[i] = True

    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    last_hh_idx = 0
    last_ll_idx = 0
    for i in range(n):
        if is_high[i]:
            hh[i] = high[i]
            last_hh_idx = i
        else:
            hh[i] = hh[last_hh_idx] if last_hh_idx > 0 else np.nan
        if is_low[i]:
            ll[i] = low[i]
            last_ll_idx = i
        else:
            ll[i] = ll[last_ll_idx] if last_ll_idx > 0 else np.nan
    df['HH'] = hh
    df['LL'] = ll

    # 3. Macro EMA
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
    df['EMA50'] = ema50
    df['EMA200'] = ema200

    # 4. Trend State
    ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    trend_state = np.where(ema9 > ema21, 1, -1)
    df['TrendState'] = trend_state

    # 5. Order Blocks (OB) & FVG
    is_bearish = close < open_
    vol_ratio = volume / np.maximum(vol_sma20, 1e-9)
    bull_ob_signal = np.concatenate([[False], is_bearish[:-1] & (vol_ratio[1:] >= 1.5)])
    is_bullish = close > open_
    bear_ob_signal = np.concatenate([[False], is_bullish[:-1] & (vol_ratio[1:] >= 1.5)])

    bull_fvg = (low > np.roll(high, 2)) & ((low - np.roll(high, 2)) >= 0.5 * atr)
    bear_fvg = (high < np.roll(low, 2)) & ((np.roll(low, 2) - high) >= 0.5 * atr)
    bull_fvg[0] = bull_fvg[1] = False
    bear_fvg[0] = bear_fvg[1] = False

    # 6. Premium / Discount
    range_top = np.maximum.accumulate(np.where(np.isnan(hh), -np.inf, hh))
    range_bot = np.minimum.accumulate(np.where(np.isnan(ll), np.inf, ll))
    range_top = np.where(np.isinf(range_top), high, range_top)
    range_bot = np.where(np.isinf(range_bot), low, range_bot)
    equilibrium = (range_top + range_bot) / 2.0
    in_discount = close <= equilibrium
    in_premium = close >= equilibrium

    # 7. Liquidity Sweep
    bsl_swept = np.zeros(n, dtype=bool)
    ssl_swept = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(hh[i-1]) and high[i] > hh[i-1] and close[i] < hh[i-1]:
            bsl_swept[i] = True
        if not np.isnan(ll[i-1]) and low[i] < ll[i-1] and close[i] > ll[i-1]:
            ssl_swept[i] = True

    # 8. Score
    f1_long = (close > ema50).astype(float)
    f2_long = (trend_state == 1).astype(float)
    f3_long = ssl_swept.astype(float)
    f4_long = np.minimum(bull_ob_signal.astype(float) * 1.0 + bull_fvg.astype(float) * 0.7, 1.0)
    f5_long = in_discount.astype(float)

    score_long = 100.0 * (0.25 * f1_long + 0.20 * f2_long + 0.20 * f3_long + 0.20 * f4_long + 0.15 * f5_long)
    factors_long = (f1_long >= 1).astype(int) + (f2_long >= 1).astype(int) + (f3_long >= 1).astype(int) + (f4_long >= 0.5).astype(int) + (f5_long >= 1).astype(int)

    f1_short = (close < ema50).astype(float)
    f2_short = (trend_state == -1).astype(float)
    f3_short = bsl_swept.astype(float)
    f4_short = np.minimum(bear_ob_signal.astype(float) * 1.0 + bear_fvg.astype(float) * 0.7, 1.0)
    f5_short = in_premium.astype(float)

    score_short = 100.0 * (0.25 * f1_short + 0.20 * f2_short + 0.20 * f3_short + 0.20 * f4_short + 0.15 * f5_short)
    factors_short = (f1_short >= 1).astype(int) + (f2_short >= 1).astype(int) + (f3_short >= 1).astype(int) + (f4_short >= 0.5).astype(int) + (f5_short >= 1).astype(int)

    df['ScoreLong'] = score_long
    df['ScoreShort'] = score_short
    df['FactorsLong'] = factors_long
    df['FactorsShort'] = factors_short

    # High Confluence Filter
    vol_ok = (df['ATR_Ratio'] >= 0.55) & (df['ATR_Ratio'] <= 1.80)
    long_sig = (score_long >= 70.0) & (factors_long >= 4) & vol_ok
    short_sig = (score_short >= 70.0) & (factors_short >= 4) & vol_ok

    df['Signal'] = 'NEUTRAL'
    df.loc[long_sig, 'Signal'] = 'LONG'
    df.loc[short_sig, 'Signal'] = 'SHORT'
    return df
