"""
Qlib-inspired Micro-Structure Smart Money Concepts (SMC) Alpha Pipeline.

Implements causal feature extraction and Maker limit signal generation:
1. Causal Swing High/Low Detection:
   - Evaluates bar p = t - swing_len strictly within [p - swing_len, t].
   - Zero lookahead bias: confirms swing at bar t without future bars.
2. Qlib Micro-Structure Features:
   - orderflow_imbalance: (Close - Low) / (High - Low + 1e-9)
   - vol_spike: Volume / SMA(Volume, 20)
   - ob_level: 50% equilibrium of validated Order Block candle
   - fvg_level: midpoint of Fair Value Gap
   - atr_ratio: ATR(14) / ATR(100) for volatility regime filtering [0.55, 1.80]
   - Multi-scale EMA trend alignment (EMA20 > EMA50)
3. Signal & Maker Execution Generation:
   - score_long, score_short bounded in [0, 100]
   - signal: 1 (LONG), -1 (SHORT), 0 (NEUTRAL)
   - limit_entry_px: Maker limit price in discount/premium zone
   - Dirichlet Credal uncertainty filtering (u <= 0.35, non-conflicting)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from src.credal_engine import CredalSMCEngine, CredalSignal
except ImportError:
    try:
        from credal_engine import CredalSMCEngine, CredalSignal
    except ImportError:
        CredalSMCEngine = None
        CredalSignal = None


@dataclass(frozen=True)
class QlibSMCConfig:
    """Configuration parameters for Qlib SMC Alpha Feature Extraction and Signals."""

    swing_len: int = 5
    vol_sma_period: int = 20
    vol_spike_threshold: float = 1.2
    atr_fast_period: int = 14
    atr_slow_period: int = 100
    atr_ratio_min: float = 0.55
    atr_ratio_max: float = 1.80
    fast_ema_period: int = 20
    slow_ema_period: int = 50
    fvg_min_atr_mult: float = 0.10
    min_score_threshold: float = 60.0
    use_credal_filter: bool = True
    credal_u_max: float = 0.35
    credal_delta: float = 0.15


def _extract_ohlcv(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Extract open, high, low, close, volume arrays with case-insensitive column matching."""
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close", "volume"]
    for req in required:
        if req not in cols:
            raise ValueError(f"DataFrame must contain '{req}' column (case-insensitive).")

    open_ = df[cols["open"]].to_numpy(dtype=float)
    high = df[cols["high"]].to_numpy(dtype=float)
    low = df[cols["low"]].to_numpy(dtype=float)
    close = df[cols["close"]].to_numpy(dtype=float)
    volume = df[cols["volume"]].to_numpy(dtype=float)
    return open_, high, low, close, volume, df.index


def compute_wilder_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Compute strictly causal Wilder's Average True Range (RMA) using exponential smoothing."""
    n = len(high)
    if n == 0:
        return np.array([], dtype=float)

    tr = np.zeros(n, dtype=float)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    # Vectorized Wilder RMA via ewm with alpha = 1.0 / period
    atr_series = pd.Series(tr).ewm(alpha=1.0 / float(period), adjust=False).mean()
    return atr_series.to_numpy(dtype=float)


def compute_causal_swings(
    df: pd.DataFrame,
    swing_len: int = 5,
) -> pd.DataFrame:
    """Causal Swing High/Low Detection.

    Strict Causality Invariant:
    At bar t, a swing high or low is confirmed ONLY for pivot bar p = t - swing_len
    over the window [p - swing_len, t] = [t - 2 * swing_len, t].
    No information from bars > t is ever accessed.
    """
    _, high, low, close, _, index = _extract_ohlcv(df)
    n = len(df)

    is_sh_confirmed = np.zeros(n, dtype=bool)
    is_sl_confirmed = np.zeros(n, dtype=bool)
    sh_px = np.full(n, np.nan, dtype=float)
    sl_px = np.full(n, np.nan, dtype=float)

    if n > 2 * swing_len:
        w = 2 * swing_len + 1
        s_high = pd.Series(high)
        s_low = pd.Series(low)

        rolled_max = s_high.rolling(w, min_periods=w).max().to_numpy()
        rolled_min = s_low.rolling(w, min_periods=w).min().to_numpy()

        cand_h = (s_high.shift(swing_len).to_numpy() == rolled_max) & ~np.isnan(rolled_max)
        cand_l = (s_low.shift(swing_len).to_numpy() == rolled_min) & ~np.isnan(rolled_min)

        for t in np.where(cand_h)[0]:
            if t >= 2 * swing_len:
                p = t - swing_len
                # Strictly higher than preceding bars in window to break plateaus
                if np.all(high[p] > high[t - 2 * swing_len : p]):
                    is_sh_confirmed[t] = True
                    sh_px[t] = high[p]

        for t in np.where(cand_l)[0]:
            if t >= 2 * swing_len:
                p = t - swing_len
                # Strictly lower than preceding bars in window to break plateaus
                if np.all(low[p] < low[t - 2 * swing_len : p]):
                    is_sl_confirmed[t] = True
                    sl_px[t] = low[p]

    last_swing_high = pd.Series(sh_px, index=index).ffill().to_numpy(dtype=float)
    last_swing_low = pd.Series(sl_px, index=index).ffill().to_numpy(dtype=float)

    # Strictly causal liquidity sweeps against prior confirmed levels
    prior_sh = pd.Series(last_swing_high, index=index).shift(1).to_numpy()
    prior_sl = pd.Series(last_swing_low, index=index).shift(1).to_numpy()

    bsl_sweep = ~np.isnan(prior_sh) & (high > prior_sh) & (close < prior_sh)
    ssl_sweep = ~np.isnan(prior_sl) & (low < prior_sl) & (close > prior_sl)

    return pd.DataFrame(
        {
            "is_swing_high_confirmed": is_sh_confirmed,
            "is_swing_low_confirmed": is_sl_confirmed,
            "swing_high": sh_px,
            "swing_low": sl_px,
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low,
            "bsl_sweep": bsl_sweep,
            "ssl_sweep": ssl_sweep,
        },
        index=index,
    )


def compute_orderflow_imbalance(df: pd.DataFrame) -> pd.Series:
    """Qlib Micro-Structure Order Flow Imbalance Proxy: (Close - Low) / (High - Low + 1e-9)."""
    _, high, low, close, _, index = _extract_ohlcv(df)
    denom = np.maximum(high - low, 1e-9)
    ofi = np.clip((close - low) / denom, 0.0, 1.0)
    return pd.Series(ofi, index=index, name="orderflow_imbalance")


def compute_vol_spike(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Qlib Volume Spike Factor: Volume / SMA(Volume, 20)."""
    _, _, _, _, volume, index = _extract_ohlcv(df)
    vol_sma = pd.Series(volume, index=index).rolling(period, min_periods=1).mean().to_numpy()
    vol_spike = volume / np.maximum(vol_sma, 1e-9)
    return pd.Series(vol_spike, index=index, name="vol_spike")


def compute_atr_ratio(
    df: pd.DataFrame,
    fast_period: int = 14,
    slow_period: int = 100,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ATR(14), ATR(100), and volatility regime ratio ATR(14) / ATR(100)."""
    _, high, low, close, _, index = _extract_ohlcv(df)
    atr14 = compute_wilder_atr(high, low, close, period=fast_period)
    atr100 = compute_wilder_atr(high, low, close, period=slow_period)
    ratio = atr14 / np.maximum(atr100, 1e-9)

    s_atr14 = pd.Series(atr14, index=index, name="atr14")
    s_atr100 = pd.Series(atr100, index=index, name="atr100")
    s_ratio = pd.Series(ratio, index=index, name="atr_ratio")
    return s_atr14, s_atr100, s_ratio


def compute_ob_levels(
    df: pd.DataFrame,
    vol_spike_threshold: float = 1.2,
    atr_fast_period: int = 14,
) -> pd.DataFrame:
    """Detect validated Order Blocks and compute 50% equilibrium level (Mean Threshold).

    - Bullish OB: Last down candle (t-1) prior to strong bullish displacement at bar t.
      Displacement: Close[t] > High[t-1], Close[t] > Open[t], with volume spike or body displacement.
      50% Equilibrium: (High[t-1] + Low[t-1]) / 2.0.
    - Bearish OB: Last up candle (t-1) prior to strong bearish displacement at bar t.
      Displacement: Close[t] < Low[t-1], Close[t] < Open[t], with volume spike or body displacement.
      50% Equilibrium: (High[t-1] + Low[t-1]) / 2.0.
    """
    open_, high, low, close, volume, index = _extract_ohlcv(df)
    n = len(df)

    vol_sma = pd.Series(volume, index=index).rolling(20, min_periods=1).mean().to_numpy()
    vol_spike = volume / np.maximum(vol_sma, 1e-9)
    atr = compute_wilder_atr(high, low, close, period=atr_fast_period)

    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)
    bull_ob_level = np.full(n, np.nan, dtype=float)
    bear_ob_level = np.full(n, np.nan, dtype=float)
    ob_level = np.full(n, np.nan, dtype=float)

    if n > 1:
        prev_open = open_[:-1]
        prev_close = close[:-1]
        prev_high = high[:-1]
        prev_low = low[:-1]

        curr_open = open_[1:]
        curr_close = close[1:]
        curr_spike = vol_spike[1:]
        curr_atr = atr[1:]

        # Bullish displacement
        is_down_prev = prev_close <= prev_open
        disp_bull = (
            (curr_close > prev_high)
            & (curr_close > curr_open)
            & ((curr_spike >= vol_spike_threshold) | ((curr_close - curr_open) >= 0.5 * curr_atr))
        )
        bull_mask = is_down_prev & disp_bull
        bull_ob[1:][bull_mask] = True
        bull_ob_level[1:][bull_mask] = (prev_high[bull_mask] + prev_low[bull_mask]) / 2.0

        # Bearish displacement
        is_up_prev = prev_close >= prev_open
        disp_bear = (
            (curr_close < prev_low)
            & (curr_close < curr_open)
            & ((curr_spike >= vol_spike_threshold) | ((curr_open - curr_close) >= 0.5 * curr_atr))
        )
        bear_mask = is_up_prev & disp_bear
        bear_ob[1:][bear_mask] = True
        bear_ob_level[1:][bear_mask] = (prev_high[bear_mask] + prev_low[bear_mask]) / 2.0

        # Combined raw OB level
        ob_level[1:][bull_mask] = bull_ob_level[1:][bull_mask]
        ob_level[1:][bear_mask] = bear_ob_level[1:][bear_mask]

    # Forward fill levels causally
    s_ob_level = pd.Series(ob_level, index=index).ffill()
    s_bull_level = pd.Series(bull_ob_level, index=index).ffill()
    s_bear_level = pd.Series(bear_ob_level, index=index).ffill()

    return pd.DataFrame(
        {
            "bull_ob": bull_ob,
            "bear_ob": bear_ob,
            "ob_level": s_ob_level.to_numpy(),
            "bull_ob_level": s_bull_level.to_numpy(),
            "bear_ob_level": s_bear_level.to_numpy(),
        },
        index=index,
    )


def compute_fvg_levels(
    df: pd.DataFrame,
    min_atr_mult: float = 0.10,
    atr_fast_period: int = 14,
) -> pd.DataFrame:
    """Detect Fair Value Gaps (FVG) and compute their midpoint.

    - Bullish FVG: Low[t] > High[t-2] with gap >= min_atr_mult * ATR(14).
      Midpoint: (Low[t] + High[t-2]) / 2.0.
    - Bearish FVG: High[t] < Low[t-2] with gap >= min_atr_mult * ATR(14).
      Midpoint: (High[t] + Low[t-2]) / 2.0.
    """
    _, high, low, close, _, index = _extract_ohlcv(df)
    n = len(df)
    atr = compute_wilder_atr(high, low, close, period=atr_fast_period)

    bull_fvg = np.zeros(n, dtype=bool)
    bear_fvg = np.zeros(n, dtype=bool)
    bull_fvg_level = np.full(n, np.nan, dtype=float)
    bear_fvg_level = np.full(n, np.nan, dtype=float)
    fvg_level = np.full(n, np.nan, dtype=float)

    if n > 2:
        high_lag2 = high[:-2]
        low_lag2 = low[:-2]
        curr_high = high[2:]
        curr_low = low[2:]
        curr_atr = atr[2:]

        min_gap = min_atr_mult * curr_atr
        b_fvg_mask = (curr_low > high_lag2) & ((curr_low - high_lag2) >= min_gap)
        bull_fvg[2:][b_fvg_mask] = True
        bull_fvg_level[2:][b_fvg_mask] = (curr_low[b_fvg_mask] + high_lag2[b_fvg_mask]) / 2.0

        br_fvg_mask = (curr_high < low_lag2) & ((low_lag2 - curr_high) >= min_gap)
        bear_fvg[2:][br_fvg_mask] = True
        bear_fvg_level[2:][br_fvg_mask] = (curr_high[br_fvg_mask] + low_lag2[br_fvg_mask]) / 2.0

        fvg_level[2:][b_fvg_mask] = bull_fvg_level[2:][b_fvg_mask]
        fvg_level[2:][br_fvg_mask] = bear_fvg_level[2:][br_fvg_mask]

    s_fvg_level = pd.Series(fvg_level, index=index).ffill()
    s_bull_fvg = pd.Series(bull_fvg_level, index=index).ffill()
    s_bear_fvg = pd.Series(bear_fvg_level, index=index).ffill()

    return pd.DataFrame(
        {
            "bull_fvg": bull_fvg,
            "bear_fvg": bear_fvg,
            "fvg_level": s_fvg_level.to_numpy(),
            "bull_fvg_level": s_bull_fvg.to_numpy(),
            "bear_fvg_level": s_bear_fvg.to_numpy(),
        },
        index=index,
    )


def compute_ema_trend(
    df: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
) -> pd.DataFrame:
    """Compute multi-scale EMA trend alignment: EMA20 > EMA50."""
    _, _, _, close, _, index = _extract_ohlcv(df)
    s_close = pd.Series(close, index=index)
    ema20 = s_close.ewm(span=fast_period, adjust=False).mean()
    ema50 = s_close.ewm(span=slow_period, adjust=False).mean()

    trend_up = ema20 > ema50
    trend_down = ema20 < ema50

    return pd.DataFrame(
        {
            "ema20": ema20.to_numpy(),
            "ema50": ema50.to_numpy(),
            "trend_up": trend_up.to_numpy(),
            "trend_down": trend_down.to_numpy(),
        },
        index=index,
    )


def extract_qlib_smc_features(
    df: pd.DataFrame,
    config: Optional[QlibSMCConfig] = None,
) -> pd.DataFrame:
    """Extract full suite of causal Qlib SMC alpha features.

    Returns DataFrame augmented with:
    - orderflow_imbalance
    - vol_spike
    - ob_level, bull_ob, bear_ob, bull_ob_level, bear_ob_level
    - fvg_level, bull_fvg, bear_fvg, bull_fvg_level, bear_fvg_level
    - atr14, atr100, atr_ratio, volatility_regime_ok
    - ema20, ema50, trend_up, trend_down
    - swing_high, swing_low, last_swing_high, last_swing_low, bsl_sweep, ssl_sweep
    - micro_displacement, vol_delta_proxy
    """
    if config is None:
        config = QlibSMCConfig()

    open_, high, low, close, volume, index = _extract_ohlcv(df)
    res = df.copy()

    # 1. Micro-Structure
    ofi = compute_orderflow_imbalance(res)
    v_spike = compute_vol_spike(res, period=config.vol_sma_period)
    res["orderflow_imbalance"] = ofi.to_numpy()
    res["vol_spike"] = v_spike.to_numpy()

    # Extra Qlib features
    range_hl = np.maximum(high - low, 1e-9)
    res["micro_displacement"] = np.clip((close - open_) / range_hl, -1.0, 1.0)
    res["vol_delta_proxy"] = volume * (2.0 * ofi.to_numpy() - 1.0)

    # 2. Causal Swings & Liquidity Sweeps
    swings_df = compute_causal_swings(res, swing_len=config.swing_len)
    for col in swings_df.columns:
        res[col] = swings_df[col].to_numpy()

    # 3. ATR & Volatility Regime
    atr14, atr100, atr_ratio = compute_atr_ratio(
        res,
        fast_period=config.atr_fast_period,
        slow_period=config.atr_slow_period,
    )
    res["atr14"] = atr14.to_numpy()
    res["atr100"] = atr100.to_numpy()
    res["atr_ratio"] = atr_ratio.to_numpy()
    res["volatility_regime_ok"] = (res["atr_ratio"] >= config.atr_ratio_min) & (
        res["atr_ratio"] <= config.atr_ratio_max
    )

    # 4. Multi-scale EMA Trend
    ema_df = compute_ema_trend(
        res,
        fast_period=config.fast_ema_period,
        slow_period=config.slow_ema_period,
    )
    for col in ema_df.columns:
        res[col] = ema_df[col].to_numpy()

    # 5. Order Blocks (50% Equilibrium)
    ob_df = compute_ob_levels(
        res,
        vol_spike_threshold=config.vol_spike_threshold,
        atr_fast_period=config.atr_fast_period,
    )
    for col in ob_df.columns:
        res[col] = ob_df[col].to_numpy()

    # 6. Fair Value Gaps (Midpoint)
    fvg_df = compute_fvg_levels(
        res,
        min_atr_mult=config.fvg_min_atr_mult,
        atr_fast_period=config.atr_fast_period,
    )
    for col in fvg_df.columns:
        res[col] = fvg_df[col].to_numpy()

    return res


def generate_qlib_smc_signals(
    df: pd.DataFrame,
    config: Optional[QlibSMCConfig] = None,
) -> pd.DataFrame:
    """Generate confluence scores in [0, 100], discrete signal (-1, 0, 1), and Maker limit_entry_px."""
    if config is None:
        config = QlibSMCConfig()

    # Ensure required features exist
    req_cols = [
        "orderflow_imbalance",
        "vol_spike",
        "ob_level",
        "fvg_level",
        "atr_ratio",
        "trend_up",
        "trend_down",
        "bsl_sweep",
        "ssl_sweep",
        "bull_ob",
        "bear_ob",
        "bull_fvg",
        "bear_fvg",
    ]
    if not all(col in df.columns for col in req_cols):
        feats = extract_qlib_smc_features(df, config=config)
    else:
        feats = df.copy()

    open_, high, low, close, _, index = _extract_ohlcv(feats)
    n = len(feats)

    # 1. Bullish Confluence Score Components (max 100)
    # Trend: up to 25 pts
    trend_up = feats["trend_up"].to_numpy(dtype=bool)
    trend_down = feats["trend_down"].to_numpy(dtype=bool)
    ema20 = feats["ema20"].to_numpy(dtype=float)
    score_long_trend = (trend_up.astype(float) * 15.0) + ((close > ema20).astype(float) * 10.0)

    # Liquidity Sweep: up to 25 pts (current or recent sweep within 5 bars)
    ssl_sweep = feats["ssl_sweep"].to_numpy(dtype=bool)
    recent_ssl = pd.Series(ssl_sweep, index=index).rolling(5, min_periods=1).max().to_numpy(dtype=bool)
    score_long_sweep = np.where(ssl_sweep, 25.0, np.where(recent_ssl, 15.0, 0.0))

    # Structural OB & FVG: up to 25 pts
    bull_ob = feats["bull_ob"].to_numpy(dtype=bool)
    recent_bull_ob = pd.Series(bull_ob, index=index).rolling(5, min_periods=1).max().to_numpy(dtype=bool)
    bull_fvg = feats["bull_fvg"].to_numpy(dtype=bool)
    score_long_struct = (bull_ob.astype(float) * 15.0) + (bull_fvg.astype(float) * 10.0)
    score_long_struct = np.where(score_long_struct == 0.0, recent_bull_ob.astype(float) * 10.0, score_long_struct)

    # Micro-structure & Volatility: up to 25 pts
    ofi = feats["orderflow_imbalance"].to_numpy(dtype=float)
    vol_spike = feats["vol_spike"].to_numpy(dtype=float)
    vol_regime_ok = feats["volatility_regime_ok"].to_numpy(dtype=bool)

    ofi_bull_pts = np.clip((ofi - 0.5) / 0.4, 0.0, 1.0) * 10.0
    vol_pts = np.where(vol_spike >= 1.5, 10.0, np.where(vol_spike >= config.vol_spike_threshold, 5.0, 0.0))
    vol_regime_pts = vol_regime_ok.astype(float) * 5.0
    score_long_micro = ofi_bull_pts + vol_pts + vol_regime_pts

    score_long = np.clip(score_long_trend + score_long_sweep + score_long_struct + score_long_micro, 0.0, 100.0)

    # 2. Bearish Confluence Score Components (max 100)
    score_short_trend = (trend_down.astype(float) * 15.0) + ((close < ema20).astype(float) * 10.0)
    bsl_sweep = feats["bsl_sweep"].to_numpy(dtype=bool)
    recent_bsl = pd.Series(bsl_sweep, index=index).rolling(5, min_periods=1).max().to_numpy(dtype=bool)
    score_short_sweep = np.where(bsl_sweep, 25.0, np.where(recent_bsl, 15.0, 0.0))

    bear_ob = feats["bear_ob"].to_numpy(dtype=bool)
    recent_bear_ob = pd.Series(bear_ob, index=index).rolling(5, min_periods=1).max().to_numpy(dtype=bool)
    bear_fvg = feats["bear_fvg"].to_numpy(dtype=bool)
    score_short_struct = (bear_ob.astype(float) * 15.0) + (bear_fvg.astype(float) * 10.0)
    score_short_struct = np.where(score_short_struct == 0.0, recent_bear_ob.astype(float) * 10.0, score_short_struct)

    ofi_bear_pts = np.clip((0.5 - ofi) / 0.4, 0.0, 1.0) * 10.0
    score_short_micro = ofi_bear_pts + vol_pts + vol_regime_pts

    score_short = np.clip(score_short_trend + score_short_sweep + score_short_struct + score_short_micro, 0.0, 100.0)

    # 3. Credal Dirichlet Uncertainty Filtering (anti-hallucination)
    in_discount = close < ((high + low) / 2.0)
    in_premium = close > ((high + low) / 2.0)

    active_ssl = ssl_sweep | recent_ssl
    active_bsl = bsl_sweep | recent_bsl
    active_bull_ob = bull_ob | recent_bull_ob
    active_bear_ob = bear_ob | recent_bear_ob

    e_bull = (
        (trend_up.astype(float) * 2.0)
        + (active_bull_ob.astype(float) * 2.0)
        + (active_ssl.astype(float) * 1.5)
        + (in_discount.astype(float) * 1.0)
    )
    e_bear = (
        (trend_down.astype(float) * 2.0)
        + (active_bear_ob.astype(float) * 2.0)
        + (active_bsl.astype(float) * 1.5)
        + (in_premium.astype(float) * 1.0)
    )
    e_neutral = np.zeros(n, dtype=float)

    # Dirichlet strength and uncertainty
    S = e_bull + e_bear + e_neutral + 3.0
    u = 3.0 / S
    b_bull = e_bull / S
    b_bear = e_bear / S

    high_uncertainty = u > config.credal_u_max
    conflicting = (e_bull + e_bear > 0.5) & (np.abs(b_bull - b_bear) < config.credal_delta)
    credal_abstain = high_uncertainty | conflicting

    feats["credal_uncertainty"] = u
    feats["should_abstain"] = credal_abstain

    # 4. Discrete Signal Triggering
    min_score = config.min_score_threshold
    long_catalyst = active_ssl | bull_ob | bull_fvg | (trend_up & (ofi >= 0.60))
    short_catalyst = active_bsl | bear_ob | bear_fvg | (trend_down & (ofi <= 0.40))

    long_trigger = (score_long >= min_score) & (score_long > score_short) & vol_regime_ok & long_catalyst
    short_trigger = (score_short >= min_score) & (score_short > score_long) & vol_regime_ok & short_catalyst

    if config.use_credal_filter:
        long_trigger = long_trigger & (~credal_abstain) & (b_bull > b_bear)
        short_trigger = short_trigger & (~credal_abstain) & (b_bear > b_bull)

    signal = np.zeros(n, dtype=int)
    signal[long_trigger] = 1
    signal[short_trigger] = -1

    # 5. Exact Limit Entry Price for Maker Execution
    # LONG limit: Discount zone at or below Close (resting bid)
    # SHORT limit: Premium zone at or above Close (resting ask)
    ob_level = feats["ob_level"].to_numpy(dtype=float)
    fvg_level = feats["fvg_level"].to_numpy(dtype=float)
    bull_ob_level = feats.get("bull_ob_level", feats["ob_level"]).to_numpy(dtype=float)
    bear_ob_level = feats.get("bear_ob_level", feats["ob_level"]).to_numpy(dtype=float)

    limit_entry_px = np.full(n, np.nan, dtype=float)

    for i in range(n):
        c = close[i]
        h = high[i]
        l = low[i]
        bar_rng = h - l
        
        if signal[i] == 1:
            # OTE (Optimal Trade Entry) Discount zone: 0.62 ~ 0.79 retracement of swing/candle
            ote_px = h - 0.62 * bar_rng if bar_rng > 0 else c
            target_px = np.nan
            if bull_ob[i] and not np.isnan(bull_ob_level[i]) and bull_ob_level[i] <= c:
                target_px = bull_ob_level[i]
            elif bull_fvg[i] and not np.isnan(fvg_level[i]) and fvg_level[i] <= c:
                target_px = fvg_level[i]
            elif not np.isnan(bull_ob_level[i]) and bull_ob_level[i] <= c:
                target_px = bull_ob_level[i]
            else:
                target_px = min(c, ote_px)

            limit_entry_px[i] = min(c, target_px if not np.isnan(target_px) else c)

        elif signal[i] == -1:
            # OTE Premium zone: 0.62 retracement upwards from low
            ote_px = l + 0.62 * bar_rng if bar_rng > 0 else c
            target_px = np.nan
            if bear_ob[i] and not np.isnan(bear_ob_level[i]) and bear_ob_level[i] >= c:
                target_px = bear_ob_level[i]
            elif bear_fvg[i] and not np.isnan(fvg_level[i]) and fvg_level[i] >= c:
                target_px = fvg_level[i]
            elif not np.isnan(bear_ob_level[i]) and bear_ob_level[i] >= c:
                target_px = bear_ob_level[i]
            else:
                target_px = max(c, ote_px)

            limit_entry_px[i] = max(c, target_px if not np.isnan(target_px) else c)
    feats["score_long"] = score_long
    feats["score_short"] = score_short
    feats["signal"] = signal
    feats["limit_entry_px"] = limit_entry_px

    return feats


def compute_qlib_smc_alpha(
    df: pd.DataFrame,
    config: Optional[QlibSMCConfig] = None,
) -> pd.DataFrame:
    """Full-pipeline functional entry point: extracts features and generates alpha signals."""
    feats = extract_qlib_smc_features(df, config=config)
    return generate_qlib_smc_signals(feats, config=config)


class QlibSMCAlphaPipeline:
    """Object-oriented Alpha Feature Extraction and Signal Generation Engine.

    Parameters
    ----------
    config : Optional[QlibSMCConfig]
        Config dataclass containing swing lengths, moving averages, and thresholds.
    """

    def __init__(self, config: Optional[QlibSMCConfig] = None):
        self.config = config or QlibSMCConfig()

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all Qlib micro-structure and SMC features."""
        return extract_qlib_smc_features(df, config=self.config)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate confluence scores, discrete signal, and Maker limit prices."""
        return generate_qlib_smc_signals(df, config=self.config)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full end-to-end alpha extraction and signal generation."""
        return compute_qlib_smc_alpha(df, config=self.config)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.run(df)


__all__ = [
    "QlibSMCConfig",
    "QlibSMCAlphaPipeline",
    "compute_wilder_atr",
    "compute_causal_swings",
    "compute_orderflow_imbalance",
    "compute_vol_spike",
    "compute_atr_ratio",
    "compute_ob_levels",
    "compute_fvg_levels",
    "compute_ema_trend",
    "extract_qlib_smc_features",
    "generate_qlib_smc_signals",
    "compute_qlib_smc_alpha",
]
