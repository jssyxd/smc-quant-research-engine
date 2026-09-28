import sys, os, gc, math
import pandas as pd
import numpy as np

def compute_smc_v1_factors(df: pd.DataFrame, swing_len: int = 6, entry_thresh: float = 65.0) -> pd.DataFrame:
    """
    SMC High-Frequency Signal Generator preserving original V1/V2 alpha core:
    - Reduced swing_len from 10 to 6
    - Slightly lowered entry_threshold from 70 to 65
    - Preserves exact Confluence weighting, Early BE buffer, and RR targets
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

    # 2. Pivots
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

    # 3. Trend State (EMA9 / EMA21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    trend_state = np.where(ema9 > ema21, 1, -1)
    df['TrendState'] = trend_state

    # 4. HTF Bias (EMA50)
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    htf_bull_bias = close > ema50
    htf_bear_bias = close < ema50
    df['HTFBullBias'] = htf_bull_bias
    df['HTFBearBias'] = htf_bear_bias

    # 5. Order Blocks (OB) & FVG
    is_bearish = close < open_
    vol_ratio = volume / np.maximum(vol_sma20, 1e-9)
    bull_ob_signal = np.concatenate([[False], is_bearish[:-1] & (vol_ratio[1:] >= 1.3)])
    is_bullish = close > open_
    bear_ob_signal = np.concatenate([[False], is_bullish[:-1] & (vol_ratio[1:] >= 1.3)])

    bull_fvg = (low > np.roll(high, 2)) & ((low - np.roll(high, 2)) >= 0.4 * atr)
    bear_fvg = (high < np.roll(low, 2)) & ((np.roll(low, 2) - high) >= 0.4 * atr)
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
    f1_long = htf_bull_bias.astype(float)
    f2_long = (trend_state == 1).astype(float)
    f3_long = ssl_swept.astype(float)
    f4_long = np.minimum(bull_ob_signal.astype(float) * 1.0 + bull_fvg.astype(float) * 0.7, 1.0)
    f5_long = in_discount.astype(float)

    score_long = 100.0 * (0.25 * f1_long + 0.20 * f2_long + 0.20 * f3_long + 0.20 * f4_long + 0.15 * f5_long)
    factors_long = (f1_long >= 1).astype(int) + (f2_long >= 1).astype(int) + (f3_long >= 1).astype(int) + (f4_long >= 0.5).astype(int) + (f5_long >= 1).astype(int)

    f1_short = htf_bear_bias.astype(float)
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

    # Trigger with volatility regime filter (0.50 ~ 2.0)
    vol_ok = (df['ATR_Ratio'] >= 0.50) & (df['ATR_Ratio'] <= 2.0)
    long_sig = (score_long >= entry_thresh) & (factors_long >= 3) & vol_ok
    short_sig = (score_short >= entry_thresh) & (factors_short >= 3) & vol_ok

    df['Signal'] = 'NEUTRAL'
    df.loc[long_sig, 'Signal'] = 'LONG'
    df.loc[short_sig, 'Signal'] = 'SHORT'
    return df

class HighFrequencySMCEngineV5:
    def __init__(
        self,
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,   # 0.02%
        taker_fee: float = 0.0005,   # 0.05%
        slippage: float = 0.0005,    # 0.05%
        risk_pct: float = 2.0,       # 2% risk
        use_compound: bool = True,   # Compounding
        rr_tp1: float = 1.5,
        rr_tp2: float = 3.0,
        early_be_rr: float = 0.7,
        early_be_buffer_atr: float = 0.15
    ):
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.risk_pct = risk_pct
        self.use_compound = use_compound
        self.rr_tp1 = rr_tp1
        self.rr_tp2 = rr_tp2
        self.early_be_rr = early_be_rr
        self.early_be_buffer_atr = early_be_buffer_atr

    def run(self, df: pd.DataFrame, swing_len: int = 6, entry_thresh: float = 65.0) -> dict:
        if len(df) < 50:
            return {"trades": [], "equity_curve": [self.initial_cash], "metrics": {}}

        data = compute_smc_v1_factors(df, swing_len=swing_len, entry_thresh=entry_thresh)

        equity = self.initial_cash
        cash = self.initial_cash
        equity_curve = [equity]
        trades = []

        position = 0 # 0=FLAT, 1=LONG, -1=SHORT
        pos_size = 0.0
        entry_price = 0.0
        entry_time = None
        planned_sl = 0.0
        tp1 = 0.0
        tp2 = 0.0
        tp1_hit = False
        early_be_triggered = False

        total_fees = 0.0
        total_slippage = 0.0

        close_arr = data['Close'].values
        high_arr = data['High'].values
        low_arr = data['Low'].values
        atr_arr = data['ATR'].values
        signal_arr = data['Signal'].values
        hh_arr = data['HH'].values
        ll_arr = data['LL'].values
        score_long_arr = data['ScoreLong'].values
        score_short_arr = data['ScoreShort'].values
        factors_long_arr = data['FactorsLong'].values
        factors_short_arr = data['FactorsShort'].values
        timestamps = data.index
        n = len(data)

        for idx in range(n):
            close = close_arr[idx]
            high = high_arr[idx]
            low = low_arr[idx]
            atr = atr_arr[idx]
            timestamp = timestamps[idx]

            # 1. Manage position
            if position != 0:
                # TP1 Check (Limit order -> Maker fee, 0 slippage)
                if not tp1_hit:
                    if (position == 1 and high >= tp1) or (position == -1 and low <= tp1):
                        exit_px = tp1
                        part_size = pos_size * 0.5
                        fee = part_size * exit_px * self.maker_fee
                        total_fees += fee
                        if position == 1:
                            pnl_part = part_size * (exit_px - entry_price) - fee
                            planned_sl = entry_price + self.early_be_buffer_atr * atr
                        else:
                            pnl_part = part_size * (entry_price - exit_px) - fee
                            planned_sl = entry_price - self.early_be_buffer_atr * atr
                        cash += pnl_part
                        equity += pnl_part
                        pos_size *= 0.5
                        tp1_hit = True
                        early_be_triggered = True

                # TP2 Check
                closed = False
                if (position == 1 and high >= tp2) or (position == -1 and low <= tp2):
                    exit_px = tp2
                    fee = pos_size * exit_px * self.maker_fee
                    total_fees += fee
                    pnl = pos_size * (exit_px - entry_price if position == 1 else entry_price - exit_px) - fee
                    cash += pnl
                    equity += pnl
                    trades.append({
                        "entry_time": str(entry_time),
                        "exit_time": str(timestamp),
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_px),
                        "size": float(pos_size),
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                        "exit_reason": "TP2"
                    })
                    position = 0
                    closed = True

                # Stop Loss Check (Market order -> Taker fee + Slippage)
                if not closed:
                    if (position == 1 and low <= planned_sl) or (position == -1 and high >= planned_sl):
                        if position == 1:
                            exit_px = planned_sl * (1.0 - self.slippage)
                            slip = pos_size * (planned_sl - exit_px)
                            pnl = pos_size * (exit_px - entry_price)
                        else:
                            exit_px = planned_sl * (1.0 + self.slippage)
                            slip = pos_size * (exit_px - planned_sl)
                            pnl = pos_size * (entry_price - exit_px)

                        fee = pos_size * exit_px * self.taker_fee
                        total_fees += fee
                        total_slippage += slip
                        pnl -= fee
                        cash += pnl
                        equity += pnl
                        trades.append({
                            "entry_time": str(entry_time),
                            "exit_time": str(timestamp),
                            "direction": "LONG" if position == 1 else "SHORT",
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_px),
                            "size": float(pos_size),
                            "pnl": float(pnl),
                            "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                            "exit_reason": "STOP_LOSS"
                        })
                        position = 0
                        closed = True

                # Early BE Check before TP1
                if not closed and not early_be_triggered:
                    stop_dist = abs(entry_price - planned_sl)
                    if position == 1 and (high - entry_price) >= self.early_be_rr * stop_dist:
                        planned_sl = entry_price + self.early_be_buffer_atr * atr
                        early_be_triggered = True
                    elif position == -1 and (entry_price - low) >= self.early_be_rr * stop_dist:
                        planned_sl = entry_price - self.early_be_buffer_atr * atr
                        early_be_triggered = True

            # 2. Check Signal if flat
            if position == 0:
                sig = signal_arr[idx]
                if sig in ('LONG', 'SHORT'):
                    direction = 1 if sig == 'LONG' else -1
                    hh = hh_arr[idx]
                    ll = ll_arr[idx]

                    if direction == 1:
                        sl_calc = (ll if not np.isnan(ll) else close) - 0.5 * atr
                        stop_dist = close - sl_calc
                        tp1_calc = close + self.rr_tp1 * stop_dist
                        tp2_calc = close + self.rr_tp2 * stop_dist
                    else:
                        sl_calc = (hh if not np.isnan(hh) else close) + 0.5 * atr
                        stop_dist = sl_calc - close
                        tp1_calc = close - self.rr_tp1 * stop_dist
                        tp2_calc = close - self.rr_tp2 * stop_dist

                    if stop_dist > 0 and stop_dist <= close * 0.05:
                        base_capital = max(equity, 10.0) if self.use_compound else self.initial_cash
                        risk_amt = base_capital * (self.risk_pct / 100.0)
                        size = risk_amt / stop_dist

                        # Max leverage 4x cap
                        max_size = (base_capital * 4.0) / close
                        size = min(size, max_size)

                        if direction == 1:
                            exec_px = close * (1.0 + self.slippage)
                        else:
                            exec_px = close * (1.0 - self.slippage)

                        slip_cost = size * abs(exec_px - close)
                        fee_cost = size * exec_px * self.taker_fee
                        total_slippage += slip_cost
                        total_fees += fee_cost

                        cash -= fee_cost
                        equity -= fee_cost

                        position = direction
                        pos_size = size
                        entry_price = exec_px
                        entry_time = timestamp
                        planned_sl = sl_calc
                        tp1 = tp1_calc
                        tp2 = tp2_calc
                        tp1_hit = False
                        early_be_triggered = False

            # Mark to market equity
            if position != 0:
                if position == 1:
                    unrealized = pos_size * (close - entry_price) - pos_size * close * self.taker_fee
                else:
                    unrealized = pos_size * (entry_price - close) - pos_size * close * self.taker_fee
                equity_curve.append(cash + unrealized)
            else:
                equity_curve.append(cash)

        # Close at end if open
        if position != 0:
            exit_px = close_arr[-1]
            fee = pos_size * exit_px * self.taker_fee
            total_fees += fee
            pnl = pos_size * (exit_px - entry_price if position == 1 else entry_price - exit_px) - fee
            cash += pnl
            equity += pnl
            trades.append({
                "entry_time": str(entry_time),
                "exit_time": str(timestamps[-1]),
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": float(entry_price),
                "exit_price": float(exit_px),
                "size": float(pos_size),
                "pnl": float(pnl),
                "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                "exit_reason": "END_OF_DATA"
            })
            equity_curve[-1] = cash

        # Metrics
        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]
        win_rate = (len(winning) / len(trades) * 100.0) if trades else 0.0

        total_win = sum(t["pnl"] for t in winning)
        total_loss = abs(sum(t["pnl"] for t in losing))
        profit_factor = (total_win / total_loss) if total_loss > 0 else (99.0 if total_win > 0 else 0.0)

        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / np.maximum(peak, 1e-9)
        max_dd = float(np.max(dd) * 100.0) if len(dd) > 0 else 0.0

        rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-9)
        std_ret = float(np.std(rets))
        ann_factor = np.sqrt(252 * 24 * 4) # 15m
        sharpe = float(np.mean(rets) / std_ret * ann_factor) if std_ret > 1e-9 else 0.0

        total_pnl = cash - self.initial_cash
        ret_pct = (total_pnl / self.initial_cash) * 100.0

        days_span = max((timestamps[-1] - timestamps[0]).total_seconds() / 86400.0, 1.0)
        daily_trades = len(trades) / days_span

        metrics = {
            "initial_cash": float(self.initial_cash),
            "final_cash": float(cash),
            "total_return_pct": float(ret_pct),
            "total_trades": len(trades),
            "daily_trades": float(daily_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_dd),
            "sharpe_ratio": float(sharpe),
            "total_pnl": float(total_pnl),
            "total_fees": float(total_fees),
            "total_slippage_cost": float(total_slippage)
        }

        return {
            "trades": trades,
            "equity_curve": [float(x) for x in equity_curve],
            "metrics": metrics
        }
