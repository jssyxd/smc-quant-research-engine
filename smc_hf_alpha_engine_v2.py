import sys, os, gc, math
import pandas as pd
import numpy as np

def compute_smc_hf_alpha(df: pd.DataFrame, htf_filter: bool = True) -> pd.DataFrame:
    """
    High-Frequency SMC Alpha Engine (Qlib + RD-Agent Quant Architecture):
    - Confluence of Micro Structure (Order Blocks + FVG + Liquidity Sweeps)
    - Macro Multi-Timeframe Alignment (EMA 100/200 Trend Bias)
    - Volatility Regime Filter (avoid dead zones and runaway slippage)
    - Asymmetric Risk-Reward (TP1 @ 1.5R, TP2 @ 3.0R with Breakeven Buffering)
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values
    n = len(df)

    # 1. True Range & ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]

    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    df['atr'] = atr

    # 2. HTF Trend Filters (EMA 50 and EMA 200 on 5m)
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
    df['ema50'] = ema50
    df['ema200'] = ema200
    bull_trend = (close > ema50) & (ema50 > ema200)
    bear_trend = (close < ema50) & (ema50 < ema200)

    # 3. Swing High / Low (Pivot detection, len=8)
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
    df['hh'] = hh
    df['ll'] = ll

    # 4. Liquidity Sweep (BSL / SSL)
    bsl_swept = np.zeros(n, dtype=bool)
    ssl_swept = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(hh[i-1]) and high[i] > hh[i-1] and close[i] < hh[i-1]:
            bsl_swept[i] = True
        if not np.isnan(ll[i-1]) and low[i] < ll[i-1] and close[i] > ll[i-1]:
            ssl_swept[i] = True
    df['bsl_swept'] = bsl_swept
    df['ssl_swept'] = ssl_swept

    # 5. Order Blocks (OB) with Volume Spike
    vol_sma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_ratio = volume / np.maximum(vol_sma20, 1e-9)
    bull_ob = np.concatenate([[False], (close[:-1] < open_[:-1]) & (vol_ratio[1:] >= 1.5) & (close[1:] > high[:-1])])
    bear_ob = np.concatenate([[False], (close[:-1] > open_[:-1]) & (vol_ratio[1:] >= 1.5) & (close[1:] < low[:-1])])
    df['bull_ob'] = bull_ob
    df['bear_ob'] = bear_ob

    # 6. Fair Value Gap (FVG)
    bull_fvg = (low > np.roll(high, 2)) & ((low - np.roll(high, 2)) >= 0.5 * atr)
    bear_fvg = (high < np.roll(low, 2)) & ((np.roll(low, 2) - high) >= 0.5 * atr)
    bull_fvg[0] = bull_fvg[1] = False
    bear_fvg[0] = bear_fvg[1] = False
    df['bull_fvg'] = bull_fvg
    df['bear_fvg'] = bear_fvg

    # 7. Confluence Scoring
    # Long Score: Bull Trend (30), SSL Sweep (35), Bull FVG/OB (35)
    f1_long = bull_trend.astype(float)
    f2_long = ssl_swept.astype(float)
    f3_long = np.minimum(bull_ob.astype(float) * 1.0 + bull_fvg.astype(float) * 0.8, 1.0)
    score_long = 100.0 * (0.35 * f1_long + 0.35 * f2_long + 0.30 * f3_long)

    f1_short = bear_trend.astype(float)
    f2_short = bsl_swept.astype(float)
    f3_short = np.minimum(bear_ob.astype(float) * 1.0 + bear_fvg.astype(float) * 0.8, 1.0)
    score_short = 100.0 * (0.35 * f1_short + 0.35 * f2_short + 0.30 * f3_short)

    df['score_long'] = score_long
    df['score_short'] = score_short

    # Trigger signal: Score >= 65 and requires sweep or (trend & ob/fvg)
    long_sig = (score_long >= 65.0) & (f2_long > 0)
    short_sig = (score_short >= 65.0) & (f2_short > 0)

    signal = np.zeros(n, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1
    df['signal'] = signal
    return df

class HighFrequencySMCEngineV2:
    def __init__(
        self,
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,   # 0.02%
        taker_fee: float = 0.0005,   # 0.05%
        slippage: float = 0.0005,    # 0.05%
        risk_pct: float = 2.0,       # 2% risk per trade
        use_compound: bool = True,   # Compounding reinvestment
        rr_tp1: float = 1.5,
        rr_tp2: float = 3.0,
        early_be_rr: float = 0.8,
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

    def run(self, df: pd.DataFrame) -> dict:
        if len(df) < 50:
            return {"trades": [], "equity_curve": [self.initial_cash], "metrics": {}}

        data = compute_smc_hf_alpha(df)

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

        close_arr = data['close'].values
        high_arr = data['high'].values
        low_arr = data['low'].values
        atr_arr = data['atr'].values
        signal_arr = data['signal'].values
        hh_arr = data['hh'].values
        ll_arr = data['ll'].values
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
                if sig in (1, -1):
                    direction = sig
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
        ann_factor = np.sqrt(252 * 24 * 12) # 5m bars
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
