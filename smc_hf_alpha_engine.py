import sys, os, gc, math
import pandas as pd
import numpy as np

def compute_qlib_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Qlib-inspired Alpha 101/158 Factor Extraction Pipeline for Micro-Structure & Momentum:
    - Ret_k: Multi-period returns (momentum & mean-reversion)
    - Vol_k: Rolling volatility & volume spikes
    - Orderflow Imbalance proxy: (Close - Low) / (High - Low)
    - VWAP divergence proxy: Close vs Volume-Weighted Average
    - Micro FVG & Micro Order Blocks
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values
    n = len(df)

    # 1. Micro ATR (14 bars)
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

    # 2. Qlib Volume Spike Factor (Vol / SMA(Vol, 20))
    vol_sma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_spike = volume / np.maximum(vol_sma20, 1e-9)
    df['vol_spike'] = vol_spike

    # 3. Micro Order Block (OB): High volume displacement candle
    # Bullish OB: last down-candle before strong bullish displacement
    is_down = close < open_
    is_up = close > open_
    bull_ob = np.concatenate([[False], is_down[:-1] & (vol_spike[1:] >= 1.25) & (close[1:] > high[:-1])])
    bear_ob = np.concatenate([[False], is_up[:-1] & (vol_spike[1:] >= 1.25) & (close[1:] < low[:-1])])
    df['bull_ob'] = bull_ob
    df['bear_ob'] = bear_ob

    # 4. Micro Fair Value Gap (FVG)
    bull_fvg = (low > np.roll(high, 2)) & ((low - np.roll(high, 2)) >= 0.25 * atr)
    bear_fvg = (high < np.roll(low, 2)) & ((np.roll(low, 2) - high) >= 0.25 * atr)
    bull_fvg[0] = bull_fvg[1] = False
    bear_fvg[0] = bear_fvg[1] = False
    df['bull_fvg'] = bull_fvg
    df['bear_fvg'] = bear_fvg

    # 5. Micro Liquidity Sweep (Lookback 5 bars)
    rolling_h5 = pd.Series(high).shift(1).rolling(5, min_periods=1).max().values
    rolling_l5 = pd.Series(low).shift(1).rolling(5, min_periods=1).min().values
    bsl_sweep = (high > rolling_h5) & (close < rolling_h5)
    ssl_sweep = (low < rolling_l5) & (close > rolling_l5)
    df['bsl_sweep'] = bsl_sweep
    df['ssl_sweep'] = ssl_sweep

    # 6. Trend State: Fast EMA (EMA 8 vs EMA 21)
    ema8 = pd.Series(close).ewm(span=8, adjust=False).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    trend_up = ema8 > ema21
    trend_down = ema8 < ema21

    # 7. Qlib Alpha Momentum & Order Flow Confluence Score
    # Score in [0, 100]
    score_long = (
        (trend_up.astype(float) * 25.0) +
        (ssl_sweep.astype(float) * 30.0) +
        (bull_fvg.astype(float) * 25.0) +
        (bull_ob.astype(float) * 20.0)
    )
    score_short = (
        (trend_down.astype(float) * 25.0) +
        (bsl_sweep.astype(float) * 30.0) +
        (bear_fvg.astype(float) * 25.0) +
        (bear_ob.astype(float) * 20.0)
    )
    df['score_long'] = score_long
    df['score_short'] = score_short

    # Signal generation: High-Frequency SMC trigger
    # Requires sweep OR (OB/FVG + Trend) with Score >= 50
    long_sig = (score_long >= 50.0) & (ssl_sweep | (trend_up & (bull_fvg | bull_ob)))
    short_sig = (score_short >= 50.0) & (bsl_sweep | (trend_down & (bear_fvg | bear_ob)))

    signal = np.zeros(n, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1
    df['signal'] = signal

    return df

class HighFrequencySMCEngine:
    def __init__(
        self,
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,   # 0.02%
        taker_fee: float = 0.0005,   # 0.05%
        slippage: float = 0.0005,    # 0.05%
        risk_pct: float = 1.5,       # 1.5% risk per trade for higher frequency compounding
        use_compound: bool = True,   # Compounding reinvestment
        rr_tp1: float = 1.2,
        rr_tp2: float = 2.4,
    ):
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.risk_pct = risk_pct
        self.use_compound = use_compound
        self.rr_tp1 = rr_tp1
        self.rr_tp2 = rr_tp2

    def run(self, df: pd.DataFrame) -> dict:
        if len(df) < 50:
            return {"trades": [], "equity_curve": [self.initial_cash], "metrics": {}}

        data = compute_qlib_alpha_factors(df)

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

        total_fees = 0.0
        total_slippage = 0.0

        close_arr = data['close'].values
        high_arr = data['high'].values
        low_arr = data['low'].values
        atr_arr = data['atr'].values
        signal_arr = data['signal'].values
        score_long_arr = data['score_long'].values
        score_short_arr = data['score_short'].values
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
                        fee = (pos_size * 0.5) * exit_px * self.maker_fee
                        total_fees += fee
                        if position == 1:
                            pnl_part = (pos_size * 0.5) * (exit_px - entry_price) - fee
                            planned_sl = entry_price + 0.1 * atr # Lock small profit
                        else:
                            pnl_part = (pos_size * 0.5) * (entry_price - exit_px) - fee
                            planned_sl = entry_price - 0.1 * atr
                        cash += pnl_part
                        equity += pnl_part
                        pos_size *= 0.5
                        tp1_hit = True

                # TP2 Check
                closed = False
                if (position == 1 and high >= tp2) or (position == -1 and low <= tp2):
                    exit_px = tp2
                    fee = pos_size * exit_px * self.maker_fee
                    total_fees += fee
                    if position == 1:
                        pnl = pos_size * (exit_px - entry_price) - fee
                    else:
                        pnl = pos_size * (entry_price - exit_px) - fee
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
                            "exit_reason": "STOP_LOSS"
                        })
                        position = 0
                        closed = True

            # 2. Check Signal if flat
            if position == 0:
                sig = signal_arr[idx]
                if sig in (1, -1):
                    direction = sig
                    # Stop distance: 1.2 * ATR
                    stop_dist = max(1.2 * atr, close * 0.003) # at least 0.3%
                    if stop_dist <= close * 0.04: # max 4% stop
                        base_capital = equity if self.use_compound else self.initial_cash
                        risk_amt = max(base_capital * (self.risk_pct / 100.0), 1.0)
                        size = risk_amt / stop_dist

                        # Max leverage 5x cap for safety
                        max_size = (base_capital * 5.0) / close
                        size = min(size, max_size)

                        if direction == 1:
                            exec_px = close * (1.0 + self.slippage)
                            sl_val = close - stop_dist
                            tp1_val = close + self.rr_tp1 * stop_dist
                            tp2_val = close + self.rr_tp2 * stop_dist
                        else:
                            exec_px = close * (1.0 - self.slippage)
                            sl_val = close + stop_dist
                            tp1_val = close - self.rr_tp1 * stop_dist
                            tp2_val = close - self.rr_tp2 * stop_dist

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
                        planned_sl = sl_val
                        tp1 = tp1_val
                        tp2 = tp2_val
                        tp1_hit = False

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

        # Calculate daily trade frequency
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
