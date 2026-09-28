import sys, os, gc, math
import pandas as pd
import numpy as np

def compute_smc_qlib_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Qlib-inspired High-Frequency Alpha Factor Pipeline on 15m/5m Bars:
    1. Alpha_Trend: EMA(12) - EMA(26) MACD-style momentum + EMA(50) slope
    2. Alpha_Breakout: Donchian High/Low channel breakout with Volume confirming
    3. Alpha_SMC_Liquidity: Sweep of previous swing extremes followed by engulfing
    4. Alpha_MicroStructure: Close position within range (Close - Low)/(High - Low)
    5. Execution: Limit Order entry at discount/premium zone + tight stop loss
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values
    n = len(df)

    # 1. ATR (14)
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

    # 2. Trend & Momentum (Fast EMA 10, Medium EMA 30, Slow EMA 100)
    ema10 = pd.Series(close).ewm(span=10, adjust=False).mean().values
    ema30 = pd.Series(close).ewm(span=30, adjust=False).mean().values
    ema100 = pd.Series(close).ewm(span=100, adjust=False).mean().values
    df['ema10'] = ema10
    df['ema30'] = ema30
    df['ema100'] = ema100

    # Trend regime: EMA10 > EMA30 and Close > EMA100
    bull_regime = (ema10 > ema30) & (close > ema100)
    bear_regime = (ema10 < ema30) & (close < ema100)

    # 3. Micro Pullback in Trend (The core of profitable trading)
    # Pullback in Bull trend: Low dips into EMA10-EMA30 band and bounces (Close > Open)
    bull_pullback = bull_regime & (low <= ema10) & (close > open_) & (close > ema10)
    bear_pullback = bear_regime & (high >= ema10) & (close < open_) & (close < ema10)

    # 4. Volume expansion confirmation
    vol_sma15 = pd.Series(volume).rolling(15, min_periods=1).mean().values
    vol_ok = volume >= 0.8 * vol_sma15

    # 5. Liquidity Sweep confirmation
    rolling_h10 = pd.Series(high).shift(1).rolling(10, min_periods=1).max().values
    rolling_l10 = pd.Series(low).shift(1).rolling(10, min_periods=1).min().values
    ssl_sweep = (low < rolling_l10) & (close > rolling_l10)
    bsl_sweep = (high > rolling_h10) & (close < rolling_h10)

    # Combined Alpha Signal
    long_sig = (bull_pullback | ssl_sweep) & vol_ok
    short_sig = (bear_pullback | bsl_sweep) & vol_ok

    signal = np.zeros(n, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1
    df['signal'] = signal
    return df

class HighFrequencySMCEngineV3:
    def __init__(
        self,
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,   # 0.02%
        taker_fee: float = 0.0005,   # 0.05%
        slippage: float = 0.0005,    # 0.05%
        risk_pct: float = 1.0,       # 1% risk per trade for high frequency
        use_compound: bool = True,   # Compounding reinvestment
        rr_tp1: float = 1.5,
        rr_tp2: float = 3.0,
        early_be_rr: float = 0.8,
        early_be_buffer_atr: float = 0.1
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

        data = compute_smc_qlib_alpha(df)

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
                    stop_dist = max(1.2 * atr, close * 0.005) # at least 0.5% stop
                    if stop_dist <= close * 0.04:
                        base_capital = max(equity, 10.0) if self.use_compound else self.initial_cash
                        risk_amt = base_capital * (self.risk_pct / 100.0)
                        size = risk_amt / stop_dist

                        # Max leverage 4x cap
                        max_size = (base_capital * 4.0) / close
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
        ann_factor = np.sqrt(252 * 24 * 12)
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
