import sys, os, gc, glob, math
from datetime import datetime
import pandas as pd
import numpy as np

# Ensure project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from smc_optimized.v1_internal_factors.indicators import SMCConfig as V1Config, compute_smc_indicators as v1_ind, get_signals as v1_sig
from smc_optimized.v2_volatility_regime.indicators import SMCConfig as V2Config, compute_smc_indicators as v2_ind, get_signals as v2_sig
from smc_optimized.v3_trend_strength.indicators import SMCConfig as V3Config, compute_smc_indicators as v3_ind, get_signals as v3_sig

def get_strategy_config_and_funcs(strat_name: str):
    if strat_name == 'v1_internal_factors':
        return V1Config(), v1_ind, v1_sig
    elif strat_name == 'v2_volatility_regime':
        return V2Config(), v2_ind, v2_sig
    elif strat_name == 'v3_trend_strength':
        return V3Config(), v3_ind, v3_sig
    else:
        raise ValueError(f"Unknown strategy {strat_name}")

def load_symbol_timeframe_data(symbol: str, timeframe: str) -> pd.DataFrame:
    pattern = f"data/raw_extracted/{symbol}/*_{timeframe}_*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {symbol} {timeframe}")
    dfs = [pd.read_parquet(f) for f in files]
    full_df = pd.concat(dfs).sort_index()
    # Deduplicate index if any
    full_df = full_df[~full_df.index.duplicated(keep='first')]
    return full_df

class SMCExecutionEngine:
    def __init__(
        self,
        engine_type: str, # 'QuantCell' or 'NautilusTrader'
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage: float = 0.0005,
        risk_pct: float = 2.0,
    ):
        self.engine_type = engine_type
        self.initial_cash = initial_cash
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.risk_pct = risk_pct

    def run(self, df: pd.DataFrame, strat_name: str) -> dict:
        """
        Run backtest on a pre-sliced window df (e.g. In-Sample or Out-of-Sample).
        Returns trades list, equity curve, and metrics dictionary.
        """
        if len(df) < 50:
            return {
                "trades": [],
                "equity_curve": [self.initial_cash],
                "metrics": {
                    "initial_cash": self.initial_cash,
                    "final_cash": self.initial_cash,
                    "total_return_pct": 0.0,
                    "total_trades": 0,
                    "win_rate_pct": 0.0,
                    "profit_factor": 0.0,
                    "max_drawdown_pct": 0.0,
                    "sharpe_ratio": 0.0,
                    "total_pnl": 0.0,
                    "total_fees": 0.0,
                    "total_slippage_cost": 0.0
                }
            }

        config, compute_ind, get_sig = get_strategy_config_and_funcs(strat_name)
        data = compute_ind(df, config)
        data = get_sig(data, config)

        cash = self.initial_cash
        equity_curve = [cash]
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
        score = 0.0
        factors = 0

        total_fees = 0.0
        total_slippage = 0.0

        n = len(data)
        close_arr = data['Close'].values
        high_arr = data['High'].values
        low_arr = data['Low'].values
        open_arr = data['Open'].values
        atr_arr = data['ATR'].values
        signal_arr = data['Signal'].values
        score_long_arr = data['ScoreLong'].values
        score_short_arr = data['ScoreShort'].values
        factors_long_arr = data['FactorsLong'].values
        factors_short_arr = data['FactorsShort'].values
        hh_arr = data['HH'].values
        ll_arr = data['LL'].values
        timestamps = data.index

        for idx in range(n):
            close = close_arr[idx]
            high = high_arr[idx]
            low = low_arr[idx]
            atr = atr_arr[idx]
            timestamp = timestamps[idx]

            # 1. Manage existing position
            if position != 0:
                # TP1 Check (Maker fee assumed for limit take-profit in Nautilus/QuantCell)
                if not tp1_hit:
                    if position == 1 and high >= tp1:
                        # Partial close
                        exit_px = tp1
                        # TP limit order has no slippage, charges maker_fee
                        fee = pos_size * exit_px * self.maker_fee
                        pnl = pos_size * (exit_px - entry_price) - fee
                        cash += pnl
                        total_fees += fee
                        tp1_hit = True

                        buffer = config.early_be_buffer_atr * atr
                        planned_sl = entry_price + buffer
                        early_be_triggered = True

                    elif position == -1 and low <= tp1:
                        exit_px = tp1
                        fee = pos_size * exit_px * self.maker_fee
                        pnl = pos_size * (entry_price - exit_px) - fee
                        cash += pnl
                        total_fees += fee
                        tp1_hit = True

                        buffer = config.early_be_buffer_atr * atr
                        planned_sl = entry_price - buffer
                        early_be_triggered = True

                # TP2 Check
                closed = False
                if position == 1 and high >= tp2:
                    exit_px = tp2
                    fee = pos_size * exit_px * self.maker_fee
                    pnl = pos_size * (exit_px - entry_price) - fee
                    cash += pnl
                    total_fees += fee
                    trades.append({
                        "entry_time": str(entry_time),
                        "exit_time": str(timestamp),
                        "direction": "LONG",
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_px),
                        "size": float(pos_size),
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                        "exit_reason": "TP2",
                        "score": float(score),
                        "factors": int(factors)
                    })
                    position = 0
                    closed = True
                elif position == -1 and low <= tp2:
                    exit_px = tp2
                    fee = pos_size * exit_px * self.maker_fee
                    pnl = pos_size * (entry_price - exit_px) - fee
                    cash += pnl
                    total_fees += fee
                    trades.append({
                        "entry_time": str(entry_time),
                        "exit_time": str(timestamp),
                        "direction": "SHORT",
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_px),
                        "size": float(pos_size),
                        "pnl": float(pnl),
                        "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                        "exit_reason": "TP2",
                        "score": float(score),
                        "factors": int(factors)
                    })
                    position = 0
                    closed = True

                # Stop Loss Check (Market order: charges taker_fee + slippage)
                if not closed:
                    if position == 1 and low <= planned_sl:
                        # Slippage makes long stop loss exit lower
                        exit_px = planned_sl * (1.0 - self.slippage)
                        slip_loss = pos_size * (planned_sl - exit_px)
                        total_slippage += slip_loss
                        fee = pos_size * exit_px * self.taker_fee
                        total_fees += fee
                        pnl = pos_size * (exit_px - entry_price) - fee
                        cash += pnl
                        trades.append({
                            "entry_time": str(entry_time),
                            "exit_time": str(timestamp),
                            "direction": "LONG",
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_px),
                            "size": float(pos_size),
                            "pnl": float(pnl),
                            "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                            "exit_reason": "STOP_LOSS",
                            "score": float(score),
                            "factors": int(factors)
                        })
                        position = 0
                        closed = True
                    elif position == -1 and high >= planned_sl:
                        # Slippage makes short stop loss exit higher
                        exit_px = planned_sl * (1.0 + self.slippage)
                        slip_loss = pos_size * (exit_px - planned_sl)
                        total_slippage += slip_loss
                        fee = pos_size * exit_px * self.taker_fee
                        total_fees += fee
                        pnl = pos_size * (entry_price - exit_px) - fee
                        cash += pnl
                        trades.append({
                            "entry_time": str(entry_time),
                            "exit_time": str(timestamp),
                            "direction": "SHORT",
                            "entry_price": float(entry_price),
                            "exit_price": float(exit_px),
                            "size": float(pos_size),
                            "pnl": float(pnl),
                            "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                            "exit_reason": "STOP_LOSS",
                            "score": float(score),
                            "factors": int(factors)
                        })
                        position = 0
                        closed = True

                # Early BE Check
                if not closed and config.use_early_be and not early_be_triggered:
                    stop_dist = abs(entry_price - planned_sl)
                    buffer = config.early_be_buffer_atr * atr
                    if position == 1:
                        mfe = high - entry_price
                        if mfe >= config.early_be_rr * stop_dist:
                            planned_sl = entry_price + buffer
                            early_be_triggered = True
                    elif position == -1:
                        mfe = entry_price - low
                        if mfe >= config.early_be_rr * stop_dist:
                            planned_sl = entry_price - buffer
                            early_be_triggered = True

            # 2. Check entry signal if flat
            if position == 0:
                sig = signal_arr[idx]
                if sig in ('LONG', 'SHORT'):
                    direction = 1 if sig == 'LONG' else -1
                    score_val = score_long_arr[idx] if sig == 'LONG' else score_short_arr[idx]
                    factors_val = factors_long_arr[idx] if sig == 'LONG' else factors_short_arr[idx]

                    buffer = config.sl_buffer_atr * atr
                    hh = hh_arr[idx]
                    ll = ll_arr[idx]

                    if direction == 1:
                        sl_calc = (ll if not np.isnan(ll) else close) - buffer
                        stop_dist = close - sl_calc
                        tp1_calc = close + config.rr_tp1 * stop_dist
                        tp2_calc = close + config.rr_tp2 * stop_dist
                    else:
                        sl_calc = (hh if not np.isnan(hh) else close) + buffer
                        stop_dist = sl_calc - close
                        tp1_calc = close - config.rr_tp1 * stop_dist
                        tp2_calc = close - config.rr_tp2 * stop_dist

                    if stop_dist > 0 and stop_dist <= close * 0.05:
                        risk_amt = cash * (self.risk_pct / 100.0)
                        size_calc = risk_amt / stop_dist

                        # Entry execution: Market order (taker_fee + slippage)
                        if direction == 1:
                            exec_price = close * (1.0 + self.slippage)
                            slip_cost = size_calc * (exec_price - close)
                        else:
                            exec_price = close * (1.0 - self.slippage)
                            slip_cost = size_calc * (close - exec_price)

                        fee_cost = size_calc * exec_price * self.taker_fee
                        total_fees += fee_cost
                        total_slippage += slip_cost
                        cash -= fee_cost

                        position = direction
                        pos_size = size_calc
                        entry_price = exec_price
                        entry_time = timestamp
                        planned_sl = sl_calc
                        tp1 = tp1_calc
                        tp2 = tp2_calc
                        tp1_hit = False
                        early_be_triggered = False
                        score = score_val
                        factors = factors_val

            # 3. Mark to market equity
            if position != 0:
                if position == 1:
                    unrealized = pos_size * (close - entry_price) - pos_size * close * self.taker_fee
                else:
                    unrealized = pos_size * (entry_price - close) - pos_size * close * self.taker_fee
                equity_curve.append(cash + unrealized)
            else:
                equity_curve.append(cash)

        # Close position at end of period if still open
        if position != 0:
            exit_px = close_arr[-1]
            fee = pos_size * exit_px * self.taker_fee
            total_fees += fee
            if position == 1:
                pnl = pos_size * (exit_px - entry_price) - fee
            else:
                pnl = pos_size * (entry_price - exit_px) - fee
            cash += pnl
            trades.append({
                "entry_time": str(entry_time),
                "exit_time": str(timestamps[-1]),
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": float(entry_price),
                "exit_price": float(exit_px),
                "size": float(pos_size),
                "pnl": float(pnl),
                "pnl_pct": float(pnl / (pos_size * entry_price) * 100),
                "exit_reason": "END_OF_DATA",
                "score": float(score),
                "factors": int(factors)
            })
            equity_curve[-1] = cash

        # Metrics calculation
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
        # Annualized Sharpe (assuming 252*24 for 1h, 252*24*4 for 15m)
        ann_factor = np.sqrt(252 * 24)
        sharpe = float(np.mean(rets) / std_ret * ann_factor) if std_ret > 1e-9 else 0.0

        total_pnl = cash - self.initial_cash
        ret_pct = (total_pnl / self.initial_cash) * 100.0

        metrics = {
            "initial_cash": float(self.initial_cash),
            "final_cash": float(cash),
            "total_return_pct": float(ret_pct),
            "total_trades": len(trades),
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
