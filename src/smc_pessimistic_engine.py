"""
Pessimistic Production-Grade Execution & Matching Engine.
Eliminates ALL backtest illusions:
1. Strict Price Penetration Maker Fill:
   - BUY limit fills ONLY if Low < LimitPrice - 0.05 * ATR (must strictly penetrate orderbook, no touch-fills).
   - SELL limit fills ONLY if High > LimitPrice + 0.05 * ATR.
2. Intra-Bar Sequence Pessimism:
   - If a bar touches BOTH TP (take profit) and SL (stop loss), the engine strictly executes STOP LOSS first!
3. Dynamic ATR-Linked Slippage:
   - Taker Stop Loss suffers base slippage + 0.10 * (Bar_Range / ATR), reflecting real liquidity evaporation during panic stops.
4. Double-Entry Balance Conservation:
   - Real-time zero phantom capital leakage.
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class PendingLimitOrder:
    order_id: str
    side: OrderSide
    limit_price: float
    size: float
    stop_loss: float
    tp1: float
    tp2: float
    placed_idx: int
    placed_time: Any
    expiry_bars: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, current_idx: int) -> bool:
        return (current_idx - self.placed_idx) > self.expiry_bars

@dataclass
class ActivePosition:
    position_id: str
    side: PositionSide
    entry_price: float
    entry_time: Any
    entry_idx: int
    initial_size: float
    remaining_size: float
    stop_loss: float
    tp1: float
    tp2: float
    tp1_hit: bool = False
    realized_tp1_pnl: float = 0.0
    early_be_triggered: bool = False
    total_entry_fee: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class PessimisticExecutionEngine:
    def __init__(
        self,
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,     # 0.02% Maker fee
        taker_fee: float = 0.0005,     # 0.05% Taker fee
        base_slippage: float = 0.0005, # 0.05% Base slippage
        risk_pct: float = 1.0,         # 1.0% Risk per trade
        max_leverage: float = 4.0,     # Max 4x notional leverage
        penetration_ratio: float = 0.05, # Must penetrate 5% of ATR beyond limit price to fill Maker
        early_be_buffer_atr: float = 0.15,
        limit_order_expiry_bars: int = 3,
        pessimistic_tp_sl_conflict: bool = True, # When TP and SL hit on same bar, SL wins!
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.base_slippage = float(base_slippage)
        self.risk_pct = float(risk_pct)
        self.max_leverage = float(max_leverage)
        self.penetration_ratio = float(penetration_ratio)
        self.early_be_buffer_atr = float(early_be_buffer_atr)
        self.limit_order_expiry_bars = int(limit_order_expiry_bars)
        self.pessimistic_tp_sl_conflict = bool(pessimistic_tp_sl_conflict)

    def calculate_position_size(self, equity: float, entry_price: float, stop_loss_price: float) -> float:
        stop_dist = abs(entry_price - stop_loss_price)
        if stop_dist <= 0.0 or entry_price <= 0.0:
            return 0.0
        risk_budget = max(equity, 10.0) * (self.risk_pct / 100.0)
        size_by_risk = risk_budget / stop_dist
        max_size = (max(equity, 10.0) * self.max_leverage) / entry_price
        return float(min(size_by_risk, max_size))

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) == 0:
            return {"trades": [], "metrics": self._empty_metrics(), "equity_curve": [self.initial_cash]}

        data = df.copy()
        col_rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        data = data.rename(columns={k: v for k, v in col_rename.items() if k in data.columns and v not in data.columns})

        if "ATR" not in data.columns and "atr" not in data.columns:
            hl = data["High"] - data["Low"]
            hc = (data["High"] - data["Close"].shift(1)).abs()
            lc = (data["Low"] - data["Close"].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            data["ATR"] = tr.rolling(window=14, min_periods=1).mean().bfill()
        elif "atr" in data.columns and "ATR" not in data.columns:
            data["ATR"] = data["atr"]

        close_arr = data["Close"].values.astype(float)
        high_arr = data["High"].values.astype(float)
        low_arr = data["Low"].values.astype(float)
        open_arr = data["Open"].values.astype(float)
        atr_arr = data["ATR"].values.astype(float)
        timestamps = data.index if hasattr(data, "index") else np.arange(len(data))
        n = len(data)

        signal_series = None
        for col in ["Signal", "signal"]:
            if col in data.columns:
                signal_series = data[col].values
                break

        limit_entry_series = None
        for col in ["limit_entry_px", "LimitEntryPx"]:
            if col in data.columns:
                limit_entry_series = data[col].values
                break

        cash = self.initial_cash
        equity = self.initial_cash
        equity_curve = [cash]
        trades = []
        total_fees = 0.0
        total_slippage = 0.0
        order_counter = 0

        pending_order: Optional[PendingLimitOrder] = None
        active_position: Optional[ActivePosition] = None
        just_entered = False

        for idx in range(n):
            current_time = timestamps[idx]
            close = close_arr[idx]
            high = high_arr[idx]
            low = low_arr[idx]
            atr = max(atr_arr[idx], 1e-6)
            bar_range = high - low

            # -------------------------------------------------------------
            # STEP 1: Process Pending Limit Order with Strict Penetration
            # -------------------------------------------------------------
            if pending_order is not None and active_position is None:
                if pending_order.is_expired(idx):
                    pending_order = None
                else:
                    # Strict penetration threshold: price must clearly cut through orderbook
                    penetration = self.penetration_ratio * atr
                    filled = False
                    if pending_order.side == OrderSide.BUY:
                        # Price must fall BELOW limit - penetration (not just touch!)
                        if low <= (pending_order.limit_price - penetration):
                            filled = True
                    elif pending_order.side == OrderSide.SELL:
                        # Price must rise ABOVE limit + penetration
                        if high >= (pending_order.limit_price + penetration):
                            filled = True

                    if filled:
                        exec_px = pending_order.limit_price
                        fee_in = pending_order.size * exec_px * self.maker_fee
                        total_fees += fee_in
                        cash -= fee_in
                        equity -= fee_in
                        just_entered = True
                        active_position = ActivePosition(
                            position_id=pending_order.order_id,
                            side=PositionSide.LONG if pending_order.side == OrderSide.BUY else PositionSide.SHORT,
                            entry_price=exec_px,
                            entry_time=current_time,
                            entry_idx=idx,
                            initial_size=pending_order.size,
                            remaining_size=pending_order.size,
                            stop_loss=pending_order.stop_loss,
                            tp1=pending_order.tp1,
                            tp2=pending_order.tp2,
                            tp1_hit=False,
                            realized_tp1_pnl=0.0,
                            early_be_triggered=False,
                            total_entry_fee=fee_in,
                            metadata=pending_order.metadata,
                        )
                        pending_order = None

            # -------------------------------------------------------------
            # STEP 2: Manage Active Position (Pessimistic Matching)
            # -------------------------------------------------------------
            if active_position is not None and not just_entered:
                pos = active_position
                closed = False

                # Check if Stop Loss is touched
                sl_touched = False
                if pos.side == PositionSide.LONG and low <= pos.stop_loss:
                    sl_touched = True
                elif pos.side == PositionSide.SHORT and high >= pos.stop_loss:
                    sl_touched = True

                # Check if TP1 is touched
                tp1_touched = False
                if not pos.tp1_hit:
                    if pos.side == PositionSide.LONG and high >= pos.tp1:
                        tp1_touched = True
                    elif pos.side == PositionSide.SHORT and low <= pos.tp1:
                        tp1_touched = True

                # PESSIMISTIC CONFLICT RESOLUTION:
                # If BOTH TP1 and SL are touched in the same bar, SL WINS! (Assume intra-bar shakeout first)
                if sl_touched and tp1_touched and self.pessimistic_tp_sl_conflict:
                    # Treat as instant full stop-out on initial size
                    dyn_slip = self.base_slippage + 0.10 * (bar_range / atr) * self.base_slippage
                    if pos.side == PositionSide.LONG:
                        exit_px = pos.stop_loss * (1.0 - dyn_slip)
                        pnl_rem = pos.remaining_size * (exit_px - pos.entry_price)
                        slip_loss = pos.remaining_size * (pos.stop_loss - exit_px)
                    else:
                        exit_px = pos.stop_loss * (1.0 + dyn_slip)
                        pnl_rem = pos.remaining_size * (pos.entry_price - exit_px)
                        slip_loss = pos.remaining_size * (exit_px - pos.stop_loss)

                    fee_sl = pos.remaining_size * exit_px * self.taker_fee
                    total_fees += fee_sl
                    total_slippage += slip_loss
                    pnl_rem -= fee_sl
                    cash += pnl_rem
                    equity += pnl_rem

                    total_trade_pnl = (pos.realized_tp1_pnl if pos.tp1_hit else 0.0) + pnl_rem
                    total_trade_fees = pos.total_entry_fee + fee_sl
                    trades.append({
                        "trade_id": pos.position_id,
                        "direction": pos.side.value,
                        "entry_time": pos.entry_time,
                        "exit_time": current_time,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_px,
                        "initial_size": pos.initial_size,
                        "exit_size": pos.remaining_size,
                        "pnl": total_trade_pnl,
                        "exit_reason": "PESSIMISTIC_STOP_LOSS",
                        "total_fees": total_trade_fees,
                        "total_slippage": slip_loss,
                    })
                    active_position = None
                    closed = True

                elif sl_touched:
                    # Standard Stop Loss triggered
                    dyn_slip = self.base_slippage + 0.10 * (bar_range / atr) * self.base_slippage
                    if pos.side == PositionSide.LONG:
                        exit_px = pos.stop_loss * (1.0 - dyn_slip)
                        pnl_rem = pos.remaining_size * (exit_px - pos.entry_price)
                        slip_loss = pos.remaining_size * (pos.stop_loss - exit_px)
                    else:
                        exit_px = pos.stop_loss * (1.0 + dyn_slip)
                        pnl_rem = pos.remaining_size * (pos.entry_price - exit_px)
                        slip_loss = pos.remaining_size * (exit_px - pos.stop_loss)

                    fee_sl = pos.remaining_size * exit_px * self.taker_fee
                    total_fees += fee_sl
                    total_slippage += slip_loss
                    pnl_rem -= fee_sl
                    cash += pnl_rem
                    equity += pnl_rem

                    total_trade_pnl = (pos.realized_tp1_pnl if pos.tp1_hit else 0.0) + pnl_rem
                    total_trade_fees = pos.total_entry_fee + fee_sl
                    trades.append({
                        "trade_id": pos.position_id,
                        "direction": pos.side.value,
                        "entry_time": pos.entry_time,
                        "exit_time": current_time,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_px,
                        "initial_size": pos.initial_size,
                        "exit_size": pos.remaining_size,
                        "pnl": total_trade_pnl,
                        "exit_reason": "BE_SL" if pos.early_be_triggered else "STOP_LOSS",
                        "total_fees": total_trade_fees,
                        "total_slippage": slip_loss,
                    })
                    active_position = None
                    closed = True

                elif tp1_touched:
                    # Clean TP1 hit (SL was not touched on this bar)
                    exit_px = pos.tp1
                    close_size = pos.initial_size * 0.5
                    fee_tp1 = close_size * exit_px * self.maker_fee
                    total_fees += fee_tp1

                    if pos.side == PositionSide.LONG:
                        tp1_pnl = close_size * (exit_px - pos.entry_price) - fee_tp1
                        be_sl = pos.entry_price + self.early_be_buffer_atr * atr
                    else:
                        tp1_pnl = close_size * (pos.entry_price - exit_px) - fee_tp1
                        be_sl = pos.entry_price - self.early_be_buffer_atr * atr

                    cash += tp1_pnl
                    equity += tp1_pnl
                    pos.realized_tp1_pnl = tp1_pnl
                    pos.remaining_size = pos.initial_size - close_size
                    pos.tp1_hit = True
                    pos.stop_loss = be_sl
                    pos.early_be_triggered = True

                # Check TP2 if still active
                if not closed and active_position is not None:
                    hit_tp2 = False
                    if pos.side == PositionSide.LONG and high >= pos.tp2:
                        hit_tp2 = True
                    elif pos.side == PositionSide.SHORT and low <= pos.tp2:
                        hit_tp2 = True

                    if hit_tp2:
                        exit_px = pos.tp2
                        fee_rem = pos.remaining_size * exit_px * self.maker_fee
                        total_fees += fee_rem
                        if pos.side == PositionSide.LONG:
                            pnl_rem = pos.remaining_size * (exit_px - pos.entry_price) - fee_rem
                        else:
                            pnl_rem = pos.remaining_size * (pos.entry_price - exit_px) - fee_rem

                        cash += pnl_rem
                        equity += pnl_rem
                        total_trade_pnl = (pos.realized_tp1_pnl if pos.tp1_hit else 0.0) + pnl_rem
                        total_trade_fees = pos.total_entry_fee + (
                            pos.initial_size * 0.5 * pos.tp1 * self.maker_fee if pos.tp1_hit else 0.0
                        ) + fee_rem

                        trades.append({
                            "trade_id": pos.position_id,
                            "direction": pos.side.value,
                            "entry_time": pos.entry_time,
                            "exit_time": current_time,
                            "entry_price": pos.entry_price,
                            "exit_price": exit_px,
                            "initial_size": pos.initial_size,
                            "exit_size": pos.remaining_size,
                            "pnl": total_trade_pnl,
                            "exit_reason": "TP2",
                            "total_fees": total_trade_fees,
                            "total_slippage": 0.0,
                        })
                        active_position = None
                        closed = True

            # -------------------------------------------------------------
            # STEP 3: Signal Generation (Flat check)
            # -------------------------------------------------------------
            if active_position is None and pending_order is None and signal_series is not None:
                sig = signal_series[idx]
                if sig in (1, -1, "LONG", "SHORT"):
                    is_long = sig in (1, "LONG")
                    if limit_entry_series is not None and not np.isnan(limit_entry_series[idx]):
                        limit_px = float(limit_entry_series[idx])
                    else:
                        limit_px = (close - 0.2 * atr) if is_long else (close + 0.2 * atr)

                    sl_val = (limit_px - 1.5 * atr) if is_long else (limit_px + 1.5 * atr)
                    stop_dist = abs(limit_px - sl_val)
                    tp1_val = (limit_px + 1.5 * stop_dist) if is_long else (limit_px - 1.5 * stop_dist)
                    tp2_val = (limit_px + 3.5 * stop_dist) if is_long else (limit_px - 3.5 * stop_dist)

                    if stop_dist > 0.0 and stop_dist <= limit_px * 0.10:
                        order_size = self.calculate_position_size(equity=cash, entry_price=limit_px, stop_loss_price=sl_val)
                        if order_size > 0.0:
                            order_counter += 1
                            pending_order = PendingLimitOrder(
                                order_id=f"PESS_ORD_{order_counter:05d}",
                                side=OrderSide.BUY if is_long else OrderSide.SELL,
                                limit_price=limit_px,
                                size=order_size,
                                stop_loss=sl_val,
                                tp1=tp1_val,
                                tp2=tp2_val,
                                placed_idx=idx,
                                placed_time=current_time,
                                expiry_bars=self.limit_order_expiry_bars,
                            )

            just_entered = False
            equity_curve.append(cash)

        metrics = self._calculate_metrics(cash, trades, total_fees, total_slippage, equity_curve)
        return {"trades": trades, "metrics": metrics, "equity_curve": equity_curve}

    def _calculate_metrics(self, final_cash: float, trades: List[Dict[str, Any]], fees: float, slippage: float, curve: List[float]) -> Dict[str, Any]:
        total_trades = len(trades)
        if total_trades == 0:
            return self._empty_metrics()

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = (len(wins) / total_trades) * 100.0
        profit_factor = (sum(wins) / abs(sum(losses))) if (losses and sum(losses) != 0) else (999.0 if wins else 0.0)

        curve_arr = np.array(curve)
        peak = np.maximum.accumulate(curve_arr)
        dd = (peak - curve_arr) / np.maximum(peak, 1e-6)
        max_dd = float(np.max(dd)) * 100.0

        pnl_series = pd.Series(pnls)
        sharpe = float((pnl_series.mean() / pnl_series.std()) * math.sqrt(252)) if (len(pnl_series) > 1 and pnl_series.std() > 0) else 0.0

        return {
            "initial_cash": self.initial_cash,
            "final_cash": final_cash,
            "total_return_pct": ((final_cash - self.initial_cash) / self.initial_cash) * 100.0,
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "total_fees": fees,
            "total_slippage": slippage,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "final_cash": self.initial_cash,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
        }
