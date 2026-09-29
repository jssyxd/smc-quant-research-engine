"""
Round 2 High-Frequency Smart Money Concepts (SMC) Execution Engine.

Features:
1. Maker Limit Order Entry:
   - Orders placed at `limit_entry_px` (Order Block / FVG equilibrium).
   - If next bar's price reaches `limit_entry_px` (low <= limit for buy, high >= limit for sell),
     order fills as Maker (maker_fee, 0.00% slippage).
   - If not reached within time-in-force (default: 3 bars), the order is cancelled.
2. Accurate Double-Entry Bookkeeping:
   - TP1 (default 1.5R): closes exactly 50% (close_size = initial_pos_size * 0.5), Maker fee.
   - Early BE buffer: shifts SL to entry_price +- early_be_buffer_atr * atr.
   - TP2 (default 3.5R): closes remaining 50% with Maker fee.
   - SL: closes remaining position with Taker fee + slippage.
   - Zero double-counting: total trade PnL strictly equals realized TP1 PnL + remaining exit PnL.
3. Compounding & Risk Control:
   - Dynamic position sizing: Size = (Equity * risk_pct) / StopDistance.
   - Leverage cap: max notional leverage (default: 4.0x) for safety.
4. Credal Uncertainty Gate:
   - Integrates with CredalSMCEngine (NeurIPS 2025 Credal Transformer / Subjective Logic):
     abstains from placing limit orders when epistemic uncertainty u > u_max (0.35) or conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.credal_engine import CredalSMCEngine, CredalSignal, CredalDecision


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


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


@dataclass
class TradeRecord:
    trade_id: str
    direction: str
    entry_time: Any
    exit_time: Any
    entry_price: float
    exit_price: float
    initial_size: float
    exit_size: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    tp1_hit: bool
    tp1_pnl: float
    remaining_pnl: float
    total_fees: float
    total_slippage: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "direction": self.direction,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "size": float(self.initial_size),
            "initial_size": float(self.initial_size),
            "exit_size": float(self.exit_size),
            "pnl": float(self.pnl),
            "pnl_pct": float(self.pnl_pct),
            "exit_reason": self.exit_reason,
            "tp1_hit": bool(self.tp1_hit),
            "tp1_pnl": float(self.tp1_pnl),
            "remaining_pnl": float(self.remaining_pnl),
            "total_fees": float(self.total_fees),
            "total_slippage": float(self.total_slippage),
            **self.metadata,
        }


class Round2SMCEngine:
    """Round 2 High-Frequency SMC execution engine with Maker limit entries and credal gating.

    Parameters
    ----------
    initial_cash : float, default=1000.0
        Starting trading capital.
    maker_fee : float, default=0.0002
        Maker execution fee (0.02% = 2 bps).
    taker_fee : float, default=0.0005
        Taker execution fee (0.05% = 5 bps).
    slippage : float, default=0.0005
        Market/Taker slippage (0.05% = 5 bps). Maker orders experience 0.00% slippage.
    risk_pct : float, default=2.0
        Risk percentage per trade (2.0%).
    use_compound : bool, default=True
        Whether position sizing dynamically scales with real-time equity.
    max_leverage : float, default=4.0
        Maximum notional leverage cap for safety.
    rr_tp1 : float, default=1.5
        Risk-reward ratio for Take Profit 1 (TP1).
    rr_tp2 : float, default=3.5
        Risk-reward ratio for Take Profit 2 (TP2).
    early_be_buffer_atr : float, default=0.15
        ATR buffer added/subtracted to entry price when early breakeven is triggered.
    limit_order_expiry_bars : int, default=3
        Maximum bars a pending limit order stays active before cancellation.
    use_credal : bool, default=True
        Whether to enforce the Credal Uncertainty Gate before placing limit orders.
    credal_u_max : float, default=0.35
        Epistemic uncertainty threshold for abstention.
    credal_delta : float, default=0.15
        Belief margin threshold for conflicting evidence abstention.
    """

    def __init__(
        self,
        initial_cash: float = 1000.0,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0005,
        slippage: float = 0.0005,
        risk_pct: float = 2.0,
        use_compound: bool = True,
        max_leverage: float = 4.0,
        sl_atr_mult: float = 1.5,
        rr_tp1: float = 1.5,
        rr_tp2: float = 3.5,
        early_be_buffer_atr: float = 0.15,
        limit_order_expiry_bars: int = 3,
        use_credal: bool = True,
        credal_u_max: float = 0.35,
        credal_delta: float = 0.15,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.maker_fee = float(maker_fee)
        self.taker_fee = float(taker_fee)
        self.slippage = float(slippage)
        self.risk_pct = float(risk_pct)
        self.use_compound = bool(use_compound)
        self.max_leverage = float(max_leverage)
        self.sl_atr_mult = float(sl_atr_mult)
        self.rr_tp1 = float(rr_tp1)
        self.rr_tp2 = float(rr_tp2)
        self.early_be_buffer_atr = float(early_be_buffer_atr)
        self.limit_order_expiry_bars = int(limit_order_expiry_bars)
        self.use_credal = bool(use_credal)
        self.credal_u_max = float(credal_u_max)
        self.credal_delta = float(credal_delta)

        self.credal_engine = (
            CredalSMCEngine(u_max=self.credal_u_max, delta=self.credal_delta)
            if self.use_credal
            else None
        )

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> float:
        """Calculate position size according to risk percentage and leverage cap.

        Size = (Equity * risk_pct) / StopDistance
        Leverage Cap = (Equity * max_leverage) / entry_price
        """
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0.0 or entry_price <= 0.0:
            return 0.0

        base_capital = max(equity, 10.0) if self.use_compound else self.initial_cash
        risk_budget = base_capital * (self.risk_pct / 100.0)
        size_by_risk = risk_budget / stop_distance

        # Leverage cap: max 4x notional leverage for safety
        max_allowed_size = (base_capital * self.max_leverage) / entry_price
        final_size = min(size_by_risk, max_allowed_size)

        return float(final_size)

    def evaluate_credal_gate(
        self,
        row: Union[pd.Series, Dict[str, Any]],
    ) -> Tuple[bool, Optional[CredalDecision]]:
        """Evaluate Credal Dirichlet uncertainty gate for entry.

        Returns
        -------
        Tuple[bool, Optional[CredalDecision]]
            (allowed: bool, decision: Optional[CredalDecision])
            allowed is True if credal uncertainty is low and there is no conflict.
        """
        if not self.use_credal or self.credal_engine is None:
            return True, None

        # Build indicator inputs for CredalSMCEngine
        if isinstance(row, pd.Series):
            indicators = row.to_dict()
        else:
            indicators = dict(row)

        decision = self.credal_engine.evaluate(**indicators)
        if decision.should_abstain or decision.uncertainty > self.credal_u_max:
            return False, decision

        return True, decision

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute Round 2 backtest on OHLCV DataFrame.

        Required columns:
            'Open', 'High', 'Low', 'Close'
        Optional/Recommended columns:
            'ATR' (calculated if missing)
            'Signal' or 'signal' ('LONG', 'SHORT', 'NEUTRAL')
            'limit_entry_px' or 'LimitEntryPx' (if absent, uses OB/FVG equilibrium or current Close)
            'stop_loss' or 'StopLoss'
            'tp1' or 'TP1', 'tp2' or 'TP2'
            Credal factor columns: 'htf_bull_bias', 'bull_ob', 'ssl_swept', etc.
        """
        if len(df) == 0:
            return {
                "trades": [],
                "equity_curve": [self.initial_cash],
                "metrics": self._empty_metrics(),
            }

        data = df.copy()
        col_rename = {
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
        }
        data = data.rename(columns={k: v for k, v in col_rename.items() if k in data.columns and v not in data.columns})

        # Ensure ATR exists
        if "ATR" not in data.columns and "atr" not in data.columns:
            high_low = data["High"] - data["Low"]
            high_prev = (data["High"] - data["Close"].shift(1)).abs()
            low_prev = (data["Low"] - data["Close"].shift(1)).abs()
            tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
            data["ATR"] = tr.rolling(window=14, min_periods=1).mean().bfill()
        elif "atr" in data.columns and "ATR" not in data.columns:
            data["ATR"] = data["atr"]

        # Cache arrays for performance
        close_arr = data["Close"].values.astype(float)
        high_arr = data["High"].values.astype(float)
        low_arr = data["Low"].values.astype(float)
        open_arr = data["Open"].values.astype(float)
        atr_arr = data["ATR"].values.astype(float)
        timestamps = data.index if hasattr(data, "index") else np.arange(len(data))
        n = len(data)

        # Standardize Signal column
        signal_series = None
        for col in ["Signal", "signal"]:
            if col in data.columns:
                signal_series = data[col].values
                break

        # Standardize limit entry price column if provided
        limit_entry_series = None
        for col in ["limit_entry_px", "LimitEntryPx", "limit_entry", "LimitEntry"]:
            if col in data.columns:
                limit_entry_series = data[col].values
                break

        # Standardize SL/TP series if provided
        sl_series = None
        for col in ["stop_loss", "StopLoss", "sl", "SL"]:
            if col in data.columns:
                sl_series = data[col].values
                break

        tp1_series = None
        for col in ["tp1", "TP1"]:
            if col in data.columns:
                tp1_series = data[col].values
                break

        tp2_series = None
        for col in ["tp2", "TP2"]:
            if col in data.columns:
                tp2_series = data[col].values
                break

        # Execution tracking state
        cash = self.initial_cash
        equity = self.initial_cash
        equity_curve: List[float] = [equity]
        trades: List[Dict[str, Any]] = []

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
            atr = atr_arr[idx]

            # -------------------------------------------------------------
            # STEP 1: Process Pending Limit Order against current bar (OHLC)
            # -------------------------------------------------------------
            if pending_order is not None and active_position is None:
                # Check expiration first: if current_idx - placed_idx > expiry_bars
                if pending_order.is_expired(idx):
                    pending_order = None
                else:
                    # Check Maker Fill:
                    # For BUY: low <= limit_price -> fills at limit_price as Maker
                    # For SELL: high >= limit_price -> fills at limit_price as Maker
                    filled = False
                    if pending_order.side == OrderSide.BUY and low <= pending_order.limit_price:
                        filled = True
                    elif pending_order.side == OrderSide.SELL and high >= pending_order.limit_price:
                        filled = True

                    if filled:
                        exec_price = pending_order.limit_price
                        # Maker fill: 0.02% fee, 0.00% slippage!
                        entry_fee = pending_order.size * exec_price * self.maker_fee
                        total_fees += entry_fee
                        cash -= entry_fee
                        equity -= entry_fee
                        just_entered = True
                        active_position = ActivePosition(
                            position_id=pending_order.order_id,
                            side=PositionSide.LONG if pending_order.side == OrderSide.BUY else PositionSide.SHORT,
                            entry_price=exec_price,
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
                            total_entry_fee=entry_fee,
                            metadata=pending_order.metadata,
                        )
                        pending_order = None

            # -------------------------------------------------------------
            # STEP 2: Manage Active Position (TP1, Early BE, TP2, SL)
            # -------------------------------------------------------------
            # If order was just filled on this bar, position management starts on subsequent bars
            if active_position is not None and not just_entered:
                pos = active_position
                closed = False

                # 2A. TP1 Check (1.5R): closes exactly 50% with Maker fee (0.02%)
                if not pos.tp1_hit:
                    hit_tp1 = False
                    if pos.side == PositionSide.LONG and high >= pos.tp1:
                        hit_tp1 = True
                    elif pos.side == PositionSide.SHORT and low <= pos.tp1:
                        hit_tp1 = True

                    if hit_tp1:
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
                        tp1_just_hit_on_this_bar = True
                    else:
                        tp1_just_hit_on_this_bar = False
                else:
                    tp1_just_hit_on_this_bar = False
                # 2B. TP2 Check (3.5R): closes remaining 50% with Maker fee (0.02%)
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

                    # Total trade PnL strictly equals realized TP1 PnL + remaining exit PnL
                    total_trade_pnl = (pos.realized_tp1_pnl if pos.tp1_hit else 0.0) + pnl_rem
                    total_trade_fees = pos.total_entry_fee + (
                        pos.initial_size * 0.5 * pos.tp1 * self.maker_fee if pos.tp1_hit else 0.0
                    ) + fee_rem

                    trades.append(
                        TradeRecord(
                            trade_id=pos.position_id,
                            direction=pos.side.value,
                            entry_time=pos.entry_time,
                            exit_time=current_time,
                            entry_price=pos.entry_price,
                            exit_price=exit_px,
                            initial_size=pos.initial_size,
                            exit_size=pos.remaining_size,
                            pnl=total_trade_pnl,
                            pnl_pct=(total_trade_pnl / (pos.initial_size * pos.entry_price)) * 100.0,
                            exit_reason="TP2",
                            tp1_hit=pos.tp1_hit,
                            tp1_pnl=pos.realized_tp1_pnl,
                            remaining_pnl=pnl_rem,
                            total_fees=total_trade_fees,
                            total_slippage=0.0,
                            metadata=pos.metadata,
                        ).to_dict()
                    )
                    active_position = None
                    closed = True
                # 2C. Stop Loss Check: closes remaining position with Taker fee 0.05% + 0.05% slippage
                # Note: if TP1 was just hit on this upward/downward expansion bar, the newly adjusted
                # BE SL is effective for subsequent bars, preventing intra-bar phantom stops of pre-TP1 low.
                if not closed and active_position is not None and not tp1_just_hit_on_this_bar:
                    hit_sl = False
                    if pos.side == PositionSide.LONG and low <= pos.stop_loss:
                        hit_sl = True
                    elif pos.side == PositionSide.SHORT and high >= pos.stop_loss:
                        hit_sl = True

                    if hit_sl:
                        # Taker execution: price suffers adverse slippage
                        if pos.side == PositionSide.LONG:
                            exit_px = pos.stop_loss * (1.0 - self.slippage)
                            slip = pos.remaining_size * (pos.stop_loss - exit_px)
                            pnl_rem = pos.remaining_size * (exit_px - pos.entry_price)
                        else:
                            exit_px = pos.stop_loss * (1.0 + self.slippage)
                            slip = pos.remaining_size * (exit_px - pos.stop_loss)
                            pnl_rem = pos.remaining_size * (pos.entry_price - exit_px)

                        fee_sl = pos.remaining_size * exit_px * self.taker_fee
                        total_fees += fee_sl
                        total_slippage += slip
                        pnl_rem -= fee_sl

                        cash += pnl_rem
                        equity += pnl_rem

                        # Zero double-counting: total trade PnL = realized TP1 PnL + remaining exit PnL
                        total_trade_pnl = (pos.realized_tp1_pnl if pos.tp1_hit else 0.0) + pnl_rem
                        total_trade_fees = pos.total_entry_fee + (
                            pos.initial_size * 0.5 * pos.tp1 * self.maker_fee if pos.tp1_hit else 0.0
                        ) + fee_sl

                        exit_label = "BE_SL" if pos.early_be_triggered else "STOP_LOSS"
                        trades.append(
                            TradeRecord(
                                trade_id=pos.position_id,
                                direction=pos.side.value,
                                entry_time=pos.entry_time,
                                exit_time=current_time,
                                entry_price=pos.entry_price,
                                exit_price=exit_px,
                                initial_size=pos.initial_size,
                                exit_size=pos.remaining_size,
                                pnl=total_trade_pnl,
                                pnl_pct=(total_trade_pnl / (pos.initial_size * pos.entry_price)) * 100.0,
                                exit_reason=exit_label,
                                tp1_hit=pos.tp1_hit,
                                tp1_pnl=pos.realized_tp1_pnl,
                                remaining_pnl=pnl_rem,
                                total_fees=total_trade_fees,
                                total_slippage=slip,
                                metadata=pos.metadata,
                            ).to_dict()
                        )
                        active_position = None
                        closed = True

                # 2D. Early BE Check before TP1: if price reaches early threshold, shift SL to entry +- 0.15*atr
                if not closed and active_position is not None and not pos.early_be_triggered:
                    stop_dist = abs(pos.entry_price - pos.stop_loss)
                    # Use 0.7R as default early BE threshold if price makes headway
                    if pos.side == PositionSide.LONG and (high - pos.entry_price) >= 0.7 * stop_dist:
                        pos.stop_loss = pos.entry_price + self.early_be_buffer_atr * atr
                        pos.early_be_triggered = True
                    elif pos.side == PositionSide.SHORT and (pos.entry_price - low) >= 0.7 * stop_dist:
                        pos.stop_loss = pos.entry_price - self.early_be_buffer_atr * atr
                        pos.early_be_triggered = True

            # -------------------------------------------------------------
            # STEP 3: Signal Generation & Pending Limit Order Placement
            # -------------------------------------------------------------
            # Only place new limit order if flat (no active position and no active pending order)
            if active_position is None and pending_order is None:
                sig = signal_series[idx] if signal_series is not None else None
                if sig in ("LONG", "SHORT", 1, -1, CredalSignal.BULLISH, CredalSignal.BEARISH):
                    is_long = sig in ("LONG", 1, CredalSignal.BULLISH)

                    # 3A. Credal Uncertainty Gate Check
                    credal_allowed, credal_decision = self.evaluate_credal_gate(data.iloc[idx])
                    if not credal_allowed:
                        # Abstain from placing limit order due to high vacuity or conflict!
                        pass
                    else:
                        # 3B. Determine Limit Entry Price (Order Block / FVG equilibrium)
                        if limit_entry_series is not None and not np.isnan(limit_entry_series[idx]):
                            limit_px = float(limit_entry_series[idx])
                        else:
                            # Default limit price: place limit at slight discount/premium or OB equilibrium
                            # If equilibrium column exists, use it; else close price
                            if "Equilibrium" in data.columns and not np.isnan(data["Equilibrium"].iloc[idx]):
                                limit_px = float(data["Equilibrium"].iloc[idx])
                            elif "equilibrium" in data.columns and not np.isnan(data["equilibrium"].iloc[idx]):
                                limit_px = float(data["equilibrium"].iloc[idx])
                            else:
                                # For Long: place limit at close - 0.2 * atr
                                # For Short: place limit at close + 0.2 * atr
                                limit_px = (close - 0.2 * atr) if is_long else (close + 0.2 * atr)

                        # 3C. Determine Stop Loss and TP levels
                        if sl_series is not None and not np.isnan(sl_series[idx]):
                            sl_val = float(sl_series[idx])
                        else:
                            sl_val = (limit_px - self.sl_atr_mult * atr) if is_long else (limit_px + self.sl_atr_mult * atr)

                        stop_dist = abs(limit_px - sl_val)

                        if tp1_series is not None and not np.isnan(tp1_series[idx]):
                            tp1_val = float(tp1_series[idx])
                        else:
                            tp1_val = (limit_px + self.rr_tp1 * stop_dist) if is_long else (limit_px - self.rr_tp1 * stop_dist)
                        if tp2_series is not None and not np.isnan(tp2_series[idx]):
                            tp2_val = float(tp2_series[idx])
                        else:
                            tp2_val = (limit_px + self.rr_tp2 * stop_dist) if is_long else (limit_px - self.rr_tp2 * stop_dist)

                        # Validate stop distance
                        if stop_dist > 0.0 and stop_dist <= limit_px * 0.10:
                            order_size = self.calculate_position_size(
                                equity=cash,
                                entry_price=limit_px,
                                stop_loss_price=sl_val,
                            )

                            if order_size > 0.0:
                                order_counter += 1
                                meta = {}
                                if credal_decision is not None:
                                    meta["credal_uncertainty"] = float(credal_decision.uncertainty)
                                    meta["credal_signal"] = credal_decision.credal_signal.value

                                pending_order = PendingLimitOrder(
                                    order_id=f"ORD_{order_counter:05d}",
                                    side=OrderSide.BUY if is_long else OrderSide.SELL,
                                    limit_price=limit_px,
                                    size=order_size,
                                    stop_loss=sl_val,
                                    tp1=tp1_val,
                                    tp2=tp2_val,
                                    placed_idx=idx,
                                    placed_time=current_time,
                                    expiry_bars=self.limit_order_expiry_bars,
                                    metadata=meta,
                                )

            # -------------------------------------------------------------
            # STEP 4: Mark to Market Equity Tracking
            # -------------------------------------------------------------
            if active_position is not None:
                pos = active_position
                if pos.side == PositionSide.LONG:
                    unrealized = pos.remaining_size * (close - pos.entry_price) - (
                        pos.remaining_size * close * self.taker_fee
                    )
                else:
                    unrealized = pos.remaining_size * (pos.entry_price - close) - (
                        pos.remaining_size * close * self.taker_fee
                    )
                current_equity = cash + unrealized
                equity_curve.append(current_equity)
            else:
                equity_curve.append(cash)
            just_entered = False

        # -----------------------------------------------------------------
        # STEP 5: End-of-Data Liquidation for any open position
        # -----------------------------------------------------------------
        if active_position is not None:
            pos = active_position
            exit_px = close_arr[-1]
            fee = pos.remaining_size * exit_px * self.taker_fee
            total_fees += fee

            if pos.side == PositionSide.LONG:
                pnl_rem = pos.remaining_size * (exit_px - pos.entry_price) - fee
            else:
                pnl_rem = pos.remaining_size * (pos.entry_price - exit_px) - fee

            cash += pnl_rem
            equity += pnl_rem
            total_trade_pnl = (pos.realized_tp1_pnl if pos.tp1_hit else 0.0) + pnl_rem
            total_trade_fees = pos.total_entry_fee + (
                pos.initial_size * 0.5 * pos.tp1 * self.maker_fee if pos.tp1_hit else 0.0
            ) + fee

            trades.append(
                TradeRecord(
                    trade_id=pos.position_id,
                    direction=pos.side.value,
                    entry_time=pos.entry_time,
                    exit_time=timestamps[-1],
                    entry_price=pos.entry_price,
                    exit_price=exit_px,
                    initial_size=pos.initial_size,
                    exit_size=pos.remaining_size,
                    pnl=total_trade_pnl,
                    pnl_pct=(total_trade_pnl / (pos.initial_size * pos.entry_price)) * 100.0,
                    exit_reason="END_OF_DATA",
                    tp1_hit=pos.tp1_hit,
                    tp1_pnl=pos.realized_tp1_pnl,
                    remaining_pnl=pnl_rem,
                    total_fees=total_trade_fees,
                    total_slippage=0.0,
                    metadata=pos.metadata,
                ).to_dict()
            )
            equity_curve[-1] = cash
            active_position = None

        # Calculate final metrics
        metrics = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps,
            total_fees=total_fees,
            total_slippage=total_slippage,
            final_cash=cash,
        )

        return {
            "trades": trades,
            "equity_curve": [float(x) for x in equity_curve],
            "metrics": metrics,
        }

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        timestamps: Any,
        total_fees: float,
        total_slippage: float,
        final_cash: float,
    ) -> Dict[str, Any]:
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
        ann_factor = np.sqrt(252 * 24 * 4)  # 15m annualized factor
        sharpe = float(np.mean(rets) / std_ret * ann_factor) if std_ret > 1e-9 else 0.0

        total_pnl = final_cash - self.initial_cash
        ret_pct = (total_pnl / self.initial_cash) * 100.0

        days_span = 1.0
        if len(timestamps) > 1 and hasattr(timestamps[0], "total_seconds"):
            try:
                days_span = max((timestamps[-1] - timestamps[0]).total_seconds() / 86400.0, 1.0)
            except Exception:
                days_span = max(len(timestamps) / (24 * 4), 1.0)
        else:
            days_span = max(len(timestamps) / (24 * 4), 1.0)

        daily_trades = len(trades) / days_span

        return {
            "initial_cash": float(self.initial_cash),
            "final_cash": float(final_cash),
            "total_pnl": float(total_pnl),
            "total_return_pct": float(ret_pct),
            "total_trades": len(trades),
            "daily_trades": float(daily_trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate_pct": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown_pct": float(max_dd),
            "sharpe_ratio": float(sharpe),
            "total_fees": float(total_fees),
            "total_slippage_cost": float(total_slippage),
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            "initial_cash": float(self.initial_cash),
            "final_cash": float(self.initial_cash),
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "daily_trades": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
        }
