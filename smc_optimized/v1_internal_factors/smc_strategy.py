"""
SMC 策略模块 - V1 内部因子优化版

主要改动:
1. 使用优化后的入场阈值 (Score≥70/72 + Factors≥4)
2. Early BE 触发后，止损移到 entry ± early_be_buffer_atr * ATR（锁定少量利润）
3. 信号判断走 get_signals 的非对称逻辑

基于 DeFiers-SMC Strategy v0.4.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .indicators import (
    SMCConfig, compute_smc_indicators, get_signals
)

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: TradeDirection
    size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: Optional[str] = None
    score: float = 0.0
    factors: int = 0


@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[float]
    initial_cash: float
    final_cash: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float

    def summary(self) -> Dict:
        return {
            'initial_cash': self.initial_cash,
            'final_cash': self.final_cash,
            'total_return': (self.final_cash - self.initial_cash) / self.initial_cash * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate * 100,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': self.sharpe_ratio,
        }


class SMCStrategy:
    """
    SMC 交易策略 - V1

    - 非对称多空入场
    - Early BE 带缓冲
    """

    def __init__(
        self,
        config: Optional[SMCConfig] = None,
        initial_cash: float = 10000.0,
        commission: float = 0.0005,
        risk_pct: float = 2.0,
    ):
        self.config = config or SMCConfig()
        self.initial_cash = initial_cash
        self.commission = commission
        self.risk_pct = risk_pct

        self.position = TradeDirection.FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time: Optional[pd.Timestamp] = None
        self.planned_sl = 0.0
        self.take_profit_1 = 0.0
        self.take_profit_2 = 0.0
        self.tp1_hit = False
        self.early_be_triggered = False

        self.cash = initial_cash
        self.equity_curve = [initial_cash]
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.data: Optional[pd.DataFrame] = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.data = compute_smc_indicators(df, self.config)
        self.data = get_signals(self.data, self.config)
        logger.info(f"数据准备完成: {len(self.data)} 条数据 (V1 优化版)")
        return self.data

    def compute_position_size(self, stop_distance: float) -> float:
        if stop_distance <= 0:
            return 0.0
        risk_amount = self.cash * (self.risk_pct / 100.0)
        size = risk_amount / stop_distance
        max_stop_pct = 5.0
        max_stop_distance = self.data['Close'].iloc[-1] * (max_stop_pct / 100.0)
        if stop_distance > max_stop_distance:
            return 0.0
        return size

    def compute_sl_tp(self, idx: int, direction: TradeDirection) -> Tuple[float, float, float]:
        atr = self.data['ATR'].iloc[idx]
        close = self.data['Close'].iloc[idx]
        buffer = self.config.sl_buffer_atr * atr
        hh = self.data['HH'].iloc[idx]
        ll = self.data['LL'].iloc[idx]

        if direction == TradeDirection.LONG:
            sl = (ll if not np.isnan(ll) else close) - buffer
            stop_distance = close - sl
            tp1 = close + self.config.rr_tp1 * stop_distance
            tp2 = close + self.config.rr_tp2 * stop_distance
        else:
            sl = (hh if not np.isnan(hh) else close) + buffer
            stop_distance = sl - close
            tp1 = close - self.config.rr_tp1 * stop_distance
            tp2 = close - self.config.rr_tp2 * stop_distance
        return sl, tp1, tp2

    def run_backtest(self) -> BacktestResult:
        if self.data is None:
            raise ValueError("请先调用 prepare_data()")

        n = len(self.data)
        for idx in range(n):
            self._process_bar(idx)

        if self.position != TradeDirection.FLAT:
            self._close_trade(self.data.index[-1], self.data['Close'].iloc[-1], "END_OF_DATA")

        return self._generate_result()

    def _process_bar(self, idx: int):
        row = self.data.iloc[idx]
        timestamp = self.data.index[idx]
        close = row['Close']
        high = row['High']
        low = row['Low']
        atr = row['ATR']

        if self.position != TradeDirection.FLAT:
            # TP1
            if not self.tp1_hit:
                if self.position == TradeDirection.LONG and high >= self.take_profit_1:
                    self._partial_close(timestamp, self.take_profit_1, "TP1")
                elif self.position == TradeDirection.SHORT and low <= self.take_profit_1:
                    self._partial_close(timestamp, self.take_profit_1, "TP1")

            # TP2
            if self.position == TradeDirection.LONG and high >= self.take_profit_2:
                self._close_trade(timestamp, self.take_profit_2, "TP2")
            elif self.position == TradeDirection.SHORT and low <= self.take_profit_2:
                self._close_trade(timestamp, self.take_profit_2, "TP2")

            # Stop Loss
            if self.position == TradeDirection.LONG and low <= self.planned_sl:
                self._close_trade(timestamp, self.planned_sl, "STOP_LOSS")
            elif self.position == TradeDirection.SHORT and high >= self.planned_sl:
                self._close_trade(timestamp, self.planned_sl, "STOP_LOSS")

            # Early BE with buffer  ★ V1 核心
            if (self.config.use_early_be and not self.early_be_triggered
                    and self.position != TradeDirection.FLAT):
                entry_price = self.entry_price
                stop_dist = abs(entry_price - self.planned_sl)
                buffer = self.config.early_be_buffer_atr * atr

                if self.position == TradeDirection.LONG:
                    mfe = high - entry_price
                    trigger = self.config.early_be_rr * stop_dist
                    if mfe >= trigger:
                        # 不是精确保本，而是锁定一点利润
                        self.planned_sl = entry_price + buffer
                        self.early_be_triggered = True
                elif self.position == TradeDirection.SHORT:
                    mfe = entry_price - low
                    trigger = self.config.early_be_rr * stop_dist
                    if mfe >= trigger:
                        self.planned_sl = entry_price - buffer
                        self.early_be_triggered = True

        # 开仓信号（使用 get_signals 已写好的 Signal 列）
        if self.position == TradeDirection.FLAT:
            signal = row.get('Signal', 'NEUTRAL')
            if signal == 'LONG':
                self._open_position(idx, TradeDirection.LONG, row['ScoreLong'], int(row['FactorsLong']))
            elif signal == 'SHORT':
                self._open_position(idx, TradeDirection.SHORT, row['ScoreShort'], int(row['FactorsShort']))

        # 权益
        if self.position != TradeDirection.FLAT:
            if self.position == TradeDirection.LONG:
                unrealized = self.position_size * (close - self.entry_price) - self.position_size * close * self.commission
            else:
                unrealized = self.position_size * (self.entry_price - close) - self.position_size * close * self.commission
            current_equity = self.cash + unrealized
        else:
            current_equity = self.cash
        self.equity_curve.append(current_equity)

    def _open_position(self, idx: int, direction: TradeDirection, score: float, factors: int):
        close = self.data['Close'].iloc[idx]
        timestamp = self.data.index[idx]
        sl, tp1, tp2 = self.compute_sl_tp(idx, direction)

        if direction == TradeDirection.LONG:
            stop_distance = close - sl
        else:
            stop_distance = sl - close

        size = self.compute_position_size(stop_distance)
        if size <= 0:
            return

        self.position = direction
        self.position_size = size
        self.entry_price = close
        self.entry_time = timestamp
        self.planned_sl = sl
        self.take_profit_1 = tp1
        self.take_profit_2 = tp2
        self.tp1_hit = False
        self.early_be_triggered = False

        cost = size * close * self.commission
        self.cash -= cost

        self.current_trade = Trade(
            entry_time=timestamp,
            entry_price=close,
            direction=direction,
            size=size,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            score=score,
            factors=factors,
        )
        logger.debug(f"开仓: {direction.name} @ {close:.2f}, SL={sl:.2f}, TP1={tp1:.2f}, TP2={tp2:.2f}")

    def _partial_close(self, timestamp: pd.Timestamp, price: float, reason: str):
        close_size = self.position_size * (self.config.tp1_qty_pct / 100.0)
        if self.position == TradeDirection.LONG:
            pnl = close_size * (price - self.entry_price)
        else:
            pnl = close_size * (self.entry_price - price)

        cost = close_size * price * self.commission
        pnl -= cost
        self.cash += pnl
        self.position_size -= close_size
        self.tp1_hit = True
        self.tp1_pnl = pnl

        # TP1 后移到带缓冲的保本
        atr = self.data.loc[timestamp, 'ATR'] if timestamp in self.data.index else 0
        buffer = self.config.early_be_buffer_atr * atr if atr else 0
        if self.position == TradeDirection.LONG:
            self.planned_sl = self.entry_price + buffer
        else:
            self.planned_sl = self.entry_price - buffer
        self.early_be_triggered = True

        logger.debug(f"TP1 部分平仓: {reason} @ {price:.2f}, PnL={pnl:.2f}, 剩余仓位={self.position_size:.4f}")

    def _close_trade(self, timestamp: pd.Timestamp, price: float, reason: str):
        if self.position == TradeDirection.FLAT:
            return

        rem_size = self.position_size
        if self.position == TradeDirection.LONG:
            pnl = rem_size * (price - self.entry_price)
        else:
            pnl = rem_size * (self.entry_price - price)

        cost = rem_size * price * self.commission
        pnl -= cost
        self.cash += pnl

        total_pnl = (getattr(self, 'tp1_pnl', 0.0) if self.tp1_hit else 0.0) + pnl
        if self.current_trade:
            self.current_trade.exit_time = timestamp
            self.current_trade.exit_price = price
            self.current_trade.pnl = total_pnl
            self.current_trade.pnl_pct = total_pnl / (self.current_trade.size * self.entry_price) * 100
            self.current_trade.exit_reason = reason
            self.trades.append(self.current_trade)

        logger.debug(f"平仓: {reason} @ {price:.2f}, 总PnL={total_pnl:.2f}")

        self.position = TradeDirection.FLAT
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.current_trade = None
        self.early_be_triggered = False
        self.tp1_hit = False
        self.tp1_pnl = 0.0
    def _generate_result(self) -> BacktestResult:
        if not self.trades:
            return BacktestResult(
                trades=[], equity_curve=self.equity_curve,
                initial_cash=self.initial_cash, final_cash=self.cash,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, profit_factor=0.0,
            )

        winning = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing = [t for t in self.trades if t.pnl and t.pnl <= 0]
        win_rate = len(winning) / len(self.trades) if self.trades else 0.0

        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        returns = np.diff(equity) / equity[:-1]
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)
                  if len(returns) > 0 and np.std(returns) > 0 else 0.0)

        total_win = sum(t.pnl for t in winning) if winning else 0.0
        total_loss = abs(sum(t.pnl for t in losing)) if losing else 0.0
        profit_factor = total_win / total_loss if total_loss > 0 else 0.0

        return BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_cash=self.initial_cash,
            final_cash=self.cash,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_pnl=self.cash - self.initial_cash,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
        )


def run_smc_backtest(
    df: pd.DataFrame,
    config: Optional[SMCConfig] = None,
    initial_cash: float = 10000.0,
    commission: float = 0.0005,
    risk_pct: float = 2.0,
) -> BacktestResult:
    strategy = SMCStrategy(config, initial_cash, commission, risk_pct)
    strategy.prepare_data(df)
    return strategy.run_backtest()
