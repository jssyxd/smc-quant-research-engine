"""
SMC V2 - 波动率 Regime 版
在 V1 基础上增加 ATR 相对波动率过滤（过高/过低暂停开仓）
"""
from .indicators import SMCConfig, SMCIndicators, compute_smc_indicators, get_signals, TrendState
from .smc_strategy import SMCStrategy, BacktestResult, Trade, run_smc_backtest

__all__ = [
    "SMCConfig", "SMCIndicators", "compute_smc_indicators", "get_signals", "TrendState",
    "SMCStrategy", "BacktestResult", "Trade", "run_smc_backtest"
]
