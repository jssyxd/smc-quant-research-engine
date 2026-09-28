"""
SMC V3 - 趋势强度二次确认版
在 V1 基础上增加 EMA200 慢趋势过滤（做多需在 EMA200 上方，做空需在下方）
"""
from .indicators import SMCConfig, SMCIndicators, compute_smc_indicators, get_signals, TrendState
from .smc_strategy import SMCStrategy, BacktestResult, Trade, run_smc_backtest

__all__ = [
    "SMCConfig", "SMCIndicators", "compute_smc_indicators", "get_signals", "TrendState",
    "SMCStrategy", "BacktestResult", "Trade", "run_smc_backtest"
]
