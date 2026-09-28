"""
SMC V1 - 内部因子优化版
Score ≥ 70/72 + Factors ≥ 4，做多更严格，Early BE 带缓冲
"""
from .indicators import SMCConfig, SMCIndicators, compute_smc_indicators, get_signals, TrendState
from .smc_strategy import SMCStrategy, BacktestResult, Trade, run_smc_backtest

__all__ = [
    "SMCConfig", "SMCIndicators", "compute_smc_indicators", "get_signals", "TrendState",
    "SMCStrategy", "BacktestResult", "Trade", "run_smc_backtest"
]
