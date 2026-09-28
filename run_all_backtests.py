import sys, os, gc, json, time, math
from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np

# Create output directories
os.makedirs("results/trades", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)
os.makedirs("results/curves", exist_ok=True)

SYMBOLS = ['BTC', 'BNB', 'SOL', 'UNIUSDT', 'NEAR', 'XAUUSD', 'USOUSD', 'GBPUSD']
TIMEFRAMES = ['15m', '1h']
STRATEGIES = ['v1_internal_factors', 'v2_volatility_regime', 'v3_trend_strength']
ENGINES = ['QuantCell', 'NautilusTrader']

INITIAL_CASH = 1000.0
MAKER_FEE = 0.0002   # 0.02%
TAKER_FEE = 0.0005   # 0.05%
SLIPPAGE = 0.0005    # 0.05%
RISK_PCT = 2.0

print(f"Starting Multi-Asset SMC Backtest Suite at {datetime.now().isoformat()}")
print(f"Symbols: {SYMBOLS}")
print(f"Timeframes: {TIMEFRAMES}")
print(f"Strategies: {STRATEGIES}")
print(f"Engines: {ENGINES}")
print(f"Initial Cash: ${INITIAL_CASH}, Maker: {MAKER_FEE*100}%, Taker: {TAKER_FEE*100}%, Slippage: {SLIPPAGE*100}%")

