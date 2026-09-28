import sys, os
from decimal import Decimal
import pandas as pd
import numpy as np

import nautilus_trader
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, OrderSide, TimeInForce
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

instrument_id = InstrumentId(Symbol('BTCUSDT'), Venue('BINANCE'))
instrument = CurrencyPair(
    instrument_id=instrument_id,
    raw_symbol=Symbol('BTCUSDT'),
    base_currency=Currency.from_str('BTC'),
    quote_currency=Currency.from_str('USDT'),
    price_precision=2,
    size_precision=6,
    price_increment=Price.from_str('0.01'),
    size_increment=Quantity.from_str('0.000001'),
    lot_size=Quantity.from_str('0.000001'),
    max_quantity=Quantity.from_str('10000.0'),
    min_quantity=Quantity.from_str('0.000001'),
    max_price=Price.from_str('1000000.0'),
    min_price=Price.from_str('0.01'),
    margin_init=Decimal('0'),
    margin_maint=Decimal('0'),
    maker_fee=Decimal('0.0002'),
    taker_fee=Decimal('0.0005'),
    ts_event=0,
    ts_init=0
)

engine_config = BacktestEngineConfig(
    trader_id="TESTER-001",
    logging=LoggingConfig(log_level="ERROR"),
)
engine = BacktestEngine(config=engine_config)
engine.add_venue(
    venue=Venue("BINANCE"),
    oms_type=nautilus_trader.model.enums.OmsType.NETTING,
    account_type=nautilus_trader.model.enums.AccountType.MARGIN,
    base_currency=Currency.from_str("USDT"),
    starting_balances=[Money(1000.0, Currency.from_str("USDT"))],
)
engine.add_instrument(instrument)

bar_spec = BarSpecification(1, BarAggregation.HOUR, PriceType.LAST)
bar_type = BarType(instrument_id, bar_spec)

df = pd.read_parquet('data/raw_extracted/BTC/BTCUSDT_1h_ohlcv_2024-01.parquet')
bars = []
for ts, row in df.iterrows():
    ts_ns = int(ts.timestamp() * 1_000_000_000)
    bar = Bar(
        bar_type=bar_type,
        open=Price(row['open'], precision=2),
        high=Price(row['high'], precision=2),
        low=Price(row['low'], precision=2),
        close=Price(row['close'], precision=2),
        volume=Quantity(row['volume'], precision=6),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )
    bars.append(bar)

engine.add_data(bars)

class DummyStrategy(Strategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.count = 0
    def on_start(self):
        self.subscribe_bars(bar_type)
    def on_bar(self, bar: Bar):
        self.count += 1
        if self.count == 10:
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_str("0.010000"),
            )
            self.submit_order(order)

strategy = DummyStrategy()
engine.add_strategy(strategy)
engine.run()

print("Engine run finished! Bars processed:", strategy.count)
fills = engine.trader.generate_order_fills_report()
print("Fills count:", len(fills))
print(fills.head(2))
positions = engine.trader.generate_positions_report()
print("Positions count:", len(positions))
print(positions.head(2))
account = engine.portfolio.account(Venue("BINANCE"))
print("Balances:", account.balances())
