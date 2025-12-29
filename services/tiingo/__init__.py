# services/tiingo/__init__.py - Public exports for Tiingo package.
"""
Tiingo data service package.

Provides market data fetching with caching support.

Main classes:
    TiingoService - High-level service for live trading (with cache)
    TiingoCache - Low-level cache operations
    TiingoAPI - Low-level API calls

Utilities:
    filter_to_market_hours - Filter to NYSE market hours

Types:
    OHLCBar - OHLC bar type

Usage (live trading):
    from services.tiingo import TiingoService

    tiingo = TiingoService()
    bars = await tiingo.get_ohlc("QQQ", days=5, interval="5min")
    await tiingo.close()

Usage (backtest with custom cache):
    from services.tiingo import TiingoCache, TiingoAPI, filter_to_market_hours

    cache = TiingoCache("data/backtest/ohlc")
    api = TiingoAPI()
    ...
"""

from .api import TiingoAPI
from .cache import TiingoCache
from .filters import (
    filter_to_market_hours,
    ET_DATETIME_FORMAT,
    ET_TZ,
    to_naive_et,
    format_et,
    format_df_dates,
)
from .service import OHLCBar, TiingoService

__all__ = [
    # High-level service
    "TiingoService",
    # Low-level components (for backtest/custom use)
    "TiingoCache",
    "TiingoAPI",
    # Market hours filtering
    "filter_to_market_hours",
    # Datetime utilities
    "ET_DATETIME_FORMAT",
    "ET_TZ",
    "to_naive_et",
    "format_et",
    "format_df_dates",
    # Types
    "OHLCBar",
]
