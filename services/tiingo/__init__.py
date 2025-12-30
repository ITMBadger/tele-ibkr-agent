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
    format_df_dates - Format DataFrame dates to naive ET strings

Types:
    OHLCBar - OHLC bar type

Note: For datetime utilities, use services.time_centralize_utils directly.

Usage (live trading):
    from services.tiingo import TiingoService

    tiingo = TiingoService()
    bars = await tiingo.get_ohlc("QQQ", days=5, interval="5min")
    await tiingo.close()
"""

from .api import TiingoAPI
from .cache import TiingoCache
from .filters import filter_to_market_hours, format_df_dates
from .service import OHLCBar, TiingoService

__all__ = [
    "TiingoService",
    "TiingoCache",
    "TiingoAPI",
    "filter_to_market_hours",
    "format_df_dates",
    "OHLCBar",
]
