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

Stock Split Handling:
    fetch_split_data - Fetch stock split data from Tiingo EOD API
    adjust_for_splits - Adjust OHLCV data for stock splits
    validate_and_adjust_cache - Validate cache against recent splits

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
from .stock_split import fetch_split_data, adjust_for_splits, validate_and_adjust_cache

__all__ = [
    "TiingoService",
    "TiingoCache",
    "TiingoAPI",
    "filter_to_market_hours",
    "format_df_dates",
    "OHLCBar",
    "fetch_split_data",
    "adjust_for_splits",
    "validate_and_adjust_cache",
]
