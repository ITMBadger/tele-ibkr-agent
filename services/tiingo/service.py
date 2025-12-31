# services/tiingo/service.py - High-level Tiingo service.
"""
High-level Tiingo service combining cache and API.

This is the main interface for live trading. It handles:
- Two-call cache strategy (historical cached, today fresh)
- Market hours filtering for intraday data
- Stock split adjustment for accurate historical prices
- TypedDict return types for type safety

Architecture (3-layer):
    Layer 1: cache.py - Pure cache operations
    Layer 2: api.py - Raw API calls
    Layer 3: service.py (this file) - High-level orchestration
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pandas as pd

from .api import TiingoAPI
from .cache import TiingoCache
from .filters import filter_to_market_hours, format_df_dates
from .stock_split import fetch_split_data, adjust_for_splits, validate_and_adjust_cache
from services.time_centralize_utils import get_et_now, get_utc_now


# Default cache directory for live trading
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class OHLCBar(TypedDict):
    """Single OHLC bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TiingoService:
    """
    High-level Tiingo service for live trading.

    Combines cache and API with two-call strategy:
    - Historical data (days ago -> yesterday): Cached permanently
    - Today's data: Always fresh, never cached

    Usage:
        tiingo = TiingoService()

        # OHLC data (cached)
        bars = await tiingo.get_ohlc("QQQ", days=5, interval="5min")

        # Current price (no cache)
        price = await tiingo.get_current_price("QQQ")

        # Cleanup
        await tiingo.close()
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize service.

        """
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache = TiingoCache(cache_dir)
        self.api = TiingoAPI(api_key)

    async def close(self) -> None:
        """Close the API session."""
        await self.api.close()

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto ticker (e.g., BTCUSD, ETHUSD)."""
        symbol_upper = symbol.upper()
        # Crypto symbols typically end with USD or USDT
        return symbol_upper.endswith("USD") or symbol_upper.endswith("USDT")

    async def _fetch_with_cache(
        self,
        symbol: str,
        days: int,
        interval: str,
        use_cache: bool = True,
        check_splits: bool = True,
    ) -> pd.DataFrame:
        """
        Two-call cache strategy for OHLC data.

        Strategy:
            1. Historical (days ago -> yesterday): Cached permanently
            2. Today: Always fresh, never cached
            3. Merge both results

        Stock splits are handled by:
            - Validating cache against recent splits (invalidate if split occurred)
            - Adjusting fresh data for splits before caching

        """
        # Use ET time for date boundaries and cache keys
        et_now = get_et_now()
        today_et = et_now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
        yesterday_et = today_et - timedelta(days=1)
        start_date_et = today_et - timedelta(days=days)

        all_dfs: list[pd.DataFrame] = []

        # Detect if this is a crypto symbol (crypto doesn't have splits)
        is_crypto = self._is_crypto_symbol(symbol)

        # For crypto API: Convert ET to UTC (Tiingo crypto expects UTC timestamps)
        # For stock API: Only uses YYYY-MM-DD dates, so timezone doesn't matter
        if is_crypto:
            utc_now = get_utc_now()
            today_utc = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_utc = today_utc - timedelta(days=1)
            start_date_utc = today_utc - timedelta(days=days)

        # ===== CALL 1: Historical data (start_date to yesterday) =====
        # Cache keys use ET dates for consistency
        historical_start = start_date_et.strftime("%Y%m%d")
        historical_end = yesterday_et.strftime("%Y%m%d")
        cache_path = self.cache.get_path(
            symbol, interval, historical_start, historical_end
        )

        historical_df = None
        cache_valid = True

        if use_cache:
            historical_df = self.cache.load(cache_path)

            # Validate cache against stock splits (only for stocks, not crypto)
            if historical_df is not None and not historical_df.empty and check_splits and not is_crypto:
                cache_valid, historical_df = await validate_and_adjust_cache(
                    cache_path, symbol, historical_df, self.api, verbose=True
                )
                if not cache_valid:
                    # Cache invalidated due to split - delete and refetch
                    print(f"   [Cache] Deleting split-invalidated cache: {cache_path.name}")
                    cache_path.unlink(missing_ok=True)
                    historical_df = None

        if historical_df is None:
            # Cache miss - fetch from API
            if is_crypto:
                # Crypto API expects UTC
                historical_df = await self.api.fetch_crypto_intraday(
                    symbol, start_date_utc, yesterday_utc, interval=interval
                )
            else:
                # Stock API uses dates only (no time component)
                historical_df = await self.api.fetch_intraday(
                    symbol, start_date_et, yesterday_et, interval=interval
                )

                # Apply stock split adjustments to fresh data (stocks only)
                if historical_df is not None and not historical_df.empty and check_splits:
                    splits_df = await fetch_split_data(
                        self.api, symbol, start_date_et, yesterday_et, verbose=True
                    )
                    if not splits_df.empty:
                        historical_df = adjust_for_splits(historical_df, splits_df, verbose=True)

            # Save to cache (historical data doesn't change)
            if use_cache and historical_df is not None and not historical_df.empty:
                self.cache.save(cache_path, historical_df)

        if historical_df is not None and not historical_df.empty:
            all_dfs.append(historical_df)

        # ===== CALL 2: Today's data (always fresh, no cache) =====
        if is_crypto:
            # Crypto API expects UTC
            today_df = await self.api.fetch_crypto_intraday(
                symbol, today_utc, utc_now, interval=interval
            )
        else:
            # Stock API uses dates only
            now_et_naive = et_now.replace(tzinfo=None)
            today_df = await self.api.fetch_intraday(
                symbol, today_et, now_et_naive, interval=interval
            )

            # Note: Today's data typically doesn't need split adjustment
            # as splits are applied at market open and are reflected immediately

        if today_df is not None and not today_df.empty:
            all_dfs.append(today_df)

        # Combine results
        if not all_dfs:
            return pd.DataFrame()

        return pd.concat(all_dfs, ignore_index=True)

    async def get_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """
        Fetch OHLC data.

        Uses two-call cache strategy:
        - Historical (days ago -> yesterday): Cached permanently
        - Today: Always fresh

        Data is filtered to NYSE market hours for stocks only.
        Crypto (24/7 markets) is NOT filtered.

        """
        df = await self._fetch_with_cache(
            symbol=symbol,
            days=days,
            interval=interval,
            use_cache=use_cache,
        )

        if df.empty:
            return []

        # Filter to market hours only for stocks (not crypto)
        is_crypto = self._is_crypto_symbol(symbol)
        if not is_crypto:
            # Stocks: filter to market hours (also converts UTC → ET)
            df = filter_to_market_hours(df)
            if df.empty:
                return []
        else:
            # Crypto: convert UTC timestamps to naive ET format (no market hours filter)
            df = format_df_dates(df, date_column="date")

        # Convert to list of OHLCBar TypedDicts
        bars = []
        for _, row in df.iterrows():
            bars.append(
                OHLCBar(
                    date=str(row["date"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row.get("volume", 0)),
                )
            )
        return bars

    async def get_closes(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[float]:
        """
        Get just the closing prices.

        """
        bars = await self.get_ohlc(symbol, days, interval, use_cache)
        return [bar["close"] for bar in bars]

    async def get_current_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        This method does NOT use cache as it's real-time data.
        Tries IEX real-time endpoint first, falls back to OHLC data.

        """
        try:
            return await self.api.fetch_current_price(symbol)
        except Exception:
            # Fallback to OHLC endpoint
            bars = await self.get_ohlc(symbol, days=1, interval="1min")
            if bars:
                return bars[-1]["close"]
            raise Exception(f"No price data for {symbol}")
