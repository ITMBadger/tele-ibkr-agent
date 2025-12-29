# services/tiingo/service.py - High-level Tiingo service.
"""
High-level Tiingo service combining cache and API.

This is the main interface for live trading. It handles:
- Two-call cache strategy (historical cached, today fresh)
- Market hours filtering for intraday data
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
from .filters import filter_to_market_hours


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

        Args:
            cache_dir: Cache directory (defaults to data/cache/)
            api_key: Tiingo API key (defaults to env var)
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
    ) -> pd.DataFrame:
        """
        Two-call cache strategy for OHLC data.

        Strategy:
            1. Historical (days ago -> yesterday): Cached permanently
            2. Today: Always fresh, never cached
            3. Merge both results

        Args:
            symbol: Stock symbol or crypto ticker (e.g., "QQQ" or "BTCUSD")
            days: Number of days of history
            interval: Bar interval (e.g., "1min", "5min")
            use_cache: Whether to use cache for historical data

        Returns:
            Combined DataFrame
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        start_date = today - timedelta(days=days)

        all_dfs: list[pd.DataFrame] = []

        # Detect if this is a crypto symbol
        is_crypto = self._is_crypto_symbol(symbol)

        # ===== CALL 1: Historical data (start_date to yesterday) =====
        historical_start = start_date.strftime("%Y%m%d")
        historical_end = yesterday.strftime("%Y%m%d")
        cache_path = self.cache.get_path(
            symbol, interval, historical_start, historical_end
        )

        historical_df = None
        if use_cache:
            historical_df = self.cache.load(cache_path)

        if historical_df is None:
            # Cache miss - fetch from API
            if is_crypto:
                historical_df = await self.api.fetch_crypto_intraday(
                    symbol, start_date, yesterday, interval=interval
                )
            else:
                historical_df = await self.api.fetch_intraday(
                    symbol, start_date, yesterday, interval=interval
                )

            # Save to cache (historical data doesn't change)
            if use_cache and historical_df is not None and not historical_df.empty:
                self.cache.save(cache_path, historical_df)

        if historical_df is not None and not historical_df.empty:
            all_dfs.append(historical_df)

        # ===== CALL 2: Today's data (always fresh, no cache) =====
        if is_crypto:
            today_df = await self.api.fetch_crypto_intraday(
                symbol, today, datetime.now(), interval=interval
            )
        else:
            today_df = await self.api.fetch_intraday(
                symbol, today, datetime.now(), interval=interval
            )

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

        Args:
            symbol: Stock symbol (e.g., "QQQ") or crypto ticker (e.g., "BTCUSD")
            days: Number of days of history (including today)
            interval: Bar interval ("1min", "5min", "15min", "30min", "1hour")
            use_cache: Whether to use cache (default True)

        Returns:
            List of OHLC bars, oldest first
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
            df = filter_to_market_hours(df)
            if df.empty:
                return []

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

        Args:
            symbol: Stock symbol
            days: Number of days of history
            interval: Bar interval
            use_cache: Whether to use cache (default True)

        Returns:
            List of closing prices, oldest first
        """
        bars = await self.get_ohlc(symbol, days, interval, use_cache)
        return [bar["close"] for bar in bars]

    async def get_current_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        This method does NOT use cache as it's real-time data.
        Tries IEX real-time endpoint first, falls back to OHLC data.

        Args:
            symbol: Stock symbol

        Returns:
            Latest price

        Raises:
            Exception: If no price data available
        """
        try:
            return await self.api.fetch_current_price(symbol)
        except Exception:
            # Fallback to OHLC endpoint
            bars = await self.get_ohlc(symbol, days=1, interval="5min")
            if bars:
                return bars[-1]["close"]
            raise Exception(f"No price data for {symbol}")
