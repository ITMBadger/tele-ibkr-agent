# market_data.py - Abstract market data provider interface.
"""
This module defines a common interface for market data providers.
Allows strategies to work with different data sources (Tiingo, Hyperliquid, etc.)
without code changes.

Usage:
    # For stocks (IBKR)
    data = TiingoDataProvider(tiingo_service)

    # For crypto (Hyperliquid)
    data = HyperliquidDataProvider()

    # Same interface for both
    bars = await data.get_daily_ohlc("SYMBOL", days=250)
    bars = await data.get_intraday_ohlc("SYMBOL", days=5, interval="5min")
    price = await data.get_current_price("SYMBOL")
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class OHLCBar(TypedDict):
    """Single OHLC bar (daily or intraday)."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def get_daily_ohlc(
        self,
        symbol: str,
        days: int = 250,
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """Fetch daily OHLC data."""
        pass

    @abstractmethod
    async def get_intraday_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """Fetch intraday OHLC data."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass

    async def close(self) -> None:
        """Close any open connections."""
        pass


class TiingoDataProvider(MarketDataProvider):
    """Market data provider using Tiingo (for stocks)."""

    def __init__(self, tiingo_service):
        """
        Args:
            tiingo_service: TiingoService instance
        """
        self.tiingo = tiingo_service

    async def get_daily_ohlc(
        self,
        symbol: str,
        days: int = 250,
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        return await self.tiingo.get_daily_ohlc(symbol, days, use_cache)

    async def get_intraday_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        return await self.tiingo.get_intraday_ohlc(symbol, days, interval, use_cache)

    async def get_current_price(self, symbol: str) -> float:
        return await self.tiingo.get_current_price(symbol)

    async def close(self) -> None:
        await self.tiingo.close()


class HyperliquidDataProvider(MarketDataProvider):
    """
    Market data provider using Hyperliquid API (for crypto perpetuals).

    Uses Hyperliquid candles_snapshot endpoint for OHLC data.
    """

    # Interval mapping: our format -> Hyperliquid format
    INTERVAL_MAP = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "1hour": "1h",
        "1h": "1h",
        "4hour": "4h",
        "4h": "4h",
        "1day": "1d",
        "1d": "1d",
    }

    def __init__(self):
        """Initialize Hyperliquid data provider."""
        self._info = None

    def _get_info(self):
        """Get or create Hyperliquid Info client."""
        if self._info is None:
            import os
            from hyperliquid.info import Info
            from hyperliquid.utils import constants

            testnet = os.getenv("HYPERLIQUID_TESTNET", "true").lower() == "true"
            base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
            self._info = Info(base_url, skip_ws=True)
        return self._info

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to Hyperliquid format (BTC, ETH, etc.)."""
        symbol = symbol.upper()
        # Remove common suffixes
        for suffix in ["USDT", "USD", "PERP", "-PERP", "_PERP"]:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
                break
        # Remove separators
        symbol = symbol.replace("/", "").replace("-", "").replace("_", "")
        return symbol

    async def get_daily_ohlc(
        self,
        symbol: str,
        days: int = 250,
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """
        Fetch daily OHLC from Hyperliquid.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            days: Number of days of history
            use_cache: Ignored for now
        """
        return await self._fetch_candles(symbol, "1d", days)

    async def get_intraday_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """
        Fetch intraday OHLC from Hyperliquid.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            days: Number of days of history
            interval: Bar interval ("1min", "5min", etc.)
            use_cache: Ignored for now
        """
        hl_interval = self.INTERVAL_MAP.get(interval, "5m")

        # Calculate number of candles based on days and interval
        minutes_per_bar = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }
        mpb = minutes_per_bar.get(hl_interval, 5)
        limit = min((days * 24 * 60) // mpb, 5000)  # Hyperliquid limit

        return await self._fetch_candles(symbol, hl_interval, limit)

    async def _fetch_candles(
        self,
        symbol: str,
        interval: str,
        limit: int
    ) -> list[OHLCBar]:
        """
        Fetch candle data from Hyperliquid.

        Uses sync client in async context via thread pool.
        """
        import asyncio
        import time

        info = self._get_info()
        symbol = self._normalize_symbol(symbol)

        # Calculate time range
        end_time = int(time.time() * 1000)  # Current time in ms

        # Calculate start time based on interval and limit
        interval_ms = {
            "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
            "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000
        }
        ms_per_candle = interval_ms.get(interval, 300_000)
        start_time = end_time - (limit * ms_per_candle)

        # Run sync call in thread pool
        loop = asyncio.get_event_loop()
        try:
            candles = await loop.run_in_executor(
                None,
                lambda: info.candles_snapshot(symbol, interval, start_time, end_time)
            )
        except Exception as e:
            print(f"[Hyperliquid] Error fetching candles: {e}")
            return []

        if not candles:
            return []

        # Convert Hyperliquid candles to OHLCBar format
        # Hyperliquid format: {"t": timestamp, "o": open, "h": high, "l": low, "c": close, "v": volume}
        bars = []
        for c in candles:
            bars.append(OHLCBar(
                date=self._format_timestamp(c.get("t", 0)),
                open=float(c.get("o", 0)),
                high=float(c.get("h", 0)),
                low=float(c.get("l", 0)),
                close=float(c.get("c", 0)),
                volume=int(float(c.get("v", 0))),
            ))

        print(f"[Hyperliquid] {symbol} {interval}: {len(bars)} bars")
        return bars

    def _format_timestamp(self, ts_ms: int) -> str:
        """Convert timestamp (ms) to ISO format string."""
        from datetime import datetime
        dt = datetime.utcfromtimestamp(ts_ms / 1000)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    async def get_current_price(self, symbol: str) -> float:
        """Get current mid price from Hyperliquid."""
        import asyncio

        info = self._get_info()
        symbol = self._normalize_symbol(symbol)

        loop = asyncio.get_event_loop()
        all_mids = await loop.run_in_executor(
            None,
            lambda: info.all_mids()
        )

        if all_mids and symbol in all_mids:
            price = float(all_mids[symbol])
            print(f"[Hyperliquid] {symbol} price: ${price:.4f}")
            return price

        raise ValueError(f"Price not found for {symbol}")

    async def close(self) -> None:
        """Close client."""
        self._info = None
