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
    bars = await data.get_ohlc("SYMBOL", days=5, interval="5min")
    price = await data.get_current_price("SYMBOL")
"""

from abc import ABC, abstractmethod
from typing import TypedDict


# Crypto symbols that need USD suffix for Tiingo API
# Shared across TiingoDataProvider and component_test
CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "AVAX", "LINK", "DOGE", "ADA", "DOT",
    "MATIC", "UNI", "AAVE", "ATOM", "XRP", "LTC", "BCH", "XLM",
}


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
    async def get_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """Fetch OHLC data."""
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass

    async def close(self) -> None:
        """Close any open connections."""
        pass


class TiingoDataProvider(MarketDataProvider):
    """Market data provider using Tiingo (for stocks and crypto)."""

    def __init__(self, tiingo_service, crypto_mode: bool = False):
        """
        Args:
            tiingo_service: TiingoService instance
            crypto_mode: If True, auto-translate symbols (BTC → BTCUSD)
        """
        self.tiingo = tiingo_service
        self.crypto_mode = crypto_mode

    def _translate_symbol(self, symbol: str) -> str:
        """Translate symbol for Tiingo API (BTC → BTCUSD in crypto mode)."""
        if not self.crypto_mode:
            return symbol

        symbol_upper = symbol.upper()
        # Already has USD suffix - return as-is
        if symbol_upper.endswith("USD") or symbol_upper.endswith("USDT"):
            return symbol

        # Known crypto symbol - add USD suffix
        if symbol_upper in CRYPTO_SYMBOLS:
            return f"{symbol_upper}USD"

        return symbol

    async def get_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        tiingo_symbol = self._translate_symbol(symbol)
        return await self.tiingo.get_ohlc(tiingo_symbol, days, interval, use_cache)

    async def get_current_price(self, symbol: str) -> float:
        tiingo_symbol = self._translate_symbol(symbol)
        return await self.tiingo.get_current_price(tiingo_symbol)

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

    async def get_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True,
    ) -> list[OHLCBar]:
        """
        Fetch OHLC from Hyperliquid.

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
        """Convert timestamp (ms) to naive ET string format."""
        from datetime import datetime
        from services.time_centralize_utils import ET_TZ, ET_DATETIME_FORMAT, UTC_TZ

        # Convert ms timestamp to UTC datetime
        dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=UTC_TZ)
        # Convert to ET and format as naive ET string
        dt_et = dt_utc.astimezone(ET_TZ)
        return dt_et.strftime(ET_DATETIME_FORMAT)

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
