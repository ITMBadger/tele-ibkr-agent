# services/tiingo.py - Tiingo API data fetcher service.
"""
This service fetches OHLC market data from Tiingo for strategy calculations.
It runs in the async event loop (Thread 1).

Architecture (3-layer):
    Layer 1 (Private): Raw API calls with rate limiting
        - _fetch_daily_api()
        - _fetch_intraday_api()

    Layer 2 (Private): Cache logic
        - _fetch_with_cache() - generic two-call strategy for any data type

    Layer 3 (Public): High-level API
        - get_daily_ohlc(use_cache=True)
        - get_intraday_ohlc(use_cache=True)
        - get_current_price() - no cache (real-time)

Caching Strategy:
    - Historical data (days ago → yesterday): Cached permanently
    - Today's data: Always fresh, never cached
    - Cache files: data/cache/{symbol}_{type}_{start}_{end}.json

Endpoints used:
    - Daily OHLC: https://api.tiingo.com/tiingo/daily/{symbol}/prices
    - Intraday OHLC: https://api.tiingo.com/iex/{symbol}/prices
    - Current Price: https://api.tiingo.com/iex/{symbol}
"""

import asyncio
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, TypedDict

import aiohttp
import pandas as pd
import pandas_market_calendars as mcal

# Cache directory for historical data
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


class OHLCBar(TypedDict):
    """Single daily OHLC bar."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class IntradayBar(TypedDict):
    """Single intraday OHLC bar."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TiingoService:
    """
    Async Tiingo API client for fetching market data.

    All data fetching methods support caching via use_cache parameter (default True).

    Usage:
        tiingo = TiingoService()

        # Daily data (cached)
        daily = await tiingo.get_daily_ohlc("QQQ", days=250)

        # Intraday data (cached)
        bars_1m = await tiingo.get_intraday_ohlc("QQQ", days=5, interval="1min")

        # Current price (no cache - real-time)
        price = await tiingo.get_current_price("QQQ")

        # Disable cache for specific call
        fresh = await tiingo.get_daily_ohlc("QQQ", use_cache=False)
    """

    BASE_URL = "https://api.tiingo.com"

    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY not set in environment")

        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {self.api_key}"
                }
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================================================
    # LAYER 1: RAW API FETCH (Private)
    # =========================================================================

    async def _apply_rate_limit(self) -> None:
        """Apply random delay before API call (1.0 + 0.4-1.4 seconds)."""
        random_delay = random.choice([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        await asyncio.sleep(1.0 + random_delay)

    async def _fetch_daily_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> list[dict]:
        """
        Raw API call to fetch daily OHLC data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of OHLC bar dicts
        """
        await self._apply_rate_limit()

        session = await self._get_session()
        url = f"{self.BASE_URL}/tiingo/daily/{symbol}/prices"
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
        }

        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Tiingo API error {response.status}: {text}")

            data = await response.json()
            bars = []
            if data:
                for item in data:
                    bars.append({
                        'date': item['date'][:10],
                        'open': float(item['open']),
                        'high': float(item['high']),
                        'low': float(item['low']),
                        'close': float(item['close']),
                        'volume': int(item['volume'])
                    })
            
            print(f"   🌐 [Tiingo] {symbol} daily: {len(bars)} bars ({start_date.date()} to {end_date.date()})")
            return bars

    async def _fetch_intraday_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> list[dict]:
        """
        Raw API call to fetch intraday OHLC data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Bar interval ("1min", "5min", etc.)

        Returns:
            List of OHLC bar dicts
        """
        await self._apply_rate_limit()

        session = await self._get_session()
        url = f"{self.BASE_URL}/iex/{symbol}/prices"
        params = {
            "startDate": start_date.strftime("%Y-%m-%d"),
            "endDate": end_date.strftime("%Y-%m-%d"),
            "resampleFreq": interval,
        }

        async with session.get(url, params=params) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"Tiingo IEX error {response.status}: {text}")

            data = await response.json()
            bars = []
            if data:
                for item in data:
                    bars.append({
                        'date': item['date'],
                        'open': float(item['open']),
                        'high': float(item['high']),
                        'low': float(item['low']),
                        'close': float(item['close']),
                        'volume': int(item.get('volume', 0))
                    })
                
                # Check Data Delay for the latest bar
                try:
                    from services.time_utils import get_et_now
                    et_now = get_et_now()
                    last_bar_dt = pd.to_datetime(bars[-1]['date']).tz_convert('America/New_York')
                    
                    delay_seconds = (et_now - last_bar_dt).total_seconds()
                    minute_match = et_now.minute == last_bar_dt.minute
                    
                    match_icon = "✅" if minute_match else "❌"
                    print(f"   ⏱️ [Delay] {symbol}: Last Bar={last_bar_dt.strftime('%H:%M:%S')}, "
                          f"System ET={et_now.strftime('%H:%M:%S')}, Delay={delay_seconds:.1f}s, "
                          f"Min Match={match_icon}")
                except Exception as e:
                    print(f"   ⚠️ Delay check failed: {e}")
            
            print(f"   🌐 [Tiingo] {symbol} {interval}: {len(bars)} bars ({start_date.date()} to {end_date.date()})")
            return bars

    # =========================================================================
    # LAYER 2: CACHE LOGIC (Private)
    # =========================================================================

    def _get_cache_path(
        self, symbol: str, data_type: str, start_date: str, end_date: str
    ) -> Path:
        """Generate cache file path."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{symbol}_{data_type}_{start_date}_{end_date}.csv"

    def _load_from_cache(self, cache_path: Path) -> list | None:
        """Load data from CSV cache using pandas."""
        if not cache_path.exists():
            return None
        try:
            df = pd.read_csv(cache_path)
            data = df.to_dict('records')
            
            # Extract info for logging
            parts = cache_path.stem.split('_')
            symbol = parts[0] if len(parts) > 0 else "???"
            dtype = parts[1] if len(parts) > 1 else "???"
            print(f"   📦 [Cache] {symbol} {dtype}: Loaded {len(data)} bars")
            
            return data
        except Exception as e:
            print(f"Error loading CSV cache {cache_path}: {e}")
            return None

    def _save_to_cache(self, cache_path: Path, data: list) -> None:
        """Save data to CSV cache using pandas."""
        if not data:
            return
            
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df = pd.DataFrame(data)
            df.to_csv(cache_path, index=False)
        except Exception as e:
            print(f"Error saving CSV cache {cache_path}: {e}")

    async def _fetch_with_cache(
        self,
        symbol: str,
        days: int,
        data_type: str,
        fetch_func: Callable,
        use_cache: bool = True,
        **fetch_kwargs
    ) -> list[dict]:
        """
        Generic two-call cache strategy for any data type.

        Strategy:
            1. Historical (days ago → yesterday): Cached permanently
            2. Today: Always fresh, never cached
            3. Merge both results

        Args:
            symbol: Stock symbol
            days: Number of days of history
            data_type: Cache key type (e.g., "daily", "1min", "5min")
            fetch_func: Function to call for API fetch
            use_cache: Whether to use cache for historical data
            **fetch_kwargs: Additional args to pass to fetch_func

        Returns:
            Combined list of OHLC bars
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - timedelta(days=1)
        start_date = today - timedelta(days=days)

        all_bars: list[dict] = []

        # ===== CALL 1: Historical data (start_date to yesterday) =====
        historical_start = start_date.strftime("%Y%m%d")
        historical_end = yesterday.strftime("%Y%m%d")
        cache_path = self._get_cache_path(
            symbol, data_type, historical_start, historical_end
        )

        historical_bars = None
        if use_cache:
            historical_bars = self._load_from_cache(cache_path)

        if historical_bars is None:
            # Cache miss - fetch from API
            historical_bars = await fetch_func(
                symbol, start_date, yesterday, **fetch_kwargs
            )
            # Save to cache (historical data doesn't change)
            if use_cache and historical_bars:
                self._save_to_cache(cache_path, historical_bars)

        if historical_bars:
            all_bars.extend(historical_bars)

        # ===== CALL 2: Today's data (always fresh, no cache) =====
        today_bars = await fetch_func(
            symbol, today, datetime.now(), **fetch_kwargs
        )
        if today_bars:
            all_bars.extend(today_bars)

        return all_bars

    def _filter_to_market_hours(self, bars: list[dict]) -> list[dict]:
        """
        Filter intraday bars to NYSE market hours only.

        Removes pre-market, after-hours, holidays, and early close data.
        Uses NYSE calendar from pandas-market-calendars.
        """
        if not bars:
            return bars

        # Convert to DataFrame for filtering
        df = pd.DataFrame(bars)
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')

        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)

        if schedule.empty:
            return []  # No market days in range

        # Convert to ET and make timezone-naive for comparison
        schedule_et = schedule.copy()
        schedule_et['market_open'] = (
            schedule_et['market_open']
            .dt.tz_convert('America/New_York')
            .dt.tz_localize(None)
        )
        schedule_et['market_close'] = (
            schedule_et['market_close']
            .dt.tz_convert('America/New_York')
            .dt.tz_localize(None)
        )

        # Make df index timezone-naive if needed
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Filter to market hours
        valid_mask = pd.Series(False, index=df.index)
        for _, row in schedule_et.iterrows():
            day_mask = (
                (df.index >= row['market_open']) &
                (df.index <= row['market_close'])
            )
            valid_mask = valid_mask | day_mask

        filtered_df = df[valid_mask]

        # Convert back to list of dicts
        result = []
        for _, row in filtered_df.iterrows():
            result.append({
                'date': row['date'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': int(row.get('volume', 0))
            })

        return result

    # =========================================================================
    # LAYER 3: PUBLIC API
    # =========================================================================

    async def get_daily_ohlc(
        self,
        symbol: str,
        days: int = 250,
        use_cache: bool = True
    ) -> list[OHLCBar]:
        """
        Fetch daily OHLC data for a symbol.

        Uses two-call cache strategy:
        - Historical (days ago → yesterday): Cached permanently
        - Today: Always fresh

        Args:
            symbol: Stock symbol (e.g., "QQQ")
            days: Number of days of history (default 250 for ~1 year)
            use_cache: Whether to use cache (default True)

        Returns:
            List of OHLC bars, oldest first
        """
        bars = await self._fetch_with_cache(
            symbol=symbol,
            days=days,
            data_type="daily",
            fetch_func=self._fetch_daily_api,
            use_cache=use_cache
        )

        # Convert to OHLCBar TypedDicts
        return [OHLCBar(**bar) for bar in bars]

    async def get_closes(
        self,
        symbol: str,
        days: int = 250,
        use_cache: bool = True
    ) -> list[float]:
        """
        Get just the closing prices for a symbol (daily).

        Args:
            symbol: Stock symbol
            days: Number of days of history
            use_cache: Whether to use cache (default True)

        Returns:
            List of closing prices, oldest first
        """
        bars = await self.get_daily_ohlc(symbol, days, use_cache)
        return [bar["close"] for bar in bars]

    async def get_intraday_ohlc(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True
    ) -> list[IntradayBar]:
        """
        Fetch intraday OHLC data for a symbol.

        Uses two-call cache strategy:
        - Historical (days ago → yesterday): Cached permanently
        - Today: Always fresh

        Data is filtered to NYSE market hours only.

        Args:
            symbol: Stock symbol (e.g., "QQQ")
            days: Number of days of history (including today)
            interval: Bar interval ("1min", "5min", "15min", "30min", "1hour")
            use_cache: Whether to use cache (default True)

        Returns:
            List of OHLC bars filtered to market hours, oldest first
        """
        bars = await self._fetch_with_cache(
            symbol=symbol,
            days=days,
            data_type=interval,
            fetch_func=self._fetch_intraday_api,
            use_cache=use_cache,
            interval=interval
        )

        # Filter to market hours
        filtered_bars = self._filter_to_market_hours(bars)

        # Convert to IntradayBar TypedDicts
        return [IntradayBar(**bar) for bar in filtered_bars]

    async def get_intraday_closes(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5min",
        use_cache: bool = True
    ) -> list[float]:
        """
        Get just the closing prices for intraday bars.

        Args:
            symbol: Stock symbol
            days: Number of days of history
            interval: Bar interval ("1min", "5min", "15min", "30min", "1hour")
            use_cache: Whether to use cache (default True)

        Returns:
            List of closing prices, oldest first
        """
        bars = await self.get_intraday_ohlc(symbol, days, interval, use_cache)
        return [bar["close"] for bar in bars]

    async def get_current_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        This method does NOT use cache as it's real-time data.
        Tries IEX real-time endpoint first, falls back to daily data.

        Args:
            symbol: Stock symbol

        Returns:
            Latest price

        Raises:
            Exception: If no price data available
        """
        await self._apply_rate_limit()

        session = await self._get_session()
        url = f"{self.BASE_URL}/iex/{symbol}"

        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                if data and len(data) > 0:
                    price = (
                        data[0].get("tngoLast") or
                        data[0].get("last") or
                        data[0].get("prevClose")
                    )
                    if price:
                        price = float(price)
                        print(f"   🌐 [Tiingo] {symbol} price: ${price:.2f}")
                        return price

            # Fallback to daily endpoint (will use cache)
            bars = await self.get_daily_ohlc(symbol, days=5)
            if bars:
                return bars[-1]["close"]

            raise Exception(f"No price data for {symbol}")
