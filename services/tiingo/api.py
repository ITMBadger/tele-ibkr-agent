# services/tiingo/api.py - Raw Tiingo API calls.
"""
Async Tiingo API client for fetching market data.

This module handles direct API calls with rate limiting.
No caching logic - that's handled by cache.py.

Endpoints:
    - Daily OHLC: https://api.tiingo.com/tiingo/daily/{symbol}/prices
    - Intraday OHLC: https://api.tiingo.com/iex/{symbol}/prices
    - Current Price: https://api.tiingo.com/iex/{symbol}
"""

import asyncio
import os
import random
from datetime import datetime
from typing import Optional

import aiohttp
import pandas as pd


class TiingoAPI:
    """
    Async Tiingo API client.

    Usage:
        api = TiingoAPI()

        # Fetch data
        df = await api.fetch_daily("QQQ", start, end)
        df = await api.fetch_intraday("QQQ", start, end, "5min")
        price = await api.fetch_current_price("QQQ")

        # Close session
        await api.close()
    """

    BASE_URL = "https://api.tiingo.com"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client.

        Args:
            api_key: Tiingo API key (defaults to TIINGO_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY not set in environment")

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {self.api_key}",
                }
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _apply_rate_limit(self) -> None:
        """Apply random delay before API call (1.0 + 0.4-1.4 seconds)."""
        random_delay = random.choice([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        await asyncio.sleep(1.0 + random_delay)

    async def fetch_daily(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLC data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
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

            if not data:
                return pd.DataFrame()

            # Convert to DataFrame
            bars = []
            for item in data:
                bars.append({
                    "date": item["date"][:10],
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": int(item["volume"]),
                })

            df = pd.DataFrame(bars)
            print(
                f"   [Tiingo] {symbol} daily: {len(df)} bars "
                f"({start_date.date()} to {end_date.date()})"
            )
            return df

    async def fetch_intraday(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLC data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Bar interval ("1min", "5min", "15min", "30min", "1hour")

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
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

            if not data:
                return pd.DataFrame()

            # Convert to DataFrame
            bars = []
            for item in data:
                bars.append({
                    "date": item["date"],
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": int(item.get("volume", 0)),
                })

            df = pd.DataFrame(bars)

            # Check Data Delay for the latest bar
            try:
                from services.time_utils import get_et_now

                et_now = get_et_now()
                last_bar_dt = pd.to_datetime(df["date"].iloc[-1]).tz_convert(
                    "America/New_York"
                )

                delay_seconds = (et_now - last_bar_dt).total_seconds()
                minute_match = et_now.minute == last_bar_dt.minute

                match_icon = "OK" if minute_match else "LATE"
                print(
                    f"   [Delay] {symbol}: Last Bar={last_bar_dt.strftime('%H:%M:%S')}, "
                    f"System ET={et_now.strftime('%H:%M:%S')}, Delay={delay_seconds:.1f}s, "
                    f"Min Match={match_icon}"
                )
            except Exception:
                pass

            print(
                f"   [Tiingo] {symbol} {interval}: {len(df)} bars "
                f"({start_date.date()} to {end_date.date()})"
            )
            return df

    async def fetch_intraday_by_month(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1min",
    ) -> pd.DataFrame:
        """
        Fetch intraday data by month boundaries (for backtest downloads).

        Downloads month by month to handle large date ranges.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Bar interval

        Returns:
            DataFrame with all bars combined
        """
        from dateutil.relativedelta import relativedelta

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Generate month boundaries
        boundaries = []
        current = start_dt.replace(day=1)

        while current <= end_dt:
            month_start = current
            next_month = current + relativedelta(months=1)
            month_end = next_month - pd.Timedelta(days=1)

            actual_start = max(month_start, start_dt)
            actual_end = min(month_end, end_dt)

            if actual_start <= actual_end:
                boundaries.append((actual_start, actual_end))

            current = next_month

        print(
            f"  Downloading {symbol} from {start_date} to {end_date} "
            f"({len(boundaries)} months)..."
        )

        all_dfs = []

        session = await self._get_session()

        for month_start, month_end in boundaries:
            url = f"{self.BASE_URL}/iex/{symbol}/prices"
            params = {
                "startDate": month_start.strftime("%Y-%m-%d"),
                "endDate": month_end.strftime("%Y-%m-%d"),
                "resampleFreq": interval,
            }

            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            df = pd.DataFrame(data)
                            all_dfs.append(df)
                            print(
                                f"    {month_start.strftime('%Y-%m')}: "
                                f"{len(data):,} bars"
                            )
                        else:
                            print(f"    {month_start.strftime('%Y-%m')}: No data")
                    elif resp.status == 404:
                        print(f"    {month_start.strftime('%Y-%m')}: No data (404)")
                    else:
                        text = await resp.text()
                        print(
                            f"    {month_start.strftime('%Y-%m')}: "
                            f"Error {resp.status}: {text[:100]}"
                        )
            except Exception as e:
                print(f"    {month_start.strftime('%Y-%m')}: Error - {e}")

            await asyncio.sleep(0.3)  # Rate limiting

        if not all_dfs:
            return pd.DataFrame()

        # Combine all months
        df = pd.concat(all_dfs, ignore_index=True)

        # Standardize columns
        if "timestamp" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"timestamp": "date"})

        required = ["date", "open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "volume" not in df.columns:
            df["volume"] = 0

        # Parse dates and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Keep only needed columns
        df = df[["date", "open", "high", "low", "close", "volume"]]

        print(f"  Total: {len(df):,} bars downloaded (raw)")

        return df

    async def fetch_current_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

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
                        data[0].get("tngoLast")
                        or data[0].get("last")
                        or data[0].get("prevClose")
                    )
                    if price:
                        price = float(price)
                        print(f"   [Tiingo] {symbol} price: ${price:.2f}")
                        return price

        raise Exception(f"No price data for {symbol}")
