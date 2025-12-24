# services/tiingo/cache.py - Cache operations for Tiingo data.
"""
Pure cache operations for Tiingo OHLC data.

This module handles reading/writing cached data to CSV files.
It has no dependencies on API calls or async operations.

Cache Strategy:
    - Files are stored as CSV for easy inspection
    - Filename format: {symbol}_{data_type}_{start}_{end}.csv
    - IPO format: {symbol}_{data_type}_IPO_{start}_{end}.csv
    - Data types: "daily", "1min", "5min", etc.

Fuzzy Matching:
    - Cache files within 7-day tolerance are reused
    - Stale files (outside tolerance) are auto-deleted
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


class TiingoCache:
    """
    Sync cache operations for Tiingo data.

    Usage:
        cache = TiingoCache(Path("data/cache"))

        # Check and load
        path = cache.get_path("QQQ", "daily", "20240101", "20241231")
        if cache.exists(path):
            df = cache.load(path)
        else:
            df = ... # fetch from API
            cache.save(path, df)

        # Clear all
        count = cache.clear_all()
    """

    def __init__(self, cache_dir: Path | str):
        """
        Initialize cache with directory.

        Args:
            cache_dir: Directory for cache files (created if missing)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_path(
        self,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Generate cache file path.

        Args:
            symbol: Stock symbol (e.g., "QQQ")
            data_type: Data type (e.g., "daily", "1min", "5min")
            start_date: Start date as YYYYMMDD string
            end_date: End date as YYYYMMDD string

        Returns:
            Path to cache file (may not exist yet)
        """
        return self.cache_dir / f"{symbol}_{data_type}_{start_date}_{end_date}.csv"

    def exists(self, cache_path: Path) -> bool:
        """Check if cache file exists."""
        return cache_path.exists()

    def load(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from cache.

        Args:
            cache_path: Path to cache file

        Returns:
            DataFrame if file exists, None otherwise
        """
        if not cache_path.exists():
            return None

        try:
            df = pd.read_csv(cache_path)

            # Log cache hit
            parts = cache_path.stem.split("_")
            symbol = parts[0] if len(parts) > 0 else "???"
            dtype = parts[1] if len(parts) > 1 else "???"
            print(f"   [Cache] {symbol} {dtype}: Loaded {len(df)} bars")

            return df
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}")
            return None

    def save(self, cache_path: Path, df: pd.DataFrame) -> None:
        """
        Save data to cache.

        Args:
            cache_path: Path to cache file
            df: DataFrame to save
        """
        if df.empty:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            df.to_csv(cache_path, index=False)
        except Exception as e:
            print(f"Error saving cache {cache_path}: {e}")

    def clear_all(self) -> int:
        """
        Delete all cache files.

        Returns:
            Number of files deleted
        """
        if not self.cache_dir.exists():
            return 0

        files = list(self.cache_dir.glob("*.csv"))
        count = 0

        for file in files:
            try:
                file.unlink()
                count += 1
            except Exception as e:
                print(f"Error deleting {file.name}: {e}")

        return count

    def list_files(self) -> list[Path]:
        """List all cache files."""
        if not self.cache_dir.exists():
            return []
        return list(self.cache_dir.glob("*.csv"))

    # =========================================================================
    # IPO HANDLING
    # =========================================================================

    def get_path_ipo(
        self,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
    ) -> Path:
        """
        Generate cache file path for IPO stock (limited history).

        Format: {symbol}_{data_type}_IPO_{start}_{end}.csv

        Args:
            symbol: Stock symbol
            data_type: Data type (e.g., "1min", "5min")
            start_date: Actual start date as YYYYMMDD
            end_date: End date as YYYYMMDD

        Returns:
            Path to IPO cache file
        """
        return self.cache_dir / f"{symbol}_{data_type}_IPO_{start_date}_{end_date}.csv"

    def is_ipo_file(self, cache_path: Path) -> bool:
        """Check if cache file is for IPO stock."""
        return "_IPO_" in cache_path.name

    def detect_ipo(
        self,
        df: pd.DataFrame,
        target_start_date: datetime,
        tolerance_days: int = 30,
    ) -> bool:
        """
        Detect if data represents an IPO stock (limited history).

        Args:
            df: Downloaded OHLC data
            target_start_date: Requested start date
            tolerance_days: Days beyond which data is considered IPO

        Returns:
            True if data starts significantly later than requested
        """
        if df.empty:
            return False

        # Get actual start date from data
        if "date" in df.columns:
            actual_start = pd.to_datetime(df["date"].min())
        elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            actual_start = df.index.min()
        else:
            return False

        # Convert to datetime if needed
        if hasattr(actual_start, "to_pydatetime"):
            actual_start = actual_start.to_pydatetime()
        if actual_start.tzinfo is not None:
            actual_start = actual_start.replace(tzinfo=None)

        target_naive = target_start_date.replace(tzinfo=None)
        days_diff = (actual_start - target_naive).days

        return days_diff > tolerance_days

    # =========================================================================
    # FUZZY MATCHING WITH AUTO-DELETE
    # =========================================================================

    def find_or_cleanup(
        self,
        symbol: str,
        data_type: str,
        target_end_date: datetime,
        tolerance_days: int = 7,
    ) -> Tuple[Optional[Path], bool]:
        """
        Find valid cache or delete stale files.

        Logic:
        1. Search for matching files: {symbol}_{data_type}_*.csv
        2. If end_date within tolerance: return file (reuse)
        3. If end_date outside tolerance: DELETE file, return None

        Args:
            symbol: Stock symbol
            data_type: Data type (e.g., "1min", "5min")
            target_end_date: Target end date (usually today)
            tolerance_days: Max days difference to reuse cache (default 7)

        Returns:
            Tuple of (cache_path or None, was_deleted: bool)
        """
        pattern = f"{symbol}_{data_type}_*.csv"
        candidates = list(self.cache_dir.glob(pattern))

        target_naive = target_end_date.replace(tzinfo=None)
        deleted_any = False

        for candidate in candidates:
            try:
                parts = candidate.stem.split("_")

                # Parse IPO vs regular format
                if "_IPO_" in candidate.name:
                    # Format: {symbol}_{data_type}_IPO_{start}_{end}
                    if len(parts) != 5:
                        continue
                    _, _, _, _, cand_end_str = parts
                else:
                    # Format: {symbol}_{data_type}_{start}_{end}
                    if len(parts) != 4:
                        continue
                    _, _, _, cand_end_str = parts

                # Parse end date
                cand_end = datetime.strptime(cand_end_str, "%Y%m%d")

                # Check tolerance
                days_diff = abs((target_naive - cand_end).days)

                if days_diff <= tolerance_days:
                    # Valid cache - reuse
                    print(f"   [Cache] Reusing {candidate.name} (end date {days_diff} days ago)")
                    return candidate, False
                else:
                    # Stale cache - delete
                    print(f"   [Cache] Deleting stale {candidate.name} (end date {days_diff} days ago)")
                    candidate.unlink()
                    deleted_any = True

            except Exception as e:
                print(f"   [Cache] Error processing {candidate.name}: {e}")
                continue

        return None, deleted_any

    def save_with_ipo_detection(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str,
        target_start_date: datetime,
        target_end_date: datetime,
    ) -> Path:
        """
        Save data to cache, auto-detecting IPO stocks.

        Args:
            df: DataFrame to save
            symbol: Stock symbol
            data_type: Data type
            target_start_date: Requested start date
            target_end_date: Requested end date

        Returns:
            Path to saved cache file
        """
        is_ipo = self.detect_ipo(df, target_start_date)

        if is_ipo:
            # Get actual date range from data
            if "date" in df.columns:
                actual_start = pd.to_datetime(df["date"].min())
                actual_end = pd.to_datetime(df["date"].max())
            else:
                actual_start = df.index.min()
                actual_end = df.index.max()

            start_str = actual_start.strftime("%Y%m%d")
            end_str = actual_end.strftime("%Y%m%d")
            cache_path = self.get_path_ipo(symbol, data_type, start_str, end_str)
            print(f"   [Cache] Detected IPO/new stock - using actual date range")
        else:
            start_str = target_start_date.strftime("%Y%m%d")
            end_str = target_end_date.strftime("%Y%m%d")
            cache_path = self.get_path(symbol, data_type, start_str, end_str)

        self.save(cache_path, df)
        return cache_path
