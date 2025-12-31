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

Smart Range Matching:
    - If requested range is within cached range, filter from CSV (no API call)
    - Cache files within DEFAULT_TOLERANCE_DAYS are reused for end date
    - Stale files (outside tolerance) are auto-deleted
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from services.time_centralize_utils import get_start_of_day, get_end_of_day


# Default tolerance for cache reuse (days)
DEFAULT_TOLERANCE_DAYS = 7


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

        """
        return self.cache_dir / f"{symbol}_{data_type}_{start_date}_{end_date}.csv"

    def exists(self, cache_path: Path) -> bool:
        """Check if cache file exists."""
        return cache_path.exists()

    def load(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from cache.

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

    # =========================================================================
    # SMART RANGE MATCHING
    # =========================================================================

    def find_containing_cache(
        self,
        symbol: str,
        data_type: str,
        target_start: datetime,
        target_end: datetime,
        tolerance_days: int = DEFAULT_TOLERANCE_DAYS,
    ) -> Tuple[Optional[Path], bool, bool]:
        """
        Find cache file that contains the requested date range.

        Logic:
        1. Search for matching files: {symbol}_{data_type}_*.csv
        2. If file's date range contains requested range (with tolerance for end):
           return file for filtering
        3. If end_date is stale (> tolerance_days): delete and return None

        """
        pattern = f"{symbol}_{data_type}_*.csv"
        candidates = list(self.cache_dir.glob(pattern))

        target_start_naive = target_start.replace(tzinfo=None)
        target_end_naive = target_end.replace(tzinfo=None)
        deleted_any = False

        for candidate in candidates:
            try:
                parts = candidate.stem.split("_")

                # Parse IPO vs regular format
                if "_IPO_" in candidate.name:
                    if len(parts) != 5:
                        continue
                    _, _, _, cand_start_str, cand_end_str = parts
                else:
                    if len(parts) != 4:
                        continue
                    _, _, cand_start_str, cand_end_str = parts

                # Parse cache file dates
                cand_start = datetime.strptime(cand_start_str, "%Y%m%d")
                cand_end = datetime.strptime(cand_end_str, "%Y%m%d")

                # Check if cache contains the requested start date
                if cand_start > target_start_naive:
                    # Cache starts after requested start - can't use
                    continue

                # Check end date with tolerance
                end_diff_days = (target_end_naive - cand_end).days

                if end_diff_days <= tolerance_days:
                    # Cache end is within tolerance of target end
                    # Check if cache fully contains requested range
                    if cand_start <= target_start_naive:
                        needs_filtering = (
                            cand_start < target_start_naive or
                            cand_end > target_end_naive
                        )
                        print(
                            f"   [Cache] Found {candidate.name} containing requested range"
                            + (" (will filter)" if needs_filtering else "")
                        )
                        return candidate, needs_filtering, False
                else:
                    # Cache is stale - delete it
                    print(f"   [Cache] Deleting stale {candidate.name} (end date {end_diff_days} days ago)")
                    candidate.unlink()
                    deleted_any = True

            except Exception as e:
                print(f"   [Cache] Error processing {candidate.name}: {e}")
                continue

        return None, False, deleted_any

    def load_filtered(
        self,
        cache_path: Path,
        target_start: datetime,
        target_end: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Load cache and filter to requested date range.

        """
        df = self.load(cache_path)
        if df is None or df.empty:
            return df

        # Ensure date column is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

            # Apply date filters using centralized utilities
            start_dt = get_start_of_day(target_start)
            end_dt = get_end_of_day(target_end)

            original_len = len(df)
            df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

            if len(df) < original_len:
                print(f"   [Cache] Filtered {original_len} → {len(df)} bars")

        return df.reset_index(drop=True)

    def rename_cache(
        self,
        old_path: Path,
        symbol: str,
        data_type: str,
        new_start: datetime,
        new_end: datetime,
    ) -> Path:
        """
        Rename cache file to reflect new date range.

        Use this when extending cache with new data.

        """
        start_str = new_start.strftime("%Y%m%d")
        end_str = new_end.strftime("%Y%m%d")
        new_path = self.get_path(symbol, data_type, start_str, end_str)

        if old_path.exists() and old_path != new_path:
            old_path.rename(new_path)
            print(f"   [Cache] Renamed {old_path.name} → {new_path.name}")

        return new_path
