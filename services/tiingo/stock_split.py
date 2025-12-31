# services/tiingo/stock_split.py - Stock split handling for Tiingo data.
"""
Stock split detection and price adjustment for Tiingo OHLC data.

This module handles:
- Fetching split data from Tiingo EOD API
- Adjusting historical OHLC prices for stock splits
- Validating cache against recent splits

Split adjustment logic:
- When a stock splits (e.g., 4:1), historical prices BEFORE the split date
  are divided by the split factor to maintain price continuity
- Volume is multiplied by the split factor to maintain volume continuity

Usage:
    from services.tiingo.stock_split import fetch_split_data, adjust_for_splits

    # Fetch split data
    splits_df = await fetch_split_data(api, "AAPL", start_date, end_date)

    # Adjust prices
    adjusted_df = adjust_for_splits(df, splits_df)
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from services.time_centralize_utils import (
    get_et_now_naive,
    to_et_naive,
)
from .api import TiingoAPI


async def fetch_split_data(
    api: TiingoAPI,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch stock split data from Tiingo EOD API.

    Args:
        api: TiingoAPI instance (for session reuse)
        symbol: Stock ticker symbol
        start_date: Start date for split data (naive ET datetime)
        end_date: End date for split data (naive ET datetime)
        verbose: Print progress messages

    Returns:
        DataFrame with split data (date, splitFactor) where splitFactor != 1.0
        Empty DataFrame if no splits found
    """
    if verbose:
        print(f"   [Splits] Checking for stock splits: {symbol}")

    # Ensure dates are naive (remove tzinfo if present)
    start_naive = to_et_naive(start_date) if start_date.tzinfo else start_date
    end_naive = to_et_naive(end_date) if end_date.tzinfo else end_date

    url = f"{api.BASE_URL}/tiingo/daily/{symbol}/prices"
    params = {
        "startDate": start_naive.strftime("%Y-%m-%d"),
        "endDate": end_naive.strftime("%Y-%m-%d"),
    }

    try:
        session = await api._get_session()

        async with session.get(url, params=params) as response:
            if response.status != 200:
                if verbose:
                    print(f"   [Splits] Could not fetch split data: HTTP {response.status}")
                return pd.DataFrame()

            data = await response.json()

            if not data:
                if verbose:
                    print(f"   [Splits] No data returned (possibly new/delisted stock)")
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Filter for actual splits (splitFactor != 1.0)
            if "splitFactor" in df.columns:
                splits_df = df[df["splitFactor"] != 1.0].copy()

                if len(splits_df) > 0:
                    # Parse dates to naive datetime
                    splits_df["date"] = pd.to_datetime(splits_df["date"]).dt.tz_localize(None)
                    splits_df = splits_df[["date", "splitFactor"]].sort_values("date")

                    if verbose:
                        print(f"   [Splits] Found {len(splits_df)} stock split(s):")
                        for _, row in splits_df.iterrows():
                            ratio = row["splitFactor"]
                            date_str = row["date"].strftime("%Y-%m-%d")
                            print(f"      {date_str}: {ratio}:1 split")

                    return splits_df
                else:
                    if verbose:
                        print(f"   [Splits] No splits detected in date range")
            else:
                if verbose:
                    print(f"   [Splits] splitFactor column not in response")

            return pd.DataFrame()

    except Exception as e:
        if verbose:
            print(f"   [Splits] Error fetching split data: {e}")
        return pd.DataFrame()


def adjust_for_splits(
    df: pd.DataFrame,
    splits_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Adjust OHLCV data for stock splits.

    For each split, adjusts all prices BEFORE the split date:
    - Divide prices by split factor
    - Multiply volume by split factor

    Args:
        df: DataFrame with OHLCV data (must have 'date' column)
        splits_df: DataFrame with columns ['date', 'splitFactor']
        verbose: Print progress messages

    Returns:
        Split-adjusted DataFrame
    """
    if splits_df.empty or len(splits_df) == 0:
        return df

    if verbose:
        print(f"   [Splits] Applying split adjustments to {len(df):,} bars...")

    adjusted_df = df.copy()

    # Ensure date column is datetime and naive ET
    if "date" in adjusted_df.columns:
        dates = pd.to_datetime(adjusted_df["date"])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        adjusted_df["date"] = dates

    # Sort splits by date (oldest first)
    splits_df = splits_df.sort_values("date")

    # For each split, adjust all prices BEFORE the split date
    for _, split_row in splits_df.iterrows():
        split_date = split_row["date"]
        split_factor = split_row["splitFactor"]

        # Ensure split_date is naive
        if hasattr(split_date, "tz") and split_date.tz is not None:
            split_date = split_date.tz_localize(None)
        elif hasattr(split_date, "to_pydatetime"):
            split_date = split_date.to_pydatetime()
            if hasattr(split_date, "tzinfo") and split_date.tzinfo is not None:
                split_date = split_date.replace(tzinfo=None)

        # Create mask for data before the split date
        mask = adjusted_df["date"] < split_date

        if mask.sum() > 0:
            # Divide prices by split factor
            adjusted_df.loc[mask, "open"] = (adjusted_df.loc[mask, "open"] / split_factor).round(4)
            adjusted_df.loc[mask, "high"] = (adjusted_df.loc[mask, "high"] / split_factor).round(4)
            adjusted_df.loc[mask, "low"] = (adjusted_df.loc[mask, "low"] / split_factor).round(4)
            adjusted_df.loc[mask, "close"] = (adjusted_df.loc[mask, "close"] / split_factor).round(4)

            # Multiply volume by split factor
            adjusted_df.loc[mask, "volume"] = (adjusted_df.loc[mask, "volume"] * split_factor).round(0)

            if verbose:
                bars_adjusted = mask.sum()
                date_str = split_date.strftime("%Y-%m-%d")
                print(f"   [Splits] Adjusted {bars_adjusted:,} bars before {date_str} (factor: {split_factor})")

    return adjusted_df


async def validate_and_adjust_cache(
    cache_filepath: Path,
    symbol: str,
    cached_df: pd.DataFrame,
    api: TiingoAPI,
    verbose: bool = True,
) -> tuple[bool, pd.DataFrame]:
    """
    Check for splits since cache creation and invalidate if needed.

    This is the main entry point for cache validation with split detection.
    If a split occurred DURING the cached data period, the cache is invalidated
    (return False) because the historical prices need re-fetching with
    split-adjusted values from the API.

    Args:
        cache_filepath: Path to cache file
        symbol: Stock ticker symbol
        cached_df: Cached DataFrame
        api: TiingoAPI instance
        verbose: Print progress messages

    Returns:
        Tuple of (cache_is_valid, cached_df)
        - cache_is_valid: False if cache should be invalidated (split in cached period)
        - cached_df: Original DataFrame (unchanged, for reference)
    """
    if len(cached_df) == 0:
        return True, cached_df

    try:
        # Get cache file modification time as naive datetime
        cache_mtime = datetime.fromtimestamp(cache_filepath.stat().st_mtime)

        # Get data date range
        if "date" in cached_df.columns:
            dates = pd.to_datetime(cached_df["date"])
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
            data_end = dates.max()
        else:
            return True, cached_df

        # Ensure data_end is naive datetime
        if hasattr(data_end, "to_pydatetime"):
            data_end = data_end.to_pydatetime()
        if hasattr(data_end, "tzinfo") and data_end.tzinfo is not None:
            data_end = data_end.replace(tzinfo=None)

        # Fetch split data from cache creation time to now
        now = get_et_now_naive()
        splits_df = await fetch_split_data(api, symbol, cache_mtime, now, verbose=verbose)

        if splits_df.empty:
            return True, cached_df

        # Check if any splits affect the cached data period
        for _, split_row in splits_df.iterrows():
            split_date = split_row["date"]

            # Ensure split_date is naive
            if hasattr(split_date, "tz") and split_date.tz is not None:
                split_date = split_date.tz_localize(None)
            elif hasattr(split_date, "to_pydatetime"):
                split_date = split_date.to_pydatetime()
                if hasattr(split_date, "tzinfo") and split_date.tzinfo is not None:
                    split_date = split_date.replace(tzinfo=None)

            # If split occurred during cached data period, invalidate cache
            if split_date <= data_end:
                if verbose:
                    print(
                        f"   [Splits] Cache invalidated: split on {split_date.strftime('%Y-%m-%d')} "
                        f"(factor: {split_row['splitFactor']})"
                    )
                return False, cached_df

        return True, cached_df

    except Exception as e:
        if verbose:
            print(f"   [Splits] Error validating cache: {e}")
        # If we can't check splits, assume cache is valid
        return True, cached_df
