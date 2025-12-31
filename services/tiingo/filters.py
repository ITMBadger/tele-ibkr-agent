# services/tiingo/filters.py - Market hours filtering for OHLC data.
"""
Filter OHLC data to NYSE market hours only.

This module provides market hours filtering for intraday data.
For datetime utilities, use services.time_centralize_utils (centralized time server).
"""

import pandas as pd
import pandas_market_calendars as mcal

# Import from centralized time server
from services.time_centralize_utils import ET_DATETIME_FORMAT


def format_df_dates(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Convert DataFrame date column to naive ET string format.

    """
    df = df.copy()

    if date_column not in df.columns:
        return df

    # Parse dates
    dates = pd.to_datetime(df[date_column])

    # Convert to ET if timezone-aware
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    else:
        # If already naive, assume it's already in the target ET format
        # This matches the project's standard of using naive ET
        pass

    # Format as string
    df[date_column] = dates.dt.strftime(ET_DATETIME_FORMAT)

    return df


# =============================================================================
# MARKET HOURS FILTERING
# =============================================================================


def filter_to_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter intraday bars to NYSE market hours only.

    Removes pre-market, after-hours, holidays, and early close data.
    Uses NYSE calendar from pandas-market-calendars.

    """
    if df.empty:
        return df

    df = df.copy()

    # Handle both column and index formats
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
    elif df.index.name != "date":
        # Assume index is already datetime
        pass

    # Convert to New York time and make naive for comparison
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)

    # Get NYSE calendar
    nyse = mcal.get_calendar("NYSE")
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)

    if schedule.empty:
        return pd.DataFrame()

    # Convert schedule boundaries to naive New York time
    schedule_et = schedule.copy()
    schedule_et["market_open"] = (
        schedule_et["market_open"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    schedule_et["market_close"] = (
        schedule_et["market_close"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )

    # Filter to market hours
    valid_mask = pd.Series(False, index=df.index)
    for _, row in schedule_et.iterrows():
        day_mask = (df.index >= row["market_open"]) & (
            df.index <= row["market_close"]
        )
        valid_mask = valid_mask | day_mask

    filtered_df = df[valid_mask].reset_index()

    return filtered_df
