# services/time_centralize_utils.py - Centralized time server for the entire application.
"""
CENTRALIZED TIME SERVER - Single source of truth for all time operations.

This module provides:
1. Consistent ET (Eastern Time) for the entire application
2. Timezone-aware and naive datetime utilities
3. Data staleness detection for signal validation
4. Time comparison utilities

ALL time operations in the project should use this module.
NEVER use datetime.now() or datetime.utcnow() directly elsewhere.

Standard formats:
- Aware: '2024-12-24 09:30:00-05:00' (with offset)
- Naive: '2024-12-24 09:30:00' (assumed ET)
"""

from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from pytz import timezone as ZoneInfo  # type: ignore


# =============================================================================
# TIMEZONE CONSTANTS
# =============================================================================

ET_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Standard format for naive ET strings (used in data/logs)
ET_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default staleness threshold (minutes) - data older than this triggers warning
DEFAULT_STALENESS_MINUTES = 5


# =============================================================================
# CORE TIME SERVER FUNCTIONS
# =============================================================================


def get_et_now() -> datetime:
    """
    Get current time in US/Eastern timezone (timezone-aware).

    This is the SINGLE SOURCE OF TRUTH for current time in the application.
    All other time operations should derive from this.

    Returns:
        Timezone-aware datetime in Eastern Time
    """
    return datetime.now(ET_TZ)


def get_utc_now() -> datetime:
    """
    Get current time in UTC (timezone-aware).

    Use this for API calls that require UTC time (e.g., Tiingo API).

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(UTC_TZ)


def get_et_now_naive() -> datetime:
    """
    Get current time as naive ET datetime (no timezone info).

    Use this when comparing with data timestamps that are stored as naive ET.

    Returns:
        Naive datetime representing current ET time
    """
    return get_et_now().replace(tzinfo=None)


def get_et_timestamp() -> str:
    """
    Get current time in US/Eastern formatted as string with offset.

    Format: YYYY-MM-DD HH:MM:SS-HH:MM (e.g. '2023-10-02 04:00:00-04:00')

    Returns:
        ISO format string with timezone offset
    """
    return get_et_now().isoformat(sep=" ", timespec="seconds")


def get_et_timestamp_naive() -> str:
    """
    Get current time as naive ET string (no offset).

    Format: YYYY-MM-DD HH:MM:SS (e.g. '2023-10-02 04:00:00')
    Use this for consistency with data timestamps.

    Returns:
        Formatted string without timezone offset
    """
    return get_et_now_naive().strftime(ET_DATETIME_FORMAT)


def get_et_date() -> str:
    """
    Get current date in ET as YYYY-MM-DD string.

    Returns:
        Date string (e.g. '2024-12-24')
    """
    return get_et_now().strftime("%Y-%m-%d")


# =============================================================================
# TIMEZONE CONVERSION UTILITIES
# =============================================================================


def to_et_aware(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """
    Convert any datetime to timezone-aware Eastern Time.

    Args:
        dt: Datetime (aware, naive, or string)

    Returns:
        Timezone-aware datetime in ET
    """
    if isinstance(dt, str):
        dt = parse_datetime(dt)
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()

    if dt.tzinfo is None:
        # Assume naive datetime is already in ET
        return dt.replace(tzinfo=ET_TZ)
    else:
        return dt.astimezone(ET_TZ)


def to_et_naive(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """
    Convert any datetime to naive Eastern Time.

    Args:
        dt: Datetime (can be timezone-aware or naive)

    Returns:
        Naive datetime in ET (no timezone info)
    """
    return to_et_aware(dt).replace(tzinfo=None)


def format_et(dt: Union[datetime, pd.Timestamp, str]) -> str:
    """
    Format datetime as naive ET string.

    Args:
        dt: Datetime to format

    Returns:
        String in format 'YYYY-MM-DD HH:MM:SS'
    """
    return to_et_naive(dt).strftime(ET_DATETIME_FORMAT)


def format_iso_to_et(iso_ts: str) -> str:
    """
    Convert an ISO timestamp string (UTC) to Eastern Time formatted string.

    Args:
        iso_ts: ISO format string (e.g. "2025-12-18T16:00:00.000Z")

    Returns:
        Formatted string (e.g. "2025-12-18 11:00:00-05:00")
    """
    try:
        dt = parse_datetime(iso_ts)
        dt_et = to_et_aware(dt)
        return dt_et.isoformat(sep=" ", timespec="seconds")
    except Exception:
        return iso_ts


def parse_datetime(dt_str: str) -> datetime:
    """
    Parse a datetime string in various formats.

    Supports:
    - ISO format: '2024-12-24T09:30:00Z', '2024-12-24 09:30:00-05:00'
    - Naive ET format: '2024-12-24 09:30:00'

    Args:
        dt_str: Datetime string to parse

    Returns:
        Parsed datetime (timezone-aware if input had timezone, naive otherwise)
    """
    # Try ISO format with Z suffix
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")

    try:
        return datetime.fromisoformat(dt_str)
    except ValueError:
        pass

    # Try standard naive format
    try:
        return datetime.strptime(dt_str, ET_DATETIME_FORMAT)
    except ValueError:
        pass

    # Fallback to pandas for more flexible parsing
    return pd.to_datetime(dt_str).to_pydatetime()


# =============================================================================
# DATA STALENESS DETECTION
# =============================================================================


def get_data_age_minutes(
    data_timestamp: Union[datetime, pd.Timestamp, str]
) -> float:
    """
    Calculate how old data is compared to current ET time.

    Args:
        data_timestamp: The timestamp of the data (assumed ET if naive)

    Returns:
        Age in minutes (positive = data is in the past)
    """
    now = get_et_now_naive()
    data_dt = to_et_naive(data_timestamp)
    delta = now - data_dt
    return delta.total_seconds() / 60


def is_data_stale(
    data_timestamp: Union[datetime, pd.Timestamp, str],
    max_age_minutes: float = DEFAULT_STALENESS_MINUTES,
) -> bool:
    """
    Check if data is stale (older than threshold).

    Use this before acting on signals to ensure data freshness.

    Args:
        data_timestamp: The timestamp of the data
        max_age_minutes: Maximum acceptable age in minutes

    Returns:
        True if data is stale and should not be trusted
    """
    return get_data_age_minutes(data_timestamp) > max_age_minutes


def validate_signal_timing(
    signal_timestamp: Union[datetime, pd.Timestamp, str],
    max_age_minutes: float = DEFAULT_STALENESS_MINUTES,
) -> tuple[bool, str]:
    """
    Validate that a signal's timestamp is recent enough to act on.

    This prevents executing trades based on stale data.

    Args:
        signal_timestamp: When the signal was generated
        max_age_minutes: Maximum acceptable signal age

    Returns:
        Tuple of (is_valid, message)
    """
    age_minutes = get_data_age_minutes(signal_timestamp)

    if age_minutes > max_age_minutes:
        return (
            False,
            f"Signal is {age_minutes:.1f} min old (max: {max_age_minutes}). "
            f"Signal time: {format_et(signal_timestamp)}, "
            f"Current ET: {get_et_timestamp_naive()}"
        )

    if age_minutes < -1:  # Allow 1 min future tolerance for clock skew
        return (
            False,
            f"Signal is {-age_minutes:.1f} min in the future. "
            f"Possible clock sync issue."
        )

    return (True, f"Signal is {age_minutes:.1f} min old - OK")


# =============================================================================
# TIME COMPARISON UTILITIES
# =============================================================================


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """
    Quick check if time is within typical market hours (9:30-16:00 ET).

    Note: This is a simple check. For accurate market calendar handling
    (holidays, early closes), use pandas_market_calendars.

    Args:
        dt: Datetime to check (defaults to current ET time)

    Returns:
        True if within 9:30 AM - 4:00 PM ET
    """
    if dt is None:
        dt = get_et_now()
    else:
        dt = to_et_aware(dt)

    # Weekday check (Mon=0, Sun=6)
    if dt.weekday() >= 5:
        return False

    market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= dt <= market_close


def minutes_until(target_hour: int, target_minute: int = 0) -> float:
    """
    Calculate minutes until a specific ET time today.

    Args:
        target_hour: Target hour (0-23)
        target_minute: Target minute (0-59)

    Returns:
        Minutes until target (negative if target has passed)
    """
    now = get_et_now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    delta = target - now
    return delta.total_seconds() / 60


def get_next_market_open() -> datetime:
    """
    Get the next market open time (9:30 AM ET on next trading day).

    Note: This is a simple implementation. Does not account for holidays.
    For production use, consider using pandas_market_calendars.

    Returns:
        Timezone-aware datetime of next market open
    """
    now = get_et_now()

    # Start with today's market open
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If already past today's open, start from tomorrow
    if now >= market_open:
        market_open += timedelta(days=1)

    # Skip weekends
    while market_open.weekday() >= 5:
        market_open += timedelta(days=1)

    return market_open


# =============================================================================
# DATE RANGE UTILITIES
# =============================================================================


def get_et_month() -> str:
    """
    Get current month in ET as YYYY-MM string.

    Returns:
        Month string (e.g., '2024-12')
    """
    return get_et_now().strftime("%Y-%m")


def get_end_of_day(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """
    Get end of day (23:59:59.999999) for a given date.

    Use this for inclusive date range filtering where you want to include
    all intraday bars on the end date.

    Args:
        dt: Date (string YYYY-MM-DD, datetime, or Timestamp)

    Returns:
        Naive datetime at 23:59:59.999999 ET
    """
    if isinstance(dt, str):
        parsed = datetime.strptime(dt[:10], "%Y-%m-%d")
    elif isinstance(dt, pd.Timestamp):
        parsed = dt.to_pydatetime().replace(tzinfo=None)
    else:
        parsed = dt.replace(tzinfo=None) if dt.tzinfo else dt

    return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_day(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """
    Get start of day (00:00:00) for a given date.

    Args:
        dt: Date (string YYYY-MM-DD, datetime, or Timestamp)

    Returns:
        Naive datetime at 00:00:00 ET
    """
    if isinstance(dt, str):
        parsed = datetime.strptime(dt[:10], "%Y-%m-%d")
    elif isinstance(dt, pd.Timestamp):
        parsed = dt.to_pydatetime().replace(tzinfo=None)
    else:
        parsed = dt.replace(tzinfo=None) if dt.tzinfo else dt

    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def get_month_start(year_month: str) -> datetime:
    """
    Get first day of month as naive ET datetime at 00:00:00.

    Args:
        year_month: Month in YYYY-MM format (e.g., "2024-01")

    Returns:
        Naive datetime for first day of month at midnight ET
    """
    return datetime.strptime(year_month, "%Y-%m")


def get_month_end(year_month: str) -> datetime:
    """
    Get last day of month as naive ET datetime at 23:59:59.999999.

    Args:
        year_month: Month in YYYY-MM format (e.g., "2024-01")

    Returns:
        Naive datetime for last day of month at end-of-day ET
    """
    dt = datetime.strptime(year_month, "%Y-%m")
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1)
    else:
        next_month = dt.replace(month=dt.month + 1)
    return next_month - timedelta(microseconds=1)
