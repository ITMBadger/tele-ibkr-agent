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
    """Get current time in US/Eastern timezone (timezone-aware).

    This is the SINGLE SOURCE OF TRUTH for current time in the application.
    """
    return datetime.now(ET_TZ)


def get_utc_now() -> datetime:
    """Get current time in UTC (timezone-aware). Use for Tiingo API calls."""
    return datetime.now(UTC_TZ)


def get_et_now_naive() -> datetime:
    """Get current time as naive ET datetime (no timezone info).

    Use when comparing with data timestamps that are stored as naive ET.
    """
    return get_et_now().replace(tzinfo=None)


def get_et_timestamp() -> str:
    """Get current time in US/Eastern formatted as string with offset.

    Format: YYYY-MM-DD HH:MM:SS-HH:MM (e.g. '2023-10-02 04:00:00-04:00')
    """
    return get_et_now().isoformat(sep=" ", timespec="seconds")


def get_et_timestamp_naive() -> str:
    """Get current time as naive ET string (no offset).

    Format: YYYY-MM-DD HH:MM:SS (e.g. '2023-10-02 04:00:00')
    """
    return get_et_now_naive().strftime(ET_DATETIME_FORMAT)


def get_et_date() -> str:
    """Get current date in ET as YYYY-MM-DD string."""
    return get_et_now().strftime("%Y-%m-%d")


# =============================================================================
# TIMEZONE CONVERSION UTILITIES
# =============================================================================


def to_et_aware(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """Convert any datetime to timezone-aware Eastern Time."""
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
    """Convert any datetime to naive Eastern Time (no timezone info)."""
    return to_et_aware(dt).replace(tzinfo=None)


def format_et(dt: Union[datetime, pd.Timestamp, str]) -> str:
    """Format datetime as naive ET string (YYYY-MM-DD HH:MM:SS)."""
    return to_et_naive(dt).strftime(ET_DATETIME_FORMAT)


def format_iso_to_et(iso_ts: str) -> str:
    """Convert ISO timestamp string (UTC) to Eastern Time formatted string.

    Example: "2025-12-18T16:00:00.000Z" → "2025-12-18 11:00:00-05:00"
    """
    try:
        dt = parse_datetime(iso_ts)
        dt_et = to_et_aware(dt)
        return dt_et.isoformat(sep=" ", timespec="seconds")
    except Exception:
        return iso_ts


def parse_datetime(dt_str: str) -> datetime:
    """Parse a datetime string in various formats.

    Supports:
    - ISO format: '2024-12-24T09:30:00Z', '2024-12-24 09:30:00-05:00'
    - Naive ET format: '2024-12-24 09:30:00'
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


def get_data_age_minutes(data_timestamp: Union[datetime, pd.Timestamp, str]) -> float:
    """Calculate how old data is compared to current ET time (in minutes)."""
    now = get_et_now_naive()
    data_dt = to_et_naive(data_timestamp)
    delta = now - data_dt
    return delta.total_seconds() / 60


def is_data_stale(
    data_timestamp: Union[datetime, pd.Timestamp, str],
    max_age_minutes: float = DEFAULT_STALENESS_MINUTES,
) -> bool:
    """Check if data is stale (older than threshold).

    Use this before acting on signals to ensure data freshness.
    """
    return get_data_age_minutes(data_timestamp) > max_age_minutes


def validate_signal_timing(
    signal_timestamp: Union[datetime, pd.Timestamp, str],
    max_age_minutes: float = DEFAULT_STALENESS_MINUTES,
) -> tuple[bool, str]:
    """Validate signal timestamp is recent enough to act on.

    Prevents executing trades based on stale data.
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
    """Quick check if time is within typical market hours (9:30-16:00 ET).

    Note: Simple check only. For holidays/early closes, use pandas_market_calendars.
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
    """Calculate minutes until a specific ET time today (negative if passed)."""
    now = get_et_now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    delta = target - now
    return delta.total_seconds() / 60


def get_next_market_open() -> datetime:
    """Get next market open time (9:30 AM ET on next trading day).

    Note: Simple implementation. Does not account for holidays.
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
    """Get current month in ET as YYYY-MM string."""
    return get_et_now().strftime("%Y-%m")


def get_end_of_day(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """Get end of day (23:59:59.999999) for a given date.

    Use for inclusive date range filtering to include all intraday bars.
    """
    if isinstance(dt, str):
        parsed = datetime.strptime(dt[:10], "%Y-%m-%d")
    elif isinstance(dt, pd.Timestamp):
        parsed = dt.to_pydatetime().replace(tzinfo=None)
    else:
        parsed = dt.replace(tzinfo=None) if dt.tzinfo else dt

    return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_day(dt: Union[datetime, pd.Timestamp, str]) -> datetime:
    """Get start of day (00:00:00) for a given date."""
    if isinstance(dt, str):
        parsed = datetime.strptime(dt[:10], "%Y-%m-%d")
    elif isinstance(dt, pd.Timestamp):
        parsed = dt.to_pydatetime().replace(tzinfo=None)
    else:
        parsed = dt.replace(tzinfo=None) if dt.tzinfo else dt

    return parsed.replace(hour=0, minute=0, second=0, microsecond=0)


def get_month_start(year_month: str) -> datetime:
    """Get first day of month as naive ET datetime at 00:00:00 (YYYY-MM format)."""
    return datetime.strptime(year_month, "%Y-%m")


def get_month_end(year_month: str) -> datetime:
    """Get last day of month as naive ET datetime at 23:59:59.999999 (YYYY-MM format)."""
    dt = datetime.strptime(year_month, "%Y-%m")
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1)
    else:
        next_month = dt.replace(month=dt.month + 1)
    return next_month - timedelta(microseconds=1)
