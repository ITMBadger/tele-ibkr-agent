from datetime import datetime
from zoneinfo import ZoneInfo

def get_et_now() -> datetime:
    """
    Get current time in US/Eastern timezone.
    """
    return datetime.now(ZoneInfo("US/Eastern"))

def get_et_timestamp() -> str:
    """
    Get current time in US/Eastern formatted as string.
    Format: YYYY-MM-DD HH:MM:SS-HH:MM (e.g. 2023-10-02 04:00:00-04:00)
    """
    return get_et_now().isoformat(sep=" ", timespec="seconds")

def format_iso_to_et(iso_ts: str) -> str:
    """
    Convert an ISO timestamp string (UTC) to Eastern Time formatted string.
    
    Args:
        iso_ts: ISO format string (e.g. "2025-12-18T16:00:00.000Z")
        
    Returns:
        Formatted string (e.g. "2025-12-18 11:00:00-05:00")
    """
    try:
        # Parse ISO string
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        
        # Convert to ET
        dt_et = dt.astimezone(ZoneInfo("US/Eastern"))
        
        return dt_et.isoformat(sep=" ", timespec="seconds")
    except Exception:
        return iso_ts  # Return original if parsing fails
