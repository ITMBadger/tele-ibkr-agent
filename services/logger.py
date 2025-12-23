# services/logger.py - Dual Logging System
"""
Dual Logging System - Terminal capture and Strategy CSV logging.

Terminal Logger:
- Captures ALL print() output without modifying existing code
- Session-based files: data/logs/terminal_YYYYMMDD_HHMMSS.txt

Strategy Logger:
- Full OHLC data dump with indicator values
- Triggered on: signal events + every 30 minutes
- Session-based files: data/logs/strategy_{symbol}_YYYYMMDD_HHMMSS.csv
"""

import sys
import csv
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO

from services.time_utils import get_et_now, format_iso_to_et


# === PATHS ===

LOG_DIR = Path(__file__).parent.parent / "data" / "logs"


def ensure_log_dir() -> None:
    """Create logs directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TERMINAL LOGGER - Captures all print() output
# =============================================================================

class TeeWriter:
    """
    A file-like object that writes to both the original stream and a log file.

    Thread-safe: uses a lock for file writes.
    """

    def __init__(self, original: TextIO, log_file: TextIO):
        self.original = original
        self.log_file = log_file
        self._lock = threading.Lock()
        self.at_line_start = True

    def write(self, message: str) -> int:
        """Write to both original stream and log file with timestamps."""
        try:
            timestamp = f"[{get_et_now().strftime('%H:%M')}] "
            out_str = ""
            parts = message.split('\n')

            for i, part in enumerate(parts):
                is_last = (i == len(parts) - 1)

                if not is_last:
                    if self.at_line_start:
                        out_str += timestamp + part + '\n'
                    else:
                        out_str += part + '\n'
                    self.at_line_start = True
                else:
                    if part:
                        if self.at_line_start:
                            out_str += timestamp + part
                        else:
                            out_str += part
                        self.at_line_start = False

            # Write formatted string to both
            self.original.write(out_str)
            
            with self._lock:
                self.log_file.write(out_str)
                self.log_file.flush()
        except Exception:
            # Fallback to original if timestamping fails
            self.original.write(message)

        return len(message)

    def flush(self) -> None:
        """Flush both streams."""
        self.original.flush()
        with self._lock:
            try:
                self.log_file.flush()
            except Exception:
                pass

    def fileno(self) -> int:
        """Return original fileno for compatibility."""
        return self.original.fileno()

    def isatty(self) -> bool:
        """Check if original is a tty."""
        return self.original.isatty()


class TerminalLogger:
    """
    Singleton terminal logger that captures all stdout/stderr.

    Usage:
        from services.logger import terminal_logger
        terminal_logger.start()

        # All subsequent print() calls are logged automatically

        terminal_logger.stop()
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._log_file: TextIO | None = None
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None
        self._session_file: Path | None = None
        self._started = False

    def start(self) -> Path:
        """
        Start capturing terminal output.

        Returns:
            Path to the log file
        """
        if self._started:
            return self._session_file

        ensure_log_dir()

        # Generate session filename
        timestamp = get_et_now().strftime("%Y%m%d_%H%M%S")
        self._session_file = LOG_DIR / f"terminal_{timestamp}.txt"

        # Open log file
        self._log_file = open(self._session_file, "w", encoding="utf-8")

        # Write header
        self._log_file.write(f"=== Terminal Log Session: {timestamp} ===\n")
        self._log_file.write(f"Started: {get_et_now().isoformat()}\n")
        self._log_file.write("=" * 60 + "\n\n")
        self._log_file.flush()

        # Save originals and redirect
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        sys.stdout = TeeWriter(self._original_stdout, self._log_file)
        sys.stderr = TeeWriter(self._original_stderr, self._log_file)

        self._started = True
        return self._session_file

    def stop(self) -> None:
        """Stop capturing and restore original stdout/stderr."""
        if not self._started:
            return

        # Restore originals
        if self._original_stdout:
            sys.stdout = self._original_stdout
        if self._original_stderr:
            sys.stderr = self._original_stderr

        # Close log file
        if self._log_file:
            self._log_file.write(f"\n{'=' * 60}\n")
            self._log_file.write(f"Session ended: {get_et_now().isoformat()}\n")
            self._log_file.close()

        self._started = False

    @property
    def log_path(self) -> Path | None:
        """Get current log file path."""
        return self._session_file


# Singleton instance
terminal_logger = TerminalLogger()


# =============================================================================
# STRATEGY LOGGER - OHLC + Indicator CSV dumps
# =============================================================================

class StrategyLogger:
    """
    Logs strategy OHLC data with indicators to CSV files.

    Features:
    - Session-based files per symbol: strategy_{symbol}_YYYYMMDD_HHMMSS.csv
    - Full OHLC dump (200+ bars) with indicator values
    - Triggered on signal events and periodic intervals
    """

    # Track loggers by symbol for session consistency
    _instances: dict[str, "StrategyLogger"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_or_create(cls, symbol: str, strategy_name: str) -> "StrategyLogger":
        """Get existing logger for symbol or create new one."""
        key = symbol.upper()
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(symbol, strategy_name)
            return cls._instances[key]

    @classmethod
    def clear_all(cls) -> None:
        """Clear all logger instances (call on shutdown)."""
        with cls._lock:
            cls._instances.clear()

    def __init__(self, symbol: str, strategy_name: str):
        self.symbol = symbol.upper()
        self.strategy_name = strategy_name
        self._session_file: Path | None = None
        self._header_written = False
        self._current_columns: list[str] | None = None
        self._lock = threading.Lock()

        self._init_session_file()

    def _init_session_file(self) -> None:
        """Initialize session-based CSV file."""
        ensure_log_dir()
        timestamp = get_et_now().strftime("%Y%m%d_%H%M%S")
        self._session_file = LOG_DIR / f"strategy_{self.symbol}_{timestamp}.csv"

    def log_event(
        self,
        ohlc_bars: list[dict],
        indicator_columns: dict[str, list[float | None]] | None = None,
        signal: str = "",
        triggered: bool = False,
        event_type: str = "periodic"  # "signal" or "periodic"
    ) -> bool:
        """
        Log OHLC data with indicators to CSV.

        Args:
            ohlc_bars: List of OHLC dicts with keys: date, open, high, low, close, volume
            indicator_columns: Dict mapping column name to list of values
                               e.g., {"ema_200": [485.5, 485.6, ...]}
                               Length must match ohlc_bars
            signal: Signal type ("BUY", "SELL", "HOLD", etc.)
            triggered: Whether order was actually submitted
            event_type: "signal" (on trigger) or "periodic" (30-min snapshot)

        Returns:
            bool: True if logged successfully
        """
        if not ohlc_bars:
            return False

        indicator_columns = indicator_columns or {}

        with self._lock:
            try:
                # Generate unique filename for this specific event
                # Format: strategy_SYMBOL_TYPE_YYYYMMDD_HHMMSS.csv
                timestamp = get_et_now().strftime("%Y%m%d_%H%M%S")
                filename = f"strategy_{self.symbol}_{event_type}_{timestamp}.csv"
                file_path = LOG_DIR / filename
                self._session_file = file_path  # Track last log path

                # Build column list
                base_cols = ["event_time", "event_type", "signal", "triggered",
                            "bar_date", "open", "high", "low", "close", "volume"]
                indicator_col_names = sorted(indicator_columns.keys())
                all_cols = base_cols + indicator_col_names

                # Open in "w" mode for a fresh file per event
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)

                    # Always write header for the new file
                    writer.writerow(all_cols)
                    self._header_written = True
                    self._current_columns = all_cols

                    # Write each bar as a row
                    event_time = get_et_now().isoformat()

                    for i, bar in enumerate(ohlc_bars):
                        # Signal/triggered only on last bar
                        is_last = i == len(ohlc_bars) - 1

                        row = [
                            event_time,
                            event_type,
                            signal if is_last else "",
                            triggered if is_last else "",
                            format_iso_to_et(bar.get("date", "")),
                            bar.get("open", ""),
                            bar.get("high", ""),
                            bar.get("low", ""),
                            bar.get("close", ""),
                            bar.get("volume", ""),
                        ]

                        # Add indicator values
                        for col_name in indicator_col_names:
                            values = indicator_columns.get(col_name, [])
                            if i < len(values):
                                val = values[i]
                                row.append(f"{val:.4f}" if val is not None else "")
                            else:
                                row.append("")

                        writer.writerow(row)

                print(f"   📝 Saved unique strategy CSV: {filename}")
                return True

            except Exception as e:
                print(f"[StrategyLogger] Error logging {self.symbol}: {e}")
                return False

    @property
    def log_path(self) -> Path | None:
        """Get current log file path."""
        return self._session_file
