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
from pathlib import Path
from typing import TextIO

from services.time_centralize_utils import get_et_now, format_iso_to_et


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
# SIGNAL LOGGER - Unified logging for live and backtest modes
# =============================================================================

# Base directories for different modes
STRATEGY_LOGS_DIR = Path(__file__).parent.parent / "data" / "strategy_logs"
BACKTEST_RESULTS_DIR = Path(__file__).parent.parent / "data" / "backtest" / "results"


class SignalLogger:
    """
    Unified signal logger for live trading and backtesting.

    Features:
    - Mode-aware: auto-switches output directory based on mode
    - Live mode: saves to data/strategy_logs/
    - Backtest mode: saves to data/backtest/results/{run_id}/
    - Supports both event logging (live) and DataFrame logging (backtest)

    Usage:
        # Live mode (default)
        logger = SignalLogger.get_or_create("QQQ", "ema_200_long")
        logger.log_event(ohlc_bars, indicator_cols, "BUY", True, "signal")

        # Backtest mode
        SignalLogger.set_mode("backtest", "20240115_143022")
        logger = SignalLogger.get_or_create("QQQ", "ema_200_long")
        logger.log_dataframe(df)  # or log_bar() per bar
        SignalLogger.reset()  # Reset to live mode after backtest
    """

    # Class-level mode state
    _mode: str = "live"  # "live" or "backtest"
    _run_id: str | None = None

    # Track loggers by symbol for session consistency
    _instances: dict[str, "SignalLogger"] = {}
    _lock = threading.Lock()

    @classmethod
    def set_mode(cls, mode: str, run_id: str | None = None, create_dir: bool = True) -> None:
        """
        Set logging mode for the session.
        """
        with cls._lock:
            cls._mode = mode
            cls._run_id = run_id

            # Create output directory if in backtest mode and saving is enabled
            if mode == "backtest" and run_id and create_dir:
                output_dir = BACKTEST_RESULTS_DIR / run_id
                output_dir.mkdir(parents=True, exist_ok=True)

            # Clear existing instances when mode changes
            cls._instances.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset to live mode and clear all instances."""
        with cls._lock:
            cls._mode = "live"
            cls._run_id = None
            cls._instances.clear()

    @classmethod
    def get_output_dir(cls) -> Path:
        """Get current output directory based on mode."""
        if cls._mode == "backtest" and cls._run_id:
            return BACKTEST_RESULTS_DIR / cls._run_id
        return STRATEGY_LOGS_DIR

    @classmethod
    def get_or_create(cls, symbol: str, strategy_name: str) -> "SignalLogger":
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
        self._instance_lock = threading.Lock()

        # For backtest mode: accumulate rows for batch writing
        self._backtest_rows: list[dict] = []

        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        if self._mode == "live":
            output_dir = self.get_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        ohlc_bars: list[dict],
        indicator_columns: dict[str, list[float | None]] | None = None,
        signal: str = "",
        triggered: bool = False,
        event_type: str = "periodic",  # "signal" or "periodic"
        row_index: int = -2
    ) -> bool:
        """
        Log OHLC data with indicators to CSV (for live mode).

        """
        if not ohlc_bars:
            return False

        indicator_columns = indicator_columns or {}

        with self._instance_lock:
            try:
                output_dir = self.get_output_dir()
                if not output_dir.exists():
                    return False

                # Generate unique filename for this specific event
                timestamp = get_et_now().strftime("%Y%m%d_%H%M%S")
                filename = f"strategy_{self.symbol}_{event_type}_{timestamp}.csv"
                file_path = output_dir / filename
                self._session_file = file_path

                # Build column list
                base_cols = ["event_time", "event_type", "signal", "triggered",
                            "bar_date", "open", "high", "low", "close", "volume"]
                indicator_col_names = sorted(indicator_columns.keys())
                all_cols = base_cols + indicator_col_names

                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(all_cols)

                    event_time = get_et_now().isoformat()

                    def _round_num(val, decimals=3):
                        """Round number to specified decimals, return empty string if not a number."""
                        if val is None or val == "":
                            return ""
                        try:
                            return round(float(val), decimals)
                        except (ValueError, TypeError):
                            return val

                    # Determine which row gets the signal/triggered values
                    # Convert negative index to positive (e.g., -2 -> len-2)
                    target_index = row_index if row_index >= 0 else len(ohlc_bars) + row_index

                    for i, bar in enumerate(ohlc_bars):
                        is_target = (i == target_index)

                        row = [
                            event_time,
                            event_type,
                            signal if is_target else "",
                            triggered if is_target else "",
                            format_iso_to_et(bar.get("date", "")),
                            _round_num(bar.get("open", "")),
                            _round_num(bar.get("high", "")),
                            _round_num(bar.get("low", "")),
                            _round_num(bar.get("close", "")),
                            bar.get("volume", ""),
                        ]

                        for col_name in indicator_col_names:
                            values = indicator_columns.get(col_name, [])
                            if i < len(values):
                                val = values[i]
                                row.append(_round_num(val) if val is not None else "")
                            else:
                                row.append("")

                        writer.writerow(row)

                print(f"   📝 Saved: {filename}")
                return True

            except Exception as e:
                print(f"[SignalLogger] Error logging {self.symbol}: {e}")
                return False

    def log_bar(self, row_dict: dict) -> None:
        """
        Log a single bar (for backtest mode).

        Accumulates rows in memory, call flush() to write to disk.

        """
        with self._instance_lock:
            self._backtest_rows.append(row_dict.copy())

    def flush(self) -> Path | None:
        """
        Write accumulated backtest rows to CSV.

        """
        with self._instance_lock:
            if not self._backtest_rows:
                return None

            try:
                import pandas as pd

                output_dir = self.get_output_dir()
                if not output_dir.exists():
                    self._backtest_rows.clear()
                    return None

                filename = f"signal_debug_{self.symbol}.csv"
                file_path = output_dir / filename

                df = pd.DataFrame(self._backtest_rows)
                self._save_dataframe(df, file_path)

                self._session_file = file_path
                self._backtest_rows.clear()

                print(f"   📝 Flushed {len(df)} rows: {filename}")
                return file_path

            except Exception as e:
                print(f"[SignalLogger] Error flushing {self.symbol}: {e}")
                return None

    def log_dataframe(self, df, filename_prefix: str = "signal_debug") -> Path | None:
        """
        Log entire DataFrame to CSV (for vectorized backtest).

        """
        with self._instance_lock:
            try:
                output_dir = self.get_output_dir()
                if not output_dir.exists():
                    return None

                filename = f"{filename_prefix}_{self.symbol}.csv"
                file_path = output_dir / filename

                self._save_dataframe(df, file_path)

                self._session_file = file_path
                print(f"   📝 Saved DataFrame: {filename}")
                return file_path

            except Exception as e:
                print(f"[SignalLogger] Error saving DataFrame {self.symbol}: {e}")
                return None

    def _save_dataframe(self, df, file_path: Path) -> None:
        """
        Save DataFrame to CSV with proper formatting.

        """
        import pandas as pd

        df_out = df.copy()

        # Format date column if present
        if "date" in df_out.columns:
            df_out["date"] = pd.to_datetime(df_out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Round float columns to 4 decimal places
        for col in df_out.select_dtypes(include=["float64", "float32"]).columns:
            df_out[col] = df_out[col].round(4)

        df_out.to_csv(file_path, index=False)

    @property
    def log_path(self) -> Path | None:
        """Get current log file path."""
        return self._session_file
