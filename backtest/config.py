# backtest/config.py
"""Configuration dataclass for backtesting."""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from services.time_centralize_utils import (
    get_et_now,
    get_et_date,
    get_et_month,
    get_month_start,
    get_month_end,
    get_end_of_day,
    parse_datetime,
)


class DateRangeEnd(str, Enum):
    """
    End date options for backtest date range.

    LATEST: Use current system date (today) as end date.
            Cache will be refreshed when stale (> DEFAULT_TOLERANCE_DAYS).
    """

    LATEST = "LATEST"


# Default tolerance for cache reuse (days)
DEFAULT_TOLERANCE_DAYS = 7

# Project root directory (parent of backtest/)
PROJECT_ROOT = Path(__file__).parent.parent

# Standard data directories (matching services/tiingo.py pattern)
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_DATA_DIR = DATA_DIR / "backtest"
OHLC_DIR = BACKTEST_DATA_DIR / "ohlc"
SIGNALS_DIR = BACKTEST_DATA_DIR / "signals"
RESULTS_DIR = BACKTEST_DATA_DIR / "results"


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    # Assets to backtest
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    strategy: str = "ha_mtf_stoch"

    # Date range - YYYY-MM format
    # from_date: Start month (e.g., "2023-01")
    # to_date: End month (e.g., "2024-12") or DateRangeEnd.LATEST for current date
    from_date: str = "2023-01"
    to_date: Union[str, DateRangeEnd] = DateRangeEnd.LATEST

    # Computed date range (set in __post_init__, YYYY-MM-DD format)
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Chunking parameters
    chunk_size: int = 0  # Bars per chunk (0 or None for auto-calculation)
    warmup_size: int = 7_000  # Warmup bars for indicator convergence

    # Execution simulation
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_per_trade: float = 1.0  # $ per trade

    # TP/SL override for backtesting (None = use strategy class defaults)
    # These override the strategy's STOP_LOSS_PCT and TAKE_PROFIT_PCT
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    # Parallelism
    max_workers: int = 0  # CPU cores (0 or None for auto-calculation)

    # Caching
    use_signal_cache: bool = True
    force_regenerate: bool = False  # Ignore cache if True

    # Signal generation options
    ignore_position: bool = True  # Generate all potential entry signals
                                  # (simulator handles position management in Step 3)

    # Output options
    save_results: bool = True  # Save results to files (trades.csv, signals.csv, dashboard.html, etc.)
    hide_signals: bool = False  # Hide signal config in dashboard (show '***')

    # Data paths (absolute, based on project root)
    ohlc_dir: str = field(default_factory=lambda: str(OHLC_DIR))
    signals_dir: str = field(default_factory=lambda: str(SIGNALS_DIR))
    results_dir: str = field(default_factory=lambda: str(RESULTS_DIR))

    def __post_init__(self):
        """Validate configuration and compute date range from YYYY-MM format."""
        # Compute start_date from from_date (YYYY-MM)
        if self.start_date is None:
            self.start_date = get_month_start(self.from_date).strftime("%Y-%m-%d")

        # Compute end_date from to_date (YYYY-MM or LATEST)
        if self.end_date is None:
            if self.to_date == DateRangeEnd.LATEST or self.to_date == "LATEST":
                # Use current date
                self.end_date = get_et_date()
            else:
                # Use end of specified month
                self.end_date = get_month_end(self.to_date).strftime("%Y-%m-%d")

        # Validate
        if self.chunk_size and self.warmup_size >= self.chunk_size:
            raise ValueError(
                f"warmup_size ({self.warmup_size}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.chunk_size and self.chunk_size < 1000:
            raise ValueError(f"chunk_size ({self.chunk_size}) too small, use >= 1000")
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
