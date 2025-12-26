# backtest/metrics/__init__.py
"""Performance metrics calculation for backtesting."""

from .calculator import (
    BacktestResult,
    DirectionStats,
    calculate_direction_stats,
    calculate_monthly_returns,
    build_backtest_result,
)
from .report import generate_report, save_results, print_summary

__all__ = [
    "BacktestResult",
    "DirectionStats",
    "calculate_direction_stats",
    "calculate_monthly_returns",
    "build_backtest_result",
    "generate_report",
    "save_results",
    "print_summary",
]
