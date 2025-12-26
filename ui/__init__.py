# ui/__init__.py
"""UI components for strategy backtesting."""

from .state import AppState
from .runner import run_backtest

__all__ = ["AppState", "run_backtest"]
