# backtest/__init__.py
"""Backtesting engine for strategy validation."""

from .config import BacktestConfig
from .engine_multiprocessing import BacktestEngine
from .engine_vectorized import VectorizedBacktestEngine
from .metrics import BacktestResult, DirectionStats
from .execution import TradeDirection, TradeStatus

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "VectorizedBacktestEngine",
    "BacktestResult",
    "DirectionStats",
    "TradeDirection",
    "TradeStatus",
]
