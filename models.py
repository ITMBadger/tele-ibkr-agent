# models.py - Pure data classes with NO dependencies.
"""
This module contains all shared data classes used across the application.
Having them in a separate file prevents circular imports.

All classes here should be:
- Pure dataclasses or TypedDicts
- Have NO imports from other project modules
- Be importable by any module in the project
"""

from dataclasses import dataclass


@dataclass
class TradeSignal:
    """Represents a trade signal from strategy to IBKR."""
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: int
    order_type: str = "MKT"
    limit_price: float | None = None
    strategy_id: str = "manual"  # Strategy ID that generated this signal


@dataclass
class LogMessage:
    """Represents a message from IBKR to Telegram."""
    message: str
    level: str = "info"  # "info", "warning", "error", "trade"

