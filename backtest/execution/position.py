# backtest/execution/position.py
"""Position and trade tracking for backtesting."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class TradeDirection(Enum):
    """Direction of a trade."""
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Status/exit reason of a trade."""
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_SIGNAL = "CLOSED_SIGNAL"
    CLOSED_EOD = "CLOSED_EOD"
    CLOSED_MANUAL = "CLOSED_MANUAL"


@dataclass
class Position:
    """Represents an open position."""

    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    direction: TradeDirection = TradeDirection.LONG
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

    @property
    def cost_basis(self) -> float:
        """Total cost to enter position."""
        return self.quantity * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - current_price) / self.entry_price

    def should_take_profit(self, high: float, low: float) -> bool:
        """Check if take profit should trigger using bar high/low."""
        if self.take_profit is None:
            return False
        if self.direction == TradeDirection.LONG:
            return high >= self.take_profit
        else:  # SHORT
            return low <= self.take_profit

    def should_stop_loss(self, high: float, low: float) -> bool:
        """Check if stop loss should trigger using bar high/low."""
        if self.stop_loss is None:
            return False
        if self.direction == TradeDirection.LONG:
            return low <= self.stop_loss
        else:  # SHORT
            return high >= self.stop_loss


@dataclass
class Trade:
    """Represents a completed trade (entry + exit)."""

    trade_id: int
    symbol: str
    direction: TradeDirection
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    position_value: float  # Entry cost (quantity * entry_price)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.CLOSED_SIGNAL
    pnl: float = 0.0  # Net P&L after commission
    pnl_pct: float = 0.0  # P&L as percentage
    commission: float = 0.0
    bars_held: int = 0

    # Internal indices for equity curve building
    _entry_idx: int = 0
    _exit_idx: int = 0

    @property
    def is_winner(self) -> bool:
        """True if trade was profitable."""
        return self.pnl > 0

    @property
    def duration(self) -> float:
        """Trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def duration_minutes(self) -> float:
        """Trade duration in minutes."""
        return self.duration / 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/export."""
        return {
            "Trade_ID": self.trade_id,
            "Symbol": self.symbol,
            "Direction": self.direction.value,
            "Quantity": round(self.quantity, 4),
            "Entry_Price": round(self.entry_price, 4),
            "Entry_Time": self.entry_time,
            "Exit_Price": round(self.exit_price, 4),
            "Exit_Time": self.exit_time,
            "Position_Value": round(self.position_value, 2),
            "Stop_Loss": round(self.stop_loss_price, 4) if self.stop_loss_price else None,
            "Take_Profit": round(self.take_profit_price, 4) if self.take_profit_price else None,
            "Status": self.status.value,
            "PnL": round(self.pnl, 2),
            "PnL_Pct": round(self.pnl_pct, 2),
            "Commission": round(self.commission, 2),
            "Bars_Held": self.bars_held,
            "Duration_Minutes": round(self.duration_minutes, 2),
            "Is_Winner": self.is_winner,
        }
