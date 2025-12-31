# backtest/execution/account.py
"""Virtual account for tracking capital and equity during backtesting."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

from .position import Position, Trade


@dataclass
class EquityPoint:
    """A point in the equity curve."""

    timestamp: datetime
    cash: float
    positions_value: float
    total_equity: float


@dataclass
class VirtualAccount:
    """
    Virtual trading account for backtesting.

    Tracks cash, positions, equity curve, and trade history.
    """

    initial_capital: float = 100_000.0
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[EquityPoint] = field(default_factory=list)
    commission_per_trade: float = 1.0

    def __post_init__(self):
        """Initialize cash to initial capital."""
        self.cash = self.initial_capital

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol, or None if no position."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if account has position in symbol."""
        return symbol in self.positions

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        side: str = "LONG",
        slippage_pct: float = 0.0,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> bool:
        """
        Open a new position.

        """
        if symbol in self.positions:
            return False  # Already have position

        # Apply slippage
        if side == "LONG":
            fill_price = price * (1 + slippage_pct)
        else:
            fill_price = price * (1 - slippage_pct)

        # Check if we have enough cash
        cost = quantity * fill_price + self.commission_per_trade
        if cost > self.cash:
            return False  # Insufficient funds

        # Deduct cash
        self.cash -= cost

        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            side=side,
            take_profit=take_profit,
            stop_loss=stop_loss,
        )
        self.positions[symbol] = position

        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str = "SIGNAL",
        slippage_pct: float = 0.0,
    ) -> Optional[Trade]:
        """
        Close an existing position.

        """
        if symbol not in self.positions:
            return None

        position = self.positions.pop(symbol)

        # Apply slippage
        if position.side == "LONG":
            fill_price = price * (1 - slippage_pct)
        else:
            fill_price = price * (1 + slippage_pct)

        # Create trade record
        trade = Trade(
            symbol=symbol,
            side=position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=fill_price,
            exit_time=timestamp,
            exit_reason=reason,
            commission=self.commission_per_trade * 2,  # Entry + exit
        )

        # Add proceeds to cash
        proceeds = position.quantity * fill_price - self.commission_per_trade
        self.cash += proceeds

        # Record trade
        self.trades.append(trade)

        return trade

    def get_positions_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total value of open positions.

        """
        total = 0.0
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.entry_price)
            if position.side == "LONG":
                total += position.quantity * price
            else:
                # For shorts, value is 2*entry - current
                total += position.quantity * (2 * position.entry_price - price)
        return total

    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """
        Calculate total account equity.

        """
        return self.cash + self.get_positions_value(prices)

    def check_tp_sl_triggers(
        self,
        prices: Dict[str, float],
        timestamp: datetime,
        slippage_pct: float = 0.0,
    ) -> List[Trade]:
        """
        Check all open positions for TP/SL triggers and close if hit.

        """
        closed_trades = []
        symbols_to_close = []

        # Find positions that should close
        for symbol, position in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue

            # Check TP
            if position.should_take_profit(price):
                symbols_to_close.append((symbol, price, "TAKE_PROFIT"))
            # Check SL
            elif position.should_stop_loss(price):
                symbols_to_close.append((symbol, price, "STOP_LOSS"))

        # Close positions
        for symbol, price, reason in symbols_to_close:
            trade = self.close_position(
                symbol=symbol,
                price=price,
                timestamp=timestamp,
                reason=reason,
                slippage_pct=slippage_pct,
            )
            if trade:
                closed_trades.append(trade)

        return closed_trades

    def record_equity(
        self,
        timestamp: datetime,
        prices: Dict[str, float],
    ):
        """
        Record current equity to equity curve.

        """
        positions_value = self.get_positions_value(prices)
        total_equity = self.cash + positions_value

        point = EquityPoint(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=positions_value,
            total_equity=total_equity,
        )
        self.equity_curve.append(point)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame(
                columns=["timestamp", "cash", "positions_value", "total_equity"]
            )

        data = [
            {
                "timestamp": p.timestamp,
                "cash": p.cash,
                "positions_value": p.positions_value,
                "total_equity": p.total_equity,
            }
            for p in self.equity_curve
        ]
        return pd.DataFrame(data)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        data = [t.to_dict() for t in self.trades]
        return pd.DataFrame(data)

    def reset(self):
        """Reset account to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
