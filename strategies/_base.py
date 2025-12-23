# strategies/_base.py - Base class for all trading strategies.
"""
base.py - Base class for all trading strategies.

Each strategy is a self-contained unit with ALL parameters fixed inside.
No external configuration, no params passing - just activate and run.

To create a new strategy:
1. Subclass BaseStrategy
2. Set ID, NAME, DESCRIPTION, INTERVAL as class attributes
3. Override execute() with your trading logic
4. Import and add to STRATEGY_REGISTRY in __init__.py

Position Management:
- Use open_long() / open_short() to enter with tracking
- Use close_position() to exit with tracking
- Positions are saved to data/positions.json for persistence across restarts
"""

from abc import ABC, abstractmethod
from typing import Any, Callable
import time

import pandas as pd

import context
from services import pos_manager, order_service
from services.logger import StrategyLogger


# Type alias for order handler: (symbol, action, quantity) -> bool
OrderHandler = Callable[[str, str, int], bool]

# Type alias for debug collector: (row_dict) -> None
DebugCollector = Callable[[dict], None]


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Each strategy defines ALL its parameters as class attributes.
    The only external input is the symbol to trade.
    """
    
    # === MUST OVERRIDE THESE ===
    ID: str = ""           # Unique identifier (e.g., "1", "2")
    NAME: str = ""         # Human-readable name
    DESCRIPTION: str = ""  # What the strategy does
    INTERVAL: int = 60     # Check interval in seconds

    # === OPTIONAL: Common attributes ===
    QUANTITY: int = 10     # Default shares per trade

    # === OPTIONAL: Position Management (override in subclass) ===
    TAKE_PROFIT_PCT: float | None = None   # e.g., 2.0 for +2%
    STOP_LOSS_PCT: float | None = None     # e.g., 1.0 for -1%
    TAKE_PROFIT_PRICE: float | None = None # Absolute TP price
    STOP_LOSS_PRICE: float | None = None   # Absolute SL price

    # === LOGGING CONFIGURATION ===
    PERIODIC_LOG_INTERVAL: int = 1800  # 30 minutes in seconds
    
    def __init__(
        self,
        symbol: str,
        tiingo: Any,
        *,
        time_provider: Callable[[], float] | None = None,
        order_handler: OrderHandler | None = None,
        position_checker: Callable[[str], bool] | None = None,
        debug_collector: DebugCollector | None = None,
    ):
        """
        Initialize strategy with symbol and data service.

        Args:
            symbol: Stock symbol to trade (e.g., "QQQ")
            tiingo: TiingoService instance for market data
            time_provider: Optional callable returning current time (for backtesting)
            order_handler: Optional callable to handle orders (for backtesting)
            position_checker: Optional callable to check position (for backtesting)
            debug_collector: Optional callable to collect debug rows (for backtesting)
        """
        self.symbol = symbol.upper()
        self.tiingo = tiingo
        self._last_check: float = 0
        self._last_strategy_log: float = 0
        self._strategy_logger: StrategyLogger | None = None

        # Dependency injection for backtesting (defaults to live behavior)
        self._time_provider = time_provider or time.time
        self._order_handler = order_handler
        self._position_checker = position_checker
        self._debug_collector = debug_collector
    
    @abstractmethod
    async def execute(self) -> None:
        """
        Execute the strategy logic.

        This method should:
        1. Fetch required market data
        2. Calculate signals using indicators
        3. Submit orders via self.buy()/self.sell() or order_service.submit_order()

        All parameters should be class attributes, not method arguments.
        """
        pass

    @classmethod
    def compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trading signals for the entire DataFrame (vectorized).

        This is the core signal logic shared by live trading and backtesting.
        The method should be pure (no side effects, no position tracking).

        Args:
            df: DataFrame with columns: date, open, high, low, close, volume
                Must be sorted by date ascending.

        Returns:
            Same DataFrame with additional columns:
            - 'signal': 1=BUY, -1=SELL, 0=HOLD
            - Indicator columns (e.g., 'ema', 'rsi') for debugging/logging

        Note:
            - Position management is handled by the caller
            - Early rows may have NaN/0 signals due to indicator warmup
            - Override in subclass; default raises NotImplementedError
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement compute_signals(). "
            "Use bar-by-bar backtest instead of vectorized."
        )

    def capture_debug_row(self, df: pd.DataFrame) -> None:
        """
        Capture the last row of a signal DataFrame for debug export.

        Call this in execute() after compute_signals() to enable bar-by-bar
        debug CSV export during backtesting. Uses dependency injection -
        the debug_collector is passed during strategy initialization.

        Args:
            df: DataFrame returned by compute_signals() with all indicator columns

        Example usage in execute():
            df = self.compute_signals(df)
            self.capture_debug_row(df)  # <-- Add this line
        """
        # Skip if no collector injected (live mode or collector not needed)
        if self._debug_collector is None:
            return

        if df is None or len(df) == 0:
            return

        # Pass the last row to the injected collector
        last_row = df.iloc[-1].to_dict()
        self._debug_collector(last_row)

    @staticmethod
    def to_df(ohlc: list[dict]) -> pd.DataFrame:
        """Convert list of OHLC dicts to DataFrame."""
        return pd.DataFrame(ohlc)
    
    def should_run(self) -> bool:
        """
        Check if enough time has passed since last execution.

        Uses clock-aligned intervals to prevent drift:
        - INTERVAL=60 → runs at :00, 1:00, 2:00, etc.
        - INTERVAL=300 → runs at :00, 5:00, 10:00, etc.

        This ensures consistent timing regardless of when strategy was activated
        or PC performance variations.
        """
        now = self._time_provider()
        
        # Calculate which interval we're in (aligned to clock)
        current_interval = int(now // self.INTERVAL)
        last_interval = int(self._last_check // self.INTERVAL)
        
        if current_interval > last_interval:
            self._last_check = now
            return True
        return False
    
    def has_position(self) -> bool:
        """Check if we currently hold a position in this symbol."""
        if self._position_checker:
            return self._position_checker(self.symbol)
        # Use get_positions_for_account() which handles "ACCOUNT:SYMBOL" keys
        account_positions = context.get_positions_for_account()
        position = account_positions.get(self.symbol)
        return position is not None and position.get("qty", 0) > 0

    def get_position_qty(self) -> int:
        """Get current position quantity (0 if none)."""
        # Use get_positions_for_account() which handles "ACCOUNT:SYMBOL" keys
        account_positions = context.get_positions_for_account()
        position = account_positions.get(self.symbol)
        return position.get("qty", 0) if position else 0
    
    def buy(self, quantity: int | None = None) -> bool:
        """Submit a buy order (low-level, no tracking)."""
        qty = quantity or self.QUANTITY
        if self._order_handler:
            return self._order_handler(self.symbol, "BUY", qty)
        return order_service.submit_order(self.symbol, "BUY", qty)

    def sell(self, quantity: int | None = None) -> bool:
        """Submit a sell order (low-level, no tracking)."""
        qty = quantity or self.get_position_qty() or self.QUANTITY
        if self._order_handler:
            return self._order_handler(self.symbol, "SELL", qty)
        return order_service.submit_order(self.symbol, "SELL", qty)

    # === POSITION MANAGEMENT WITH TRACKING ===

    def open_long(
        self,
        quantity: int | None = None,
        entry_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None
    ) -> bool:
        """
        Open a LONG position with tracking.

        Args:
            quantity: Number of shares (uses QUANTITY if None)
            entry_price: Entry price for tracking (optional)
            take_profit: Take profit price (uses class default if None)
            stop_loss: Stop loss price (uses class default if None)

        Returns:
            bool: True if order submitted successfully
        """
        qty = quantity or self.QUANTITY
        tp = take_profit or self.TAKE_PROFIT_PRICE
        sl = stop_loss or self.STOP_LOSS_PRICE

        if self._order_handler:
            success = self._order_handler(self.symbol, "BUY", qty)
        else:
            success = order_service.submit_order(self.symbol, "BUY", qty)

        if success and not self._order_handler:
            pos_manager.save_position(
                symbol=self.symbol,
                account=context.current_account or "",
                strategy_id=self.ID,
                action="LONG",
                quantity=qty,
                entry_price=entry_price,
                take_profit=tp,
                stop_loss=sl
            )
            self.log(f"Opened LONG {qty} shares")

        return success

    def open_short(
        self,
        quantity: int | None = None,
        entry_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None
    ) -> bool:
        """
        Open a SHORT position with tracking.

        Args:
            quantity: Number of shares (uses QUANTITY if None)
            entry_price: Entry price for tracking (optional)
            take_profit: Take profit price (uses class default if None)
            stop_loss: Stop loss price (uses class default if None)

        Returns:
            bool: True if order submitted successfully
        """
        qty = quantity or self.QUANTITY
        tp = take_profit or self.TAKE_PROFIT_PRICE
        sl = stop_loss or self.STOP_LOSS_PRICE

        if self._order_handler:
            success = self._order_handler(self.symbol, "SELL", qty)
        else:
            success = order_service.submit_order(self.symbol, "SELL", qty)

        if success and not self._order_handler:
            pos_manager.save_position(
                symbol=self.symbol,
                account=context.current_account or "",
                strategy_id=self.ID,
                action="SHORT",
                quantity=qty,
                entry_price=entry_price,
                take_profit=tp,
                stop_loss=sl
            )
            self.log(f"Opened SHORT {qty} shares")

        return success

    def close_position(self, quantity: int | None = None) -> bool:
        """
        Close the current position with tracking.

        Automatically determines if LONG (sell) or SHORT (buy to cover).

        Args:
            quantity: Number of shares to close (all if None)

        Returns:
            bool: True if order submitted successfully
        """
        # In backtest mode, delegate to order handler
        if self._order_handler:
            qty = quantity or self.QUANTITY
            return self._order_handler(self.symbol, "CLOSE", qty)

        tracked = self.get_tracked_position()

        if not tracked:
            # No tracked position, try to close IBKR position
            qty = quantity or self.get_position_qty()
            if qty > 0:
                return self.sell(qty)
            return False

        qty = quantity or tracked.get("quantity", self.get_position_qty())
        action = tracked.get("action", "LONG")

        # Close opposite to opening action
        if action == "LONG":
            success = order_service.submit_order(self.symbol, "SELL", qty)
        else:  # SHORT
            success = order_service.submit_order(self.symbol, "BUY", qty)

        if success:
            pos_manager.remove_position(self.symbol)
            self.log(f"Closed {action} {qty} shares")

        return success

    def get_tracked_position(self) -> dict | None:
        """Get tracked position data from pos_manager."""
        return pos_manager.get_position(self.symbol)

    def is_tracked(self) -> bool:
        """Check if this symbol has a tracked position."""
        return self.get_tracked_position() is not None

    def update_stops(
        self,
        take_profit: float | None = None,
        stop_loss: float | None = None
    ) -> bool:
        """
        Update take profit and/or stop loss for tracked position.

        Args:
            take_profit: New take profit price
            stop_loss: New stop loss price

        Returns:
            bool: True if updated successfully
        """
        updates = {}
        if take_profit is not None:
            updates["take_profit"] = take_profit
        if stop_loss is not None:
            updates["stop_loss"] = stop_loss

        if not updates:
            return False

        return pos_manager.update_position(self.symbol, **updates)

    def check_take_profit(self, current_price: float) -> bool:
        """
        Check if take profit should trigger.

        Args:
            current_price: Current market price

        Returns:
            bool: True if TP should trigger
        """
        tracked = self.get_tracked_position()
        if not tracked:
            return False

        tp = tracked.get("take_profit")
        if tp is None:
            return False

        action = tracked.get("action", "LONG")

        if action == "LONG":
            return current_price >= tp
        else:  # SHORT
            return current_price <= tp

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Check if stop loss should trigger.

        Args:
            current_price: Current market price

        Returns:
            bool: True if SL should trigger
        """
        tracked = self.get_tracked_position()
        if not tracked:
            return False

        sl = tracked.get("stop_loss")
        if sl is None:
            return False

        action = tracked.get("action", "LONG")

        if action == "LONG":
            return current_price <= sl
        else:  # SHORT
            return current_price >= sl

    def log(self, message: str) -> None:
        """Log a message (will be sent to Telegram). Disabled in backtest mode."""
        if self._order_handler:
            return  # Skip logging in backtest mode
        context.log(f"[{self.NAME}] {message}", "trade")

    # === STRATEGY CSV LOGGING ===

    def _get_strategy_logger(self) -> StrategyLogger:
        """Get or create strategy logger for this symbol."""
        if self._strategy_logger is None:
            self._strategy_logger = StrategyLogger.get_or_create(
                symbol=self.symbol,
                strategy_name=self.NAME
            )
        return self._strategy_logger

    def should_log_periodic(self) -> bool:
        """
        Check if enough time has passed for periodic logging.

        Uses clock-aligned intervals to prevent drift (same logic as should_run).
        Disabled in backtest mode for performance.
        """
        # Skip logging in backtest mode (when order_handler is set)
        if self._order_handler:
            return False

        now = self._time_provider()

        # Calculate which interval we're in (aligned to clock)
        current_interval = int(now // self.PERIODIC_LOG_INTERVAL)
        last_interval = int(self._last_strategy_log // self.PERIODIC_LOG_INTERVAL)

        return current_interval > last_interval

    def log_strategy_data(
        self,
        ohlc_bars: list[dict],
        indicator_columns: dict[str, list[float | None]] | None = None,
        signal: str = "",
        triggered: bool = False,
        event_type: str = "periodic"
    ) -> None:
        """
        Log OHLC data with indicators to CSV.

        Call this method from execute() on two occasions:
        1. When a trigger happens (signal buy/sell)
        2. Every 30 minutes (periodic snapshot)

        Disabled in backtest mode for performance.

        Args:
            ohlc_bars: Full OHLC data (all bars used for calculation)
            indicator_columns: Dict of indicator name -> list of values per bar
                               e.g., {"ema_200": [val1, val2, ...]}
            signal: "BUY", "SELL", "HOLD", or "" for no signal
            triggered: Whether an order was actually placed
            event_type: "signal" or "periodic"
        """
        # Skip logging in backtest mode (when order_handler is set)
        if self._order_handler:
            return

        logger = self._get_strategy_logger()

        # Debug: show when logging is attempted
        if event_type == "periodic":
            print(f"   📝 [{self.NAME}] Periodic log triggered for {self.symbol}")

        success = logger.log_event(
            ohlc_bars=ohlc_bars,
            indicator_columns=indicator_columns,
            signal=signal,
            triggered=triggered,
            event_type=event_type
        )

        if success and event_type == "periodic":
            self._last_strategy_log = self._time_provider()
            print(f"   ✅ [{self.NAME}] Periodic log saved successfully")
        elif not success and event_type == "periodic":
            print(f"   ❌ [{self.NAME}] Periodic log FAILED")

    @classmethod
    def info(cls) -> str:
        """Return strategy info as formatted string."""
        return (
            f"**Strategy {cls.ID}**: {cls.NAME}\n"
            f"  • {cls.DESCRIPTION}\n"
            f"  • Interval: {cls.INTERVAL}s"
        )

