# strategies/_trading_mech.py - Trading mechanics base class.
"""
_trading_mech.py - Core trading mechanics for all strategies.

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
from services.exits import check_exit
from services.logger import SignalLogger


# Type alias for order handler: (symbol, action, quantity) -> bool
OrderHandler = Callable[[str, str, float], bool]


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
    QUANTITY: float = 10   # Default units per trade (shares/coins)

    # === EXIT STRATEGY (override in subclass if needed) ===
    STOP_LOSS_PCT: float = 1.0     # Default 1% stop loss
    TAKE_PROFIT_PCT: float = 2.0   # Default 2% take profit

    # === LOGGING CONFIGURATION ===
    PERIODIC_LOG_INTERVAL: int = 1800  # 30 minutes in seconds

    # === SIGNAL BAR INDEX ===
    # Which bar the strategy acts on. After finalize_signals() shifts by 1:
    # - signal[-1] corresponds to data from bar[-2]
    # - We trade on the completed bar's close, not the current incomplete bar
    # Override this in a subclass if your strategy uses a different bar.
    TRIGGER_BAR_INDEX: int = -2
    
    def __init__(
        self,
        symbol: str,
        tiingo: Any,
        *,
        time_provider: Callable[[], float] | None = None,
        order_handler: OrderHandler | None = None,
        position_checker: Callable[[str], bool] | None = None,
    ):
        """
        Initialize strategy with symbol and data service.

        """
        self.symbol = self._normalize_symbol(symbol)
        self.tiingo = tiingo
        self._last_check: float = 0
        self._last_strategy_log: float = 0
        self._strategy_logger: SignalLogger | None = None

        # Dependency injection for backtesting (defaults to live behavior)
        self._time_provider = time_provider or time.time
        self._order_handler = order_handler
        self._position_checker = position_checker
    
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
        debug CSV export during backtesting. Uses SignalLogger in backtest mode.

        Example usage in execute():
            df = self.compute_signals(df)
            self.capture_debug_row(df)  # <-- Add this line
        """
        # Only capture in backtest mode
        if SignalLogger._mode != "backtest":
            return

        if df is None or len(df) == 0:
            return

        # Log the last row via SignalLogger
        logger = self._get_strategy_logger()
        last_row = df.iloc[-1].to_dict()
        logger.log_bar(last_row)

    @staticmethod
    def to_df(ohlc: list[dict]) -> pd.DataFrame:
        """Convert list of OHLC dicts to DataFrame."""
        return pd.DataFrame(ohlc)

    @classmethod
    def finalize_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and stabilize signals before returning.

        This centralizes the 'Safety Shift' logic:
        1. Shifts the 'signal' column by 1 to prevent repainting.
        2. Fills NaNs with 0 (HOLD).
        3. Ensures signal is integer type.

        Usage at the end of compute_signals:
            return cls.finalize_signals(df)
        """
        if "signal" not in df.columns:
            return df

        df = df.copy()
        # The Safety Shift: Signal at N is based on completion of N-1
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return df
    
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """
        Normalize symbol format based on active broker.

        - IBKR: Keep as-is (e.g., "QQQ", "SPY")
        - Hyperliquid: Simple symbol (e.g., "BTC", "ETH") - remove suffixes

        This ensures consistency between:
        - Strategy symbol (self.symbol)
        - Position tracking (context.positions)
        - Order placement (broker.place_order)
        """
        symbol = symbol.upper()

        # For Hyperliquid, normalize to simple symbol (BTC, ETH, etc.)
        if context.active_broker == "hyperliquid":
            # Remove common suffixes and separators
            for suffix in ["USDT", "USD", "PERP", "-PERP", "_PERP"]:
                if symbol.endswith(suffix):
                    symbol = symbol[:-len(suffix)]
                    break
            symbol = symbol.replace("/", "").replace("-", "").replace("_", "")

        return symbol

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
        """
        Check if we currently hold a position in this symbol.
        Checks both IBKR real-time data and local tracking to prevent double-entries.
        """
        if self._position_checker:
            return self._position_checker(self.symbol)
            
        # 1. Check local pos_manager (immediate update after submission)
        if self.is_tracked():
            return True

        # 2. Check IBKR actual positions (source of truth from exchange)
        account_positions = context.get_positions_for_account()
        position = account_positions.get(self.symbol)
        return position is not None and position.get("qty", 0) > 0

    def get_position_qty(self) -> float:
        """Get current position quantity (0 if none)."""
        # Use get_positions_for_account() which handles "ACCOUNT:SYMBOL" keys
        account_positions = context.get_positions_for_account()
        position = account_positions.get(self.symbol)
        return position.get("qty", 0) if position else 0

    def buy(self, quantity: float | None = None) -> bool:
        """Submit a buy order (low-level, no tracking)."""
        qty = quantity or self.QUANTITY
        if self._order_handler:
            return self._order_handler(self.symbol, "BUY", qty)
        return order_service.submit_order(self.symbol, "BUY", qty)

    def sell(self, quantity: float | None = None) -> bool:
        """Submit a sell order (low-level, no tracking)."""
        qty = quantity or self.get_position_qty() or self.QUANTITY
        if self._order_handler:
            return self._order_handler(self.symbol, "SELL", qty)
        return order_service.submit_order(self.symbol, "SELL", qty)

    # === POSITION MANAGEMENT WITH TRACKING ===

    def open_long(
        self,
        quantity: float | None = None,
        entry_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None
    ) -> bool:
        """
        Open a LONG position with tracking.

        """
        qty = quantity or self.QUANTITY

        # Auto-calculate TP/SL from percentages if entry_price provided
        if entry_price and (take_profit is None or stop_loss is None):
            calc_tp, calc_sl = self.calculate_stops(entry_price, "LONG")
            tp = take_profit if take_profit is not None else calc_tp
            sl = stop_loss if stop_loss is not None else calc_sl
        else:
            tp = take_profit
            sl = stop_loss

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
            self.log(f"Opened LONG {qty} units [TP: ${tp:.2f}] [SL: ${sl:.2f}]" if tp and sl else f"Opened LONG {qty} units")

        return success

    def open_short(
        self,
        quantity: float | None = None,
        entry_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None
    ) -> bool:
        """
        Open a SHORT position with tracking.

        """
        qty = quantity or self.QUANTITY

        # Auto-calculate TP/SL from percentages if entry_price provided
        if entry_price and (take_profit is None or stop_loss is None):
            calc_tp, calc_sl = self.calculate_stops(entry_price, "SHORT")
            tp = take_profit if take_profit is not None else calc_tp
            sl = stop_loss if stop_loss is not None else calc_sl
        else:
            tp = take_profit
            sl = stop_loss

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
            self.log(f"Opened SHORT {qty} units [TP: ${tp:.2f}] [SL: ${sl:.2f}]" if tp and sl else f"Opened SHORT {qty} units")

        return success

    def close_position(self, quantity: float | None = None) -> bool:
        """
        Close the current position with tracking.

        Automatically determines if LONG (sell) or SHORT (buy to cover).

        """
        # In backtest mode, delegate to order handler
        if self._order_handler:
            qty = quantity or self.QUANTITY
            return self._order_handler(self.symbol, "CLOSE", qty)

        tracked = self.get_tracked_position()

        if not tracked:
            # No tracked position, try to close broker position
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
            self.log(f"Closed {action} {qty} units")

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

    def calculate_stops(self, entry_price: float, direction: str) -> tuple[float, float]:
        """
        Calculate TP and SL prices from percentages.

        """
        if direction == "LONG":
            tp = entry_price * (1 + self.TAKE_PROFIT_PCT / 100)
            sl = entry_price * (1 - self.STOP_LOSS_PCT / 100)
        else:  # SHORT
            tp = entry_price * (1 - self.TAKE_PROFIT_PCT / 100)
            sl = entry_price * (1 + self.STOP_LOSS_PCT / 100)
        return round(tp, 2), round(sl, 2)

    def check_stops(self, current_price: float) -> str | None:
        """
        Check if TP or SL should trigger.

        """
        # Check SL first (risk management priority)
        if self.check_stop_loss(current_price):
            return "SL"
        if self.check_take_profit(current_price):
            return "TP"
        return None

    async def check_and_close_if_stopped(self) -> bool:
        """
        Check TP/SL and close position if triggered.
        Call this at start of execute() before entry logic.

        """
        if not self.is_tracked():
            return False

        # Get current price from context or fetch it
        current_price = context.latest_prices.get(self.symbol)
        if current_price is None:
            # Try to fetch from tiingo
            try:
                ohlc = await self.tiingo.get_ohlc(self.symbol, days=1, interval="1min")
                if ohlc:
                    current_price = ohlc[-1].get("close")
            except Exception:
                pass

        if current_price is None:
            return False

        exit_reason = self.check_stops(current_price)
        if exit_reason:
            tracked = self.get_tracked_position()
            entry_price = tracked.get("entry_price", 0)
            if entry_price:
                pnl_pct = ((current_price - entry_price) / entry_price * 100)
                self.log(f"{exit_reason} @ ${current_price:.2f} ({pnl_pct:+.1f}%)")
            else:
                self.log(f"{exit_reason} @ ${current_price:.2f}")
            self.close_position()
            return True
        return False

    @classmethod
    def exit_check(cls, position: dict, bar_data: dict) -> dict | None:
        """
        Check if position should exit based on current bar data.
        Called by backtest simulator each bar while in position.

        Override this method for complex exit logic using indicators.

        """
        # Use shared exit logic from services/exits.py
        return check_exit(
            direction=position.get("direction", "LONG"),
            entry_price=position.get("entry_price", 0),
            tp_price=position.get("tp_price"),
            sl_price=position.get("sl_price"),
            bar_high=bar_data["high"],
            bar_low=bar_data["low"],
            bar_open=bar_data["open"],
        )

    def log(self, message: str) -> None:
        """Log a message (will be sent to Telegram). Disabled in backtest mode."""
        if self._order_handler:
            return  # Skip logging in backtest mode
        context.log(f"[{self.NAME}] {message}", "trade")

    # === STRATEGY CSV LOGGING ===

    def _get_strategy_logger(self) -> SignalLogger:
        """Get or create strategy logger for this symbol."""
        if self._strategy_logger is None:
            self._strategy_logger = SignalLogger.get_or_create(
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

        Uses self.TRIGGER_BAR_INDEX to determine which bar triggered the signal.
        This ensures logging reflects the exact bar the strategy acted on.

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
            event_type=event_type,
            row_index=self.TRIGGER_BAR_INDEX
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

