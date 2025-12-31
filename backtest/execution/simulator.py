# backtest/execution/simulator.py
"""Trade execution simulation for backtesting - Sequential position tracking."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .position import Position, Trade, TradeDirection, TradeStatus
from services.exits import check_exit


def _normalize_dates_to_naive_et(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Normalize date column to naive ET format.

    - If timezone-aware: convert to ET, then strip timezone
    - If naive: assume already ET (project standard)

    Args:
        df: DataFrame with date column
        date_column: Name of the date column

    Returns:
        DataFrame with naive ET dates
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_column])

    if dates.dt.tz is not None:
        # Has timezone - convert to ET then strip
        dates = dates.dt.tz_convert("America/New_York").dt.tz_localize(None)
    # If naive, assume already ET (no conversion needed)

    df[date_column] = dates
    return df


class SimulatedBroker:
    """
    Simulates trade execution for backtesting.

    Supports both LONG and SHORT positions with:
    - Take profit and stop loss (configurable or from strategy class)
    - Slippage and commission
    - Bar-by-bar tracking
    - Full equity curve at every bar
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.001,
        commission_per_trade: float = 1.0,
        position_size_pct: float = 95.0,
        strategy_class: type = None,
        stop_loss_pct: float = None,
        take_profit_pct: float = None,
    ):
        """
        Initialize simulated broker.

        Args:
            initial_capital: Starting capital
            slippage_pct: Slippage percentage (0.001 = 0.1%)
            commission_per_trade: Commission per trade in dollars
            position_size_pct: Percentage of capital to use per trade
            strategy_class: Strategy class to get TP/SL from class attributes
            stop_loss_pct: Override stop loss % (None = use strategy default)
            take_profit_pct: Override take profit % (None = use strategy default)
        """
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        self.position_size_pct = position_size_pct
        self.strategy_class = strategy_class

        # Get TP/SL: config override > strategy class > hardcoded default
        if take_profit_pct is not None:
            self.take_profit_pct = take_profit_pct
        elif strategy_class:
            self.take_profit_pct = getattr(strategy_class, 'TAKE_PROFIT_PCT', 2.0)
        else:
            self.take_profit_pct = 2.0

        if stop_loss_pct is not None:
            self.stop_loss_pct = stop_loss_pct
        elif strategy_class:
            self.stop_loss_pct = getattr(strategy_class, 'STOP_LOSS_PCT', 1.0)
        else:
            self.stop_loss_pct = 1.0

        # State
        self.cash = initial_capital
        self.trades: List[Trade] = []

    def run_simulation(
        self,
        signals: List[Dict[str, Any]],
        ohlc_df: pd.DataFrame,
        indicator_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run trading simulation with sequential position tracking.

        Supports both LONG (BUY) and SHORT (SELL) signals.
        Uses strategy's exit_check() method for complex exit logic.

        Args:
            signals: List of signal dicts with keys: timestamp, action, symbol, price, quantity
            ohlc_df: OHLC DataFrame with columns: date, open, high, low, close
            indicator_df: Optional DataFrame with all indicators (for complex exit logic)

        Returns:
            Dict with trades, equity_curve DataFrame (DatetimeIndex), final_equity, etc.
        """
        if not signals:
            return self._empty_results(ohlc_df)

        # Prepare data - normalize to naive ET
        ohlc_df = _normalize_dates_to_naive_et(ohlc_df, "date")
        ohlc_df = ohlc_df.sort_values("date").reset_index(drop=True)

        n_bars = len(ohlc_df)
        timestamps = ohlc_df["date"].values
        opens = ohlc_df["open"].values
        highs = ohlc_df["high"].values
        lows = ohlc_df["low"].values
        closes = ohlc_df["close"].values

        # Create timestamp -> bar_index lookup
        timestamp_to_idx = {pd.Timestamp(ts): idx for idx, ts in enumerate(timestamps)}

        # Prepare signals - normalize to naive ET
        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df = _normalize_dates_to_naive_et(signals_df, "timestamp")
        signals_df = signals_df.sort_values("timestamp").reset_index(drop=True)

        # Filter to entry signals (BUY = LONG entry, SELL = SHORT entry)
        entry_signals = signals_df[signals_df["action"].isin(["BUY", "SELL"])].to_dict("records")

        print(f"  Processing {len(entry_signals)} entry signals (BUY/SELL)...")

        # Initialize equity array
        equity = np.full(n_bars, self.initial_capital, dtype=np.float64)

        # Track state
        last_exit_idx = -1
        trade_id = 0
        realized_pnl = 0.0
        tp_exits = 0
        sl_exits = 0
        long_trades = 0
        short_trades = 0

        pos_frac = self.position_size_pct / 100.0

        for signal in entry_signals:
            signal_time = pd.Timestamp(signal["timestamp"])

            # Find bar index for this signal
            if signal_time not in timestamp_to_idx:
                bar_idx = self._find_nearest_bar(signal_time, timestamps)
                if bar_idx is None:
                    continue
            else:
                bar_idx = timestamp_to_idx[signal_time]

            # Skip if signal is before last position close
            if bar_idx <= last_exit_idx:
                continue

            # Determine direction
            action = signal["action"]
            direction = TradeDirection.LONG if action == "BUY" else TradeDirection.SHORT

            symbol = signal["symbol"]
            entry_price = signal["price"]

            # Apply slippage to entry
            if direction == TradeDirection.LONG:
                fill_price = entry_price * (1 + self.slippage_pct)
            else:  # SHORT - we want worse price (lower)
                fill_price = entry_price * (1 - self.slippage_pct)

            # Calculate position size
            current_cash = self.initial_capital + realized_pnl
            pos_value = current_cash * pos_frac

            if pos_value < 10.0:
                continue

            quantity = pos_value / fill_price

            # Calculate TP/SL prices
            take_profit_price = None
            stop_loss_price = None

            if self.take_profit_pct is not None:
                if direction == TradeDirection.LONG:
                    take_profit_price = fill_price * (1 + self.take_profit_pct / 100)
                else:  # SHORT
                    take_profit_price = fill_price * (1 - self.take_profit_pct / 100)

            if self.stop_loss_pct is not None:
                if direction == TradeDirection.LONG:
                    stop_loss_price = fill_price * (1 - self.stop_loss_pct / 100)
                else:  # SHORT
                    stop_loss_price = fill_price * (1 + self.stop_loss_pct / 100)

            # Track bar-by-bar until exit
            exit_price = None
            exit_idx = None
            status = TradeStatus.CLOSED_MANUAL

            # Build position dict for exit_check
            direction_str = "LONG" if direction == TradeDirection.LONG else "SHORT"
            position = {
                "entry_price": fill_price,
                "direction": direction_str,
                "tp_price": take_profit_price,
                "sl_price": stop_loss_price,
            }

            for i in range(bar_idx + 1, n_bars):
                bar_high = highs[i]
                bar_low = lows[i]
                bar_open = opens[i]
                bar_close = closes[i]

                # Build bar_data dict for exit_check
                bar_data = {
                    "high": bar_high,
                    "low": bar_low,
                    "open": bar_open,
                    "close": bar_close,
                }

                # Add indicator values if available
                if indicator_df is not None and i < len(indicator_df):
                    try:
                        bar_data.update(indicator_df.iloc[i].to_dict())
                    except Exception:
                        pass  # Continue with OHLC only

                # Use strategy's exit_check if available, otherwise fallback to default
                if self.strategy_class and hasattr(self.strategy_class, 'exit_check'):
                    exit_result = self.strategy_class.exit_check(position, bar_data)
                else:
                    exit_result = self._default_exit_check(position, bar_data)

                if exit_result:
                    exit_price = exit_result["price"]
                    exit_idx = i
                    exit_status = exit_result["status"]
                    if exit_status == "CLOSED_SL":
                        status = TradeStatus.CLOSED_SL
                        sl_exits += 1
                    elif exit_status == "CLOSED_TP":
                        status = TradeStatus.CLOSED_TP
                        tp_exits += 1
                    else:
                        status = TradeStatus.CLOSED_SIGNAL
                    break

            # If never exited, close at last bar
            if exit_price is None:
                exit_idx = n_bars - 1
                exit_price = closes[exit_idx]
                status = TradeStatus.CLOSED_MANUAL

            # Apply slippage to exit
            if direction == TradeDirection.LONG:
                exit_fill_price = exit_price * (1 - self.slippage_pct)
            else:  # SHORT - buying back
                exit_fill_price = exit_price * (1 + self.slippage_pct)

            # Calculate P&L
            if direction == TradeDirection.LONG:
                gross_pnl = (exit_fill_price - fill_price) * quantity
            else:  # SHORT
                gross_pnl = (fill_price - exit_fill_price) * quantity

            commission = self.commission_per_trade * 2  # Entry + exit
            net_pnl = gross_pnl - commission
            pnl_pct = (net_pnl / pos_value) * 100 if pos_value > 0 else 0.0

            # Update realized P&L
            realized_pnl += net_pnl

            # Build equity for this trade period
            # Before entry: flat at previous realized
            if last_exit_idx >= 0 and bar_idx > last_exit_idx + 1:
                equity[last_exit_idx + 1:bar_idx] = self.initial_capital + realized_pnl - net_pnl

            # During trade: mark-to-market
            for j in range(bar_idx, exit_idx):
                if direction == TradeDirection.LONG:
                    mtm_pnl = (closes[j] - fill_price) * quantity - commission
                else:
                    mtm_pnl = (fill_price - closes[j]) * quantity - commission
                equity[j] = self.initial_capital + (realized_pnl - net_pnl) + mtm_pnl

            # At exit
            equity[exit_idx] = self.initial_capital + realized_pnl

            # After exit (will be overwritten by next trade if any)
            if exit_idx < n_bars - 1:
                equity[exit_idx + 1:] = self.initial_capital + realized_pnl

            # Create trade record
            trade_id += 1
            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                entry_price=fill_price,
                entry_time=pd.Timestamp(timestamps[bar_idx]),
                exit_price=exit_fill_price,
                exit_time=pd.Timestamp(timestamps[exit_idx]),
                position_value=pos_value,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                status=status,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=commission,
                bars_held=exit_idx - bar_idx,
                _entry_idx=bar_idx,
                _exit_idx=exit_idx,
            )
            self.trades.append(trade)

            if direction == TradeDirection.LONG:
                long_trades += 1
            else:
                short_trades += 1

            last_exit_idx = exit_idx

        # Build equity DataFrame with DatetimeIndex
        equity_df = pd.DataFrame(
            {"Equity": equity},
            index=pd.DatetimeIndex(timestamps)
        )

        # Calculate drawdown
        rolling_max = np.maximum.accumulate(equity)
        drawdown_pct = np.where(
            rolling_max > 0,
            (equity - rolling_max) / rolling_max * 100,
            0.0
        )
        equity_df["Drawdown_Pct"] = drawdown_pct

        print(f"  Trades executed: {len(self.trades)} (Long: {long_trades}, Short: {short_trades})")
        print(f"  TP exits: {tp_exits}, SL exits: {sl_exits}, Other: {len(self.trades) - tp_exits - sl_exits}")

        self.cash = self.initial_capital + realized_pnl

        return {
            "trades": self.trades,
            "equity_curve": equity_df,
            "final_equity": self.cash,
            "total_trades": len(self.trades),
        }

    def _find_nearest_bar(self, target_time: pd.Timestamp, timestamps: np.ndarray) -> Optional[int]:
        """Find the nearest bar index for a given timestamp."""
        target_np = np.datetime64(target_time)
        idx = np.searchsorted(timestamps, target_np)
        if idx < len(timestamps):
            return int(idx)
        return None

    def _empty_results(self, ohlc_df: pd.DataFrame) -> Dict[str, Any]:
        """Return empty results structure with proper equity curve."""
        if len(ohlc_df) > 0:
            ohlc_df = _normalize_dates_to_naive_et(ohlc_df, "date")
            equity_df = pd.DataFrame(
                {
                    "Equity": self.initial_capital,
                    "Drawdown_Pct": 0.0,
                },
                index=pd.DatetimeIndex(ohlc_df["date"])
            )
        else:
            equity_df = pd.DataFrame(columns=["Equity", "Drawdown_Pct"])

        return {
            "trades": [],
            "equity_curve": equity_df,
            "final_equity": self.initial_capital,
            "total_trades": 0,
        }

    def _default_exit_check(self, position: dict, bar_data: dict) -> dict | None:
        """
        Default TP/SL exit check (fallback when no strategy class provided).

        Uses shared exit logic from services/exits.py.

        Args:
            position: Dict with entry_price, direction, tp_price, sl_price
            bar_data: Dict with high, low, open, close

        Returns:
            None if no exit, or dict with {price, status, reason}
        """
        return check_exit(
            direction=position.get("direction", "LONG"),
            entry_price=position.get("entry_price", 0),
            tp_price=position.get("tp_price"),
            sl_price=position.get("sl_price"),
            bar_high=bar_data["high"],
            bar_low=bar_data["low"],
            bar_open=bar_data["open"],
        )

    def reset(self):
        """Reset broker for new simulation."""
        self.cash = self.initial_capital
        self.trades.clear()


def run_simulation(
    signals: List[Dict[str, Any]],
    ohlc_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    slippage_pct: float = 0.001,
    commission_per_trade: float = 1.0,
    position_size_pct: float = 95.0,
    strategy_class: type = None,
    indicator_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run simulation.

    Args:
        signals: List of signal dicts
        ohlc_df: OHLC DataFrame
        initial_capital: Starting capital
        slippage_pct: Slippage percentage
        commission_per_trade: Commission per trade
        position_size_pct: Percentage of capital per trade

        strategy_class: Strategy class to get TP/SL from (optional)
        indicator_df: DataFrame with indicators for complex exit logic (optional)

    Returns:
        Simulation results dict
    """
    broker = SimulatedBroker(
        initial_capital=initial_capital,
        slippage_pct=slippage_pct,
        commission_per_trade=commission_per_trade,
        position_size_pct=position_size_pct,
        strategy_class=strategy_class,
    )
    return broker.run_simulation(signals, ohlc_df, indicator_df)
