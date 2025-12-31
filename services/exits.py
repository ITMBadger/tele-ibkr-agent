# services/exits.py
"""
Centralized exit logic for TP/SL - Single source of truth.

Used by both live trading (BaseStrategy) and backtesting (SimulatedBroker).
This ensures exit behavior is identical across both modes.

To add a new exit type:
1. Add a new function here (e.g., check_trailing_stop)
2. Strategy calls it from exit_check() override
3. No base class changes needed
"""


def check_exit(
    direction: str,
    entry_price: float,
    tp_price: float | None,
    sl_price: float | None,
    bar_high: float,
    bar_low: float,
    bar_open: float,
) -> dict | None:
    """
    Check if position should exit based on TP/SL levels.

    This is the single source of truth for TP/SL exit logic.
    Both live trading and backtesting use this function.

    """
    if direction == "LONG":
        # Check SL first (risk management priority)
        if sl_price and bar_low <= sl_price:
            # Gap down: exit at open if it's below SL
            exit_price = min(bar_open, sl_price) if bar_open < sl_price else sl_price
            return {"price": exit_price, "status": "CLOSED_SL", "reason": "Stop Loss"}
        # Check TP
        if tp_price and bar_high >= tp_price:
            # Gap up: exit at open if it's above TP
            exit_price = max(bar_open, tp_price) if bar_open > tp_price else tp_price
            return {"price": exit_price, "status": "CLOSED_TP", "reason": "Take Profit"}

    elif direction == "SHORT":
        # Check SL first (risk management priority)
        if sl_price and bar_high >= sl_price:
            # Gap up: exit at open if it's above SL
            exit_price = max(bar_open, sl_price) if bar_open > sl_price else sl_price
            return {"price": exit_price, "status": "CLOSED_SL", "reason": "Stop Loss"}
        # Check TP
        if tp_price and bar_low <= tp_price:
            # Gap down: exit at open if it's below TP
            exit_price = min(bar_open, tp_price) if bar_open < tp_price else tp_price
            return {"price": exit_price, "status": "CLOSED_TP", "reason": "Take Profit"}

    return None


def calculate_tp_sl_prices(
    entry_price: float,
    direction: str,
    tp_pct: float,
    sl_pct: float,
) -> tuple[float, float]:
    """
    Calculate TP and SL prices from percentages.

    """
    if direction == "LONG":
        tp_price = entry_price * (1 + tp_pct / 100)
        sl_price = entry_price * (1 - sl_pct / 100)
    else:  # SHORT
        tp_price = entry_price * (1 - tp_pct / 100)
        sl_price = entry_price * (1 + sl_pct / 100)

    return round(tp_price, 2), round(sl_price, 2)
