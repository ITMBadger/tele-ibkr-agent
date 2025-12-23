# order_service.py - Order submission with guardrail validation.
"""
This module handles order submission with all safety checks:
- Account validation (whitelist)
- Quantity limits
- Slippage checks (Tiingo vs IBKR price)

All orders go through submit_order() - the single entry point for trading.
"""

import time

import context
from models import TradeSignal
from services import guardrails


def check_slippage(symbol: str, timeout: float = 5.0) -> tuple[bool, str, float | None, float | None, float | None]:
    """
    Check slippage between Tiingo trigger price and IBKR market price.

    Uses reqMktData snapshot approach: request streaming, capture first tick, cancel.

    Args:
        symbol: Stock symbol
        timeout: Max time to wait for IBKR price (seconds)

    Returns:
        (is_ok, error_message, slippage_pct, trigger_price, ibkr_price)
    """
    # Get Tiingo trigger price from latest_prices
    trigger_price = context.latest_prices.get(symbol)
    if trigger_price is None:
        # No trigger price available - allow trade but warn
        return True, "", None, None, None

    # Request IBKR market price
    context.price_request_queue.put(symbol)

    # Wait for response with polling
    start = time.time()
    ibkr_price = None

    while (time.time() - start) < timeout:
        ibkr_price = context.market_prices.get(symbol)
        if ibkr_price is not None:
            break
        time.sleep(0.1)

    if ibkr_price is None:
        # Could not get IBKR price - allow trade but warn
        print(f"   ⚠️  Could not get IBKR market price for {symbol} - skipping slippage check")
        return True, "", None, trigger_price, None

    # Calculate slippage percentage
    slippage_pct = abs(ibkr_price - trigger_price) / trigger_price * 100

    if slippage_pct > guardrails.SLIPPAGE_TOLERANCE:
        error_msg = (
            f"🚫 SLIPPAGE BLOCKED: Price difference too high for {symbol}!\n"
            f"   Tiingo trigger: ${trigger_price:.2f}\n"
            f"   IBKR market:    ${ibkr_price:.2f}\n"
            f"   Slippage:       {slippage_pct:.2f}% (max allowed: {guardrails.SLIPPAGE_TOLERANCE}%)"
        )
        return False, error_msg, slippage_pct, trigger_price, ibkr_price

    # Clear market price after check (force fresh fetch next time)
    # context.market_prices.delete(symbol)

    return True, "", slippage_pct, trigger_price, ibkr_price


def submit_order(
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "MKT",
    limit_price: float | None = None,
    strategy_id: str = "manual"
) -> bool:
    """
    Submit a trade signal to the order queue.

    This is the SINGLE ENTRY POINT for all orders. All guardrails are enforced here.

    Args:
        symbol: Stock symbol (e.g., "QQQ")
        action: "BUY" or "SELL"
        quantity: Number of shares
        order_type: "MKT" or "LMT"
        limit_price: Required if order_type is "LMT"
        strategy_id: Strategy ID for position tracking (default "manual")

    Returns:
        True if order was submitted, False if blocked by guardrails.
    """
    # Validate against guardrails (account, quantity)
    is_valid, error_msg = guardrails.validate_order_guardrails(
        context.current_account, quantity
    )

    if not is_valid:
        # Log error to both terminal and Telegram
        print(f"\n{'='*60}")
        print(error_msg)
        print(f"Attempted: {action} {quantity} {symbol}")
        print(f"{'='*60}\n")
        context.log(error_msg, "error")
        return False

    # Log account and quantity guardrail pass
    print(f"   ✓ Account guardrail passed: {context.current_account} (allowed: {', '.join(guardrails.ALLOWED_ACCOUNTS)})")
    print(f"   ✓ Quantity guardrail passed: {quantity} shares (max: {guardrails.MAX_ORDER_QUANTITY})")

    # Check slippage (Tiingo trigger price vs IBKR market price)
    slippage_ok, slippage_error, slippage_pct, trigger_price, ibkr_price = check_slippage(symbol)

    if not slippage_ok:
        # Log error to both terminal and Telegram
        print(f"\n{'='*60}")
        print(slippage_error)
        print(f"Attempted: {action} {quantity} {symbol}")
        print(f"{'='*60}\n")
        context.log(slippage_error, "error")
        return False

    # Log slippage info if available
    if slippage_pct is not None and trigger_price is not None and ibkr_price is not None:
        print(f"   ✓ Slippage guardrail passed: {slippage_pct:.2f}% (max: {guardrails.SLIPPAGE_TOLERANCE}%)")
        print(f"     Tiingo: ${trigger_price:.2f}, IBKR: ${ibkr_price:.2f}")

    signal = TradeSignal(
        symbol=symbol,
        action=action,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        strategy_id=strategy_id
    )
    context.order_queue.put(signal)
    return True

