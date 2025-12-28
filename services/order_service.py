# order_service.py - Order submission with guardrail validation.
"""
This module handles order submission with all safety checks:
- Account validation (whitelist)
- Quantity limits
- Slippage checks (Tiingo vs broker market price)

All orders go through submit_order() - the single entry point for trading.
"""

import asyncio
import time

import context
from models import TradeSignal
from services import guardrails


def _translate_symbol_for_tiingo(symbol: str) -> str:
    """
    Translate broker symbols to Tiingo format.

    Hyperliquid: BTC → BTCUSD
    IBKR: QQQ → QQQ (no change)
    """
    if context.active_broker == "hyperliquid":
        # Crypto symbols need USD suffix for Tiingo
        crypto_map = {
            "BTC": "BTCUSD",
            "ETH": "ETHUSD",
            "SOL": "SOLUSD",
            "AVAX": "AVAXUSD",
            "MATIC": "MATICUSD",
            "LINK": "LINKUSD",
            "UNI": "UNIUSD",
            "DOGE": "DOGEUSD",
        }
        return crypto_map.get(symbol.upper(), f"{symbol.upper()}USD")
    return symbol


def check_slippage(symbol: str, timeout: float = 5.0) -> tuple[bool, str, float | None, float | None, float | None]:
    """
    Check slippage between Tiingo trigger price and broker market price.

    Two modes:
    1. Strategy mode: Reuses trigger price from context.latest_prices (efficient)
    2. Component test mode: Fetches fresh 1min OHLC if no cached price available

    Works for both IBKR (stocks) and Hyperliquid (crypto).

    Args:
        symbol: Stock/crypto symbol (e.g., "QQQ" or "BTC")
        timeout: Max time to wait for broker price (seconds)

    Returns:
        (is_ok, error_message, slippage_pct, trigger_price, broker_price)
    """
    # === STEP 1: Get trigger price (cached or fresh) ===
    trigger_price = context.latest_prices.get(symbol)

    # If no cached price, fetch fresh OHLC (component test mode)
    if trigger_price is None:
        from services.tiingo import TiingoService
        from datetime import datetime, timedelta

        tiingo_symbol = _translate_symbol_for_tiingo(symbol)

        try:
            # Create event loop if needed (for sync context)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Fetch 1min bars from last 24 hours (to ensure we get recent data)
            # Using direct API call with precise time range (UTC)
            from services.tiingo.api import TiingoAPI

            tiingo_api = TiingoAPI()
            end_time = datetime.utcnow()  # Use UTC time
            start_time = end_time - timedelta(hours=24)  # Last 24 hours for better data availability

            # Detect if crypto or stock
            is_crypto = tiingo_symbol.upper().endswith("USD") or tiingo_symbol.upper().endswith("USDT")

            if is_crypto:
                df = loop.run_until_complete(
                    tiingo_api.fetch_crypto_intraday(
                        ticker=tiingo_symbol,
                        start_date=start_time,
                        end_date=end_time,
                        interval="1min"
                    )
                )
            else:
                df = loop.run_until_complete(
                    tiingo_api.fetch_intraday(
                        symbol=tiingo_symbol,
                        start_date=start_time,
                        end_date=end_time,
                        interval="1min"
                    )
                )

            loop.run_until_complete(tiingo_api.close())

            if df is None or df.empty or len(df) < 2:
                # No OHLC data - allow trade but warn
                print(f"   ⚠️  Could not get Tiingo OHLC for {symbol} - skipping slippage check")
                return True, "", None, None, None

            # Trigger price = last completed bar's close (iloc[-2])
            trigger_price = float(df.iloc[-2]['close'])

        except Exception as e:
            print(f"   ⚠️  Error fetching Tiingo data for {symbol}: {e} - skipping slippage check")
            return True, "", None, None, None

    # === STEP 2: Get broker market price ===
    context.price_request_queue.put(symbol)

    # Wait for response with polling
    start = time.time()
    broker_price = None

    while (time.time() - start) < timeout:
        broker_price = context.market_prices.get(symbol)
        if broker_price is not None:
            break
        time.sleep(0.1)

    if broker_price is None:
        # Could not get broker price - allow trade but warn
        broker_name = context.active_broker.upper() if context.active_broker else "BROKER"
        print(f"   ⚠️  Could not get {broker_name} market price for {symbol} - skipping slippage check")
        return True, "", None, trigger_price, None

    # === STEP 3: Calculate slippage ===
    slippage_pct = abs(broker_price - trigger_price) / trigger_price * 100

    if slippage_pct > guardrails.SLIPPAGE_TOLERANCE:
        broker_name = context.active_broker.upper() if context.active_broker else "BROKER"
        error_msg = (
            f"🚫 SLIPPAGE BLOCKED: Price difference too high for {symbol}!\n"
            f"   Tiingo trigger: ${trigger_price:.2f}\n"
            f"   {broker_name} market:    ${broker_price:.2f}\n"
            f"   Slippage:       {slippage_pct:.2f}% (max allowed: {guardrails.SLIPPAGE_TOLERANCE}%)"
        )
        return False, error_msg, slippage_pct, trigger_price, broker_price

    return True, "", slippage_pct, trigger_price, broker_price


def submit_order(
    symbol: str,
    action: str,
    quantity: float,
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

    # Check slippage (Tiingo trigger price vs broker market price)
    slippage_ok, slippage_error, slippage_pct, trigger_price, broker_price = check_slippage(symbol)

    if not slippage_ok:
        # Log error to both terminal and Telegram
        print(f"\n{'='*60}")
        print(slippage_error)
        print(f"Attempted: {action} {quantity} {symbol}")
        print(f"{'='*60}\n")
        context.log(slippage_error, "error")
        return False

    # Log slippage info if available
    if slippage_pct is not None and trigger_price is not None and broker_price is not None:
        broker_name = context.active_broker.upper() if context.active_broker else "BROKER"
        print(f"   ✓ Slippage guardrail passed: {slippage_pct:.2f}% (max: {guardrails.SLIPPAGE_TOLERANCE}%)")
        print(f"     Tiingo: ${trigger_price:.2f}, {broker_name}: ${broker_price:.2f}")

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

