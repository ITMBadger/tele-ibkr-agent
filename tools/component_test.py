"""
Component Testing Module

Manual testing tool for validating trading components via Telegram.
Uses strategy ID 888 and fetches 1min OHLC data for guardrail checks.

Features:
- TEST BUY: Places real order at minute boundary (like real strategies)
- TEST GUARDRAILS: Tests all 3 guardrails without placing order

Behavior (same as other strategies):
- BUY triggers at xx:00s (beginning of new minute) like real strategies
- Fetches last 1 hour of 1min OHLC data from market data provider
- Uses global shift: trigger price = completed bar's close (iloc[-2])
- Activates all guardrails (account, quantity, slippage)
- Logs OHLC trigger to CSV in data/logs/
- Creates position.json tracking file

Supports both IBKR (stocks) and Hyperliquid (crypto) based on active broker.
Enable/disable this feature in services/telegram.py with ENABLE_TESTING_BUTTONS flag.
"""

import time

import context
from services import order_service, pos_manager
from services.market_data import CRYPTO_SYMBOLS
from services.time_utils import format_iso_to_et, get_et_timestamp
from services.logger import SignalLogger


# === TESTING CONFIGURATION ===

STRATEGY_ID = "888"
INTERVAL = 60  # Trigger at every minute boundary (xx:00s)

# Broker-specific defaults
BROKER_CONFIG = {
    "ibkr": {
        "symbol": "QQQ",
        "quantity": 1,
        "unit_name": "share",
        "broker_name": "IBKR",
    },
    "hyperliquid": {
        "symbol": "BTC",
        "quantity": 0.001,
        "unit_name": "contract",
        "broker_name": "Hyperliquid",
    },
}


def get_config() -> dict:
    """Get broker-specific test configuration."""
    broker = context.active_broker or "ibkr"
    return BROKER_CONFIG.get(broker, BROKER_CONFIG["ibkr"])


def get_test_symbol() -> str:
    """Get test symbol for current broker."""
    return get_config()["symbol"]


def get_test_quantity() -> float:
    """Get test quantity for current broker."""
    return get_config()["quantity"]


def get_unit_name() -> str:
    """Get unit name (share/coin) for current broker."""
    return get_config()["unit_name"]


def get_broker_name() -> str:
    """Get broker display name."""
    return get_config()["broker_name"]


# === TIINGO DATA HELPERS ===
# Shared functions for fetching OHLC data from Tiingo (used by both modes)
# CRYPTO_SYMBOLS is imported from services.market_data


def translate_symbol_for_tiingo(symbol: str) -> str:
    """
    Translate symbol to Tiingo format.

    For Hyperliquid crypto: BTC → BTCUSD
    For IBKR stocks: QQQ → QQQ (unchanged)
    """
    if context.active_broker != "hyperliquid":
        return symbol

    symbol_upper = symbol.upper()
    # Already has USD suffix
    if symbol_upper.endswith("USD") or symbol_upper.endswith("USDT"):
        return symbol

    # Known crypto - add USD suffix
    if symbol_upper in CRYPTO_SYMBOLS:
        return f"{symbol_upper}USD"

    return symbol


async def fetch_ohlc_1hour(symbol: str) -> list[dict]:
    """
    Fetch last 1 hour of 1min OHLC data from Tiingo.

    Unified function for both IBKR and Hyperliquid modes.
    Handles symbol translation and crypto/stock detection automatically.

    Returns:
        List of OHLC bar dicts, or empty list on error
    """
    from datetime import datetime, timedelta
    from services.tiingo.api import TiingoAPI

    tiingo_symbol = translate_symbol_for_tiingo(symbol)
    is_crypto = tiingo_symbol.upper().endswith("USD") or tiingo_symbol.upper().endswith("USDT")

    tiingo_api = TiingoAPI()
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        if is_crypto:
            df = await tiingo_api.fetch_crypto_intraday(
                ticker=tiingo_symbol,
                start_date=start_time,
                end_date=end_time,
                interval="1min"
            )
        else:
            df = await tiingo_api.fetch_intraday(
                symbol=tiingo_symbol,
                start_date=start_time,
                end_date=end_time,
                interval="1min"
            )

        if df is None or df.empty:
            return []

        # Convert DataFrame to list of dicts
        ohlc_data = []
        for _, row in df.iterrows():
            ohlc_data.append({
                'date': str(row['date']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row.get('volume', 0))
            })
        return ohlc_data
    finally:
        await tiingo_api.close()


# === PENDING BUY STATE ===
# When user clicks buy, we schedule it for next minute boundary

_pending_buy: dict | None = None  # {"scheduled_at": time, "chat_id": int}
_last_check_interval: int = 0  # Track last executed interval


# === INLINE KEYBOARD DEFINITION ===

def get_testing_keyboard() -> dict:
    """
    Returns Telegram InlineKeyboard with test buttons.

    Returns:
        dict: Telegram reply_markup for inline keyboard
    """
    symbol = get_test_symbol()
    return {
        "inline_keyboard": [
            [
                {"text": f"📈 TEST BUY {symbol}", "callback_data": "test_buy"}
            ],
            [
                {"text": "📊 Test Guardrails", "callback_data": "test_guardrails"}
            ],
            [
                {"text": "🚫 Cancel Pending", "callback_data": "test_cancel"},
                {"text": "❌ Close", "callback_data": "test_close"}
            ]
        ]
    }


# === BUTTON HANDLERS ===

def handle_test_command() -> tuple[str, dict | None]:
    """
    Handle /test command - show component test keyboard.

    Returns:
        tuple: (message_text, reply_markup or None)
    """
    if not context.broker_connected.is_set():
        broker = get_broker_name()
        return (f"❌ {broker} not connected. Cannot test trading.", None)

    symbol = get_test_symbol()
    quantity = get_test_quantity()
    unit = get_unit_name()
    broker = get_broker_name()

    message = (
        f"🧪 **Component Test Mode** ({broker})\n\n"
        f"Strategy ID: {STRATEGY_ID}\n"
        f"Symbol: {symbol}\n"
        f"Quantity: {quantity} {unit}\n\n"
        f"Current Account: {context.current_account or 'Not set'}\n\n"
        "⚠️ TEST BUY will place a REAL order!\n"
        "• Fetches last 1hr of 1min OHLC\n"
        "• Tests all guardrails (account, qty, slippage)\n"
        "• Logs via standard buy method\n\n"
        "Select action:"
    )

    return (message, get_testing_keyboard())


async def handle_callback(callback_data: str) -> str:
    """
    Handle inline keyboard button callbacks (async).

    Args:
        callback_data: The callback_data from button press

    Returns:
        str: Response message to show user
    """
    if callback_data == "test_buy":
        return await _schedule_test_buy()

    elif callback_data == "test_guardrails":
        return await _test_guardrails()

    elif callback_data == "test_cancel":
        return cancel_pending_buy()

    elif callback_data == "test_close":
        return "Component test menu closed."

    else:
        return f"Unknown test action: {callback_data}"


def has_position() -> bool:
    """
    Check if we already have a component test position.
    Same pattern as BaseStrategy.has_position().
    """
    symbol = get_test_symbol()

    # Check local pos_manager
    if pos_manager.get_position(symbol) is not None:
        return True

    # Check broker actual positions
    account_positions = context.get_positions_for_account()
    position = account_positions.get(symbol)
    return position is not None and position.get("qty", 0) > 0


async def _schedule_test_buy() -> str:
    """
    Schedule a component test BUY for next minute boundary.

    Does NOT execute immediately - schedules for xx:00s like real strategies.
    Returns immediately with scheduled time info.
    """
    global _pending_buy

    broker = get_broker_name()
    symbol = get_test_symbol()
    quantity = get_test_quantity()
    unit = get_unit_name()

    if not context.broker_connected.is_set():
        return f"❌ {broker} not connected."

    if has_position():
        return f"🚫 Already have position in {symbol}. Close first before buying again."

    if _pending_buy is not None:
        return "⏳ Buy already scheduled. Wait for execution or cancel."

    # Calculate next minute boundary
    now = time.time()
    next_minute = ((int(now) // INTERVAL) + 1) * INTERVAL
    seconds_until = next_minute - now

    # Schedule the buy
    _pending_buy = {
        "scheduled_at": now,
        "target_time": next_minute,
    }

    return (
        f"⏰ BUY scheduled for next minute boundary\n"
        f"   Symbol: {symbol}\n"
        f"   Quantity: {quantity} {unit}\n"
        f"   Executes in: ~{int(seconds_until)}s\n\n"
        f"Will fetch market data, apply guardrails, and track position."
    )


async def _test_guardrails() -> str:
    """
    Test all 3 guardrails without placing an order.

    Tests:
    1. Account guardrail - Is current account in allowed list?
    2. Quantity guardrail - Is test quantity within limit?
    3. Slippage guardrail - Is price difference within tolerance?

    Mimics strategy flow: fetches 1hr of 1min OHLC, uses [-2] as trigger price,
    compares with live market price.

    Prints result to both terminal and returns message for Telegram.
    """
    from services.guardrails import (
        SLIPPAGE_TOLERANCE,
        get_allowed_accounts,
        get_max_order_quantity,
        validate_account,
        validate_quantity,
    )

    broker = get_broker_name()
    symbol = get_test_symbol()
    quantity = get_test_quantity()
    allowed_accounts = get_allowed_accounts()
    max_qty = get_max_order_quantity()

    if not context.broker_connected.is_set():
        msg = f"❌ {broker} not connected. Cannot test guardrails."
        print(f"\n{msg}")
        return msg

    print(f"\n{'='*60}")
    print(f"📊 GUARDRAILS TEST for {symbol}")
    print(f"{'='*60}")

    results = []  # Track pass/fail for each guardrail
    telegram_lines = [f"📊 **Guardrails Test Result**\n", f"**Symbol:** {symbol}\n"]

    # === GUARDRAIL 1: Account ===
    print(f"\n   [1/3] ACCOUNT GUARDRAIL")
    account_ok, account_error = validate_account(context.current_account)
    if account_ok:
        print(f"   ✅ PASS - Account '{context.current_account}' is allowed")
        print(f"      Allowed: {', '.join(allowed_accounts)}")
        results.append(True)
        telegram_lines.append(f"**1. Account:** ✅ PASS\n   `{context.current_account}` in allowed list")
    else:
        print(f"   ❌ FAIL - {account_error}")
        results.append(False)
        telegram_lines.append(f"**1. Account:** ❌ FAIL\n   `{context.current_account}` not allowed")

    # === GUARDRAIL 2: Quantity ===
    print(f"\n   [2/3] QUANTITY GUARDRAIL")
    quantity_ok, quantity_error = validate_quantity(quantity)
    if quantity_ok:
        print(f"   ✅ PASS - Quantity {quantity} <= max {max_qty}")
        results.append(True)
        telegram_lines.append(f"\n**2. Quantity:** ✅ PASS\n   {quantity} <= {max_qty}")
    else:
        print(f"   ❌ FAIL - {quantity_error}")
        results.append(False)
        telegram_lines.append(f"\n**2. Quantity:** ❌ FAIL\n   {quantity} > {max_qty}")

    # === GUARDRAIL 3: Slippage ===
    print(f"\n   [3/3] SLIPPAGE GUARDRAIL")

    # Log symbol translation for debugging
    tiingo_symbol = translate_symbol_for_tiingo(symbol)
    if tiingo_symbol != symbol:
        print(f"   Translating symbol: {symbol} → {tiingo_symbol}")

    print(f"   Fetching 1min OHLC from Tiingo (last 1 hour)...")

    try:
        ohlc_data = await fetch_ohlc_1hour(symbol)

        if not ohlc_data or len(ohlc_data) < 2:
            print(f"   ❌ FAIL - Insufficient OHLC data from Tiingo")
            print(f"      Bars received: {len(ohlc_data) if ohlc_data else 0}")
            results.append(False)
            telegram_lines.append(f"\n**3. Slippage:** ❌ FAIL\n   No OHLC data from Tiingo")
        else:
            # Trigger price = iloc[-2] (last completed bar's close)
            trigger_bar = ohlc_data[-2]
            trigger_price = trigger_bar['close']
            trigger_time = format_iso_to_et(trigger_bar['date'])

            print(f"   ✓ Trigger price: ${trigger_price:.4f}")
            print(f"     From bar: {trigger_time} (completed)")
            print(f"     Total bars: {len(ohlc_data)}")

            # Get current market price from broker
            print(f"   Fetching current price from {broker}...")
            context.price_request_queue.put(symbol)

            # Wait for response with polling
            import time as time_module
            import asyncio
            start = time_module.time()
            timeout = 5.0
            market_price = None

            while (time_module.time() - start) < timeout:
                market_price = context.market_prices.get(symbol)
                if market_price is not None:
                    break
                await asyncio.sleep(0.1)

            if market_price is None:
                print(f"   ⚠️ Could not get {broker} price (timeout)")
                print(f"   Slippage check SKIPPED")
                results.append(True)  # Don't fail if we can't get market price
                telegram_lines.append(f"\n**3. Slippage:** ⚠️ SKIPPED\n   Could not fetch {broker} price")
            else:
                print(f"   ✓ {broker} market price: ${market_price:.4f}")

                # Calculate slippage
                slippage_pct = abs(market_price - trigger_price) / trigger_price * 100
                price_diff = market_price - trigger_price

                print(f"   Analysis:")
                print(f"      Tiingo (trigger): ${trigger_price:.4f}")
                print(f"      {broker} (market): ${market_price:.4f}")
                print(f"      Difference:        ${price_diff:+.4f}")
                print(f"      Slippage:          {slippage_pct:.3f}%")
                print(f"      Tolerance:         {SLIPPAGE_TOLERANCE}%")

                if slippage_pct <= SLIPPAGE_TOLERANCE:
                    print(f"   ✅ PASS - Slippage within tolerance")
                    results.append(True)
                    telegram_lines.append(
                        f"\n**3. Slippage:** ✅ PASS\n"
                        f"   Tiingo: ${trigger_price:.4f}\n"
                        f"   {broker}: ${market_price:.4f}\n"
                        f"   Slippage: {slippage_pct:.3f}% <= {SLIPPAGE_TOLERANCE}%"
                    )
                else:
                    print(f"   ❌ FAIL - Slippage exceeds tolerance!")
                    results.append(False)
                    telegram_lines.append(
                        f"\n**3. Slippage:** ❌ FAIL\n"
                        f"   Tiingo: ${trigger_price:.4f}\n"
                        f"   {broker}: ${market_price:.4f}\n"
                        f"   Slippage: {slippage_pct:.3f}% > {SLIPPAGE_TOLERANCE}%"
                    )

    except Exception as e:
        print(f"   ❌ FAIL - Error fetching data: {e}")
        results.append(False)
        telegram_lines.append(f"\n**3. Slippage:** ❌ FAIL\n   Error: {e}")

    # === SUMMARY ===
    passed = sum(results)
    total = len(results)
    all_passed = all(results)

    print(f"\n{'='*60}")
    if all_passed:
        print(f"✅ ALL GUARDRAILS PASSED ({passed}/{total})")
        print(f"   Order would PROCEED")
        summary_emoji = "✅"
        summary_text = f"ALL PASSED ({passed}/{total}) - Order would proceed"
    else:
        print(f"❌ GUARDRAILS FAILED ({passed}/{total} passed)")
        print(f"   Order would be BLOCKED")
        summary_emoji = "🚫"
        summary_text = f"FAILED ({passed}/{total} passed) - Order would be blocked"
    print(f"{'='*60}\n")

    # Build Telegram message
    telegram_lines.append(f"\n\n{summary_emoji} **{summary_text}**")
    msg = "\n".join(telegram_lines)

    return msg


def has_pending_buy() -> bool:
    """Check if there's a pending buy scheduled."""
    return _pending_buy is not None


def cancel_pending_buy() -> str:
    """Cancel any pending scheduled buy."""
    global _pending_buy

    if _pending_buy is None:
        return "No pending buy to cancel."

    _pending_buy = None
    return "✅ Pending buy cancelled."


def should_execute_pending() -> bool:
    """
    Check if it's time to execute the pending buy.

    Uses clock-aligned interval logic (same as BaseStrategy.should_run).
    Returns True once per minute boundary.
    """
    global _last_check_interval

    if _pending_buy is None:
        return False

    now = time.time()
    current_interval = int(now // INTERVAL)

    if current_interval > _last_check_interval:
        _last_check_interval = current_interval
        return True

    return False


async def execute_pending_buy() -> str | None:
    """
    Execute the pending buy at minute boundary.

    Same behavior as other strategies:
    - Fetches 1min OHLC from market data provider (~2s delay)
    - Uses global shift: trigger price = iloc[-2] (completed bar)
    - Activates all guardrails via order_service
    - Logs to CSV via SignalLogger
    - Tracks position via pos_manager

    Returns:
        str: Result message, or None if no pending buy
    """
    global _pending_buy

    if _pending_buy is None:
        return None

    # Clear pending state (only execute once)
    _pending_buy = None

    # Get broker-specific config
    broker = get_broker_name()
    symbol = get_test_symbol()
    quantity = get_test_quantity()

    if not context.broker_connected.is_set():
        return f"❌ Scheduled BUY failed: {broker} not connected."

    # Safety check: position may have been opened between schedule and execution
    if has_position():
        return f"🚫 Scheduled BUY cancelled: already have position in {symbol}."

    # Fetch last 1 hour of 1min OHLC data from Tiingo (unified for both IBKR and Hyperliquid)
    ohlc_data = await fetch_ohlc_1hour(symbol)

    if not ohlc_data or len(ohlc_data) < 2:
        return (
            f"🚫 Scheduled BUY BLOCKED\n"
            f"   Insufficient OHLC data (need >= 2 bars)\n"
            f"   Bars received: {len(ohlc_data) if ohlc_data else 0}"
        )

    # === GLOBAL SHIFT LOGIC (same as other strategies) ===
    # Signal bar = iloc[-1] (current incomplete bar)
    # Trigger price = iloc[-2] (last completed bar's close)
    trigger_bar = ohlc_data[-2]
    trigger_price = trigger_bar['close']
    trigger_time = format_iso_to_et(trigger_bar['date'])

    bar_count = len(ohlc_data)
    data_info = f"\n📊 Fetched {bar_count} 1min bars (last 1 hour)"
    data_info += f"\n   Trigger: ${trigger_price:.4f} @ {trigger_time} (completed bar)"

    # Store for slippage guardrail check
    context.latest_prices.set(symbol, trigger_price)

    # Submit order through guardrails (same as all strategies)
    success = order_service.submit_order(
        symbol=symbol,
        action="BUY",
        quantity=quantity,
        strategy_id=STRATEGY_ID
    )

    # Log OHLC data to CSV (same as other strategies)
    logger = SignalLogger.get_or_create(
        symbol=symbol,
        strategy_name="Component Test"
    )
    logger.log_event(
        ohlc_bars=ohlc_data,
        indicator_columns=None,  # No indicators for component test
        signal="BUY",
        triggered=success,
        event_type="signal"
    )

    # Track position via pos_manager (same as other strategies)
    if success:
        pos_manager.save_position(
            symbol=symbol,
            account=context.current_account or "",
            strategy_id=STRATEGY_ID,
            action="LONG",
            quantity=quantity,
            entry_price=trigger_price,
            take_profit=None,  # Component test doesn't use TP/SL
            stop_loss=None
        )

        msg = (
            f"✅ Scheduled BUY executed @ {get_et_timestamp()}\n"
            f"   Strategy ID: {STRATEGY_ID}\n"
            f"   Symbol: {symbol}\n"
            f"   Quantity: {quantity}\n"
            f"   Entry: ${trigger_price:.4f}"
            f"{data_info}"
        )
        context.log(msg, "trade")
    else:
        msg = (
            f"🚫 Scheduled BUY blocked by guardrails\n"
            f"   Check terminal for details"
            f"{data_info}"
        )
        context.log(msg, "trade")


# === UTILITY FUNCTIONS ===

def is_test_command(text: str) -> bool:
    """Check if message is the /test command."""
    return text.strip().lower() == "/test"

def is_test_callback(callback_data: str) -> bool:
    """Check if callback is from testing buttons."""
    return callback_data.startswith("test_")
