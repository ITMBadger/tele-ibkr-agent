"""
Component Testing Module

Manual testing tool for validating trading components via Telegram.
Uses strategy ID 888 and fetches 1min OHLC data for guardrail slippage checks.

Behavior (same as other strategies):
- BUY triggers at xx:00s (beginning of new minute) like real strategies
- Fetches 1min OHLC data from market data provider (~2s delay)
- Uses global shift: trigger price = completed bar's close (iloc[-2])
- Activates all guardrails (slippage, position limits, etc.)
- Logs OHLC trigger to CSV in data/logs/
- Creates position.json tracking file

Supports both IBKR (stocks) and Hyperliquid (crypto) based on active broker.
Enable/disable this feature in services/telegram.py with ENABLE_TESTING_BUTTONS flag.
"""

import time

import context
from services import order_service, pos_manager
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


def translate_symbol_for_tiingo(symbol: str) -> str:
    """
    Translate crypto symbols to Tiingo format.

    Hyperliquid uses: BTC, ETH, SOL, etc.
    Tiingo needs: BTCUSD, ETHUSD, SOLUSD, etc.

    For stocks (IBKR), return as-is.
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
    else:
        # Stock symbols stay as-is
        return symbol


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
                {"text": "📊 Test Slippage", "callback_data": "test_slippage"}
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
        "⚠️ This will place a REAL order!\n"
        "• Fetches 1min OHLC for slippage check\n"
        "• Activates all guardrails\n"
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

    elif callback_data == "test_slippage":
        return await _test_slippage()

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


async def _test_slippage() -> str:
    """
    Test slippage check without placing an order.

    Fetches 1min OHLC from Tiingo, gets current market price from broker,
    and compares them like the guardrail does.

    Note: Only applicable for IBKR (stocks). Hyperliquid uses its own data.

    Prints result to both terminal and returns message for Telegram.
    """
    from services.guardrails import SLIPPAGE_TOLERANCE

    broker = get_broker_name()
    symbol = get_test_symbol()

    if not context.broker_connected.is_set():
        msg = f"❌ {broker} not connected. Cannot test slippage."
        print(f"\n{msg}")
        return msg

    # === STEP 1: Fetch 1min OHLC from Tiingo (strategy data source) ===
    print(f"\n{'='*60}")
    print(f"📊 SLIPPAGE TEST for {symbol}")
    print(f"{'='*60}")

    # Always use Tiingo for trigger price (this is what strategies use)
    from services.tiingo import TiingoService
    tiingo = TiingoService()

    # Translate symbol to Tiingo format (e.g., BTC → BTCUSD for crypto)
    tiingo_symbol = translate_symbol_for_tiingo(symbol)

    try:
        if tiingo_symbol != symbol:
            print(f"   Fetching 1min OHLC from Tiingo (last 24 hours)...")
            print(f"   Translating: {symbol} → {tiingo_symbol}")
        else:
            print(f"   Fetching 1min OHLC from Tiingo (last 24 hours)...")

        # Fetch just last 24 hours using direct API for precise time range
        from services.tiingo.api import TiingoAPI
        from datetime import datetime, timedelta

        tiingo_api = TiingoAPI()
        end_time = datetime.utcnow()  # Use UTC time
        start_time = end_time - timedelta(hours=24)  # Last 24 hours for better data availability

        # Detect if crypto or stock
        is_crypto = tiingo_symbol.upper().endswith("USD") or tiingo_symbol.upper().endswith("USDT")

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

        await tiingo_api.close()

        # Convert DataFrame to list of dicts (matching get_intraday_ohlc format)
        if df is None or df.empty:
            ohlc_data = []
        else:
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

        if not ohlc_data or len(ohlc_data) < 2:
            msg = (
                f"❌ Slippage Test Failed\n"
                f"   Insufficient OHLC data from Tiingo\n"
                f"   Bars received: {len(ohlc_data) if ohlc_data else 0}"
            )
            print(f"   {msg}")
            print(f"{'='*60}\n")
            return msg

        # Trigger price = iloc[-2] (last completed bar's close)
        trigger_bar = ohlc_data[-2]
        trigger_price = trigger_bar['close']
        trigger_time = format_iso_to_et(trigger_bar['date'])

        print(f"   ✓ Tiingo trigger price: ${trigger_price:.4f}")
        print(f"     From bar: {trigger_time} (completed)")
        print(f"     Total bars: {len(ohlc_data)}")

    except Exception as e:
        msg = f"❌ Failed to fetch Tiingo data: {e}"
        print(f"   {msg}")
        print(f"{'='*60}\n")
        return msg

    # === STEP 2: Get current market price from broker ===
    print(f"\n   Fetching current price from {broker}...")

    # Request price via queue
    context.price_request_queue.put(symbol)

    # Wait for response with polling
    import time as time_module
    start = time_module.time()
    timeout = 5.0
    market_price = None

    import asyncio
    while (time_module.time() - start) < timeout:
        market_price = context.market_prices.get(symbol)
        if market_price is not None:
            break
        await asyncio.sleep(0.1)

    if market_price is None:
        msg = (
            f"⚠️ Slippage Test - No Market Price\n"
            f"   Tiingo trigger: ${trigger_price:.4f} @ {trigger_time}\n"
            f"   {broker} price: Could not fetch (timeout)\n\n"
            f"   Slippage check would be SKIPPED"
        )
        print(f"   ⚠️ Could not get {broker} price (timeout)")
        print(f"{'='*60}\n")
        return msg

    print(f"   ✓ {broker} market price: ${market_price:.4f}")

    # === STEP 3: Calculate slippage ===
    slippage_pct = abs(market_price - trigger_price) / trigger_price * 100
    price_diff = market_price - trigger_price

    print(f"\n   📈 SLIPPAGE ANALYSIS:")
    print(f"      Tiingo (trigger):  ${trigger_price:.4f}")
    print(f"      {broker} (market): ${market_price:.4f}")
    print(f"      Difference:        ${price_diff:+.4f}")
    print(f"      Slippage:          {slippage_pct:.3f}%")
    print(f"      Tolerance:         {SLIPPAGE_TOLERANCE}%")

    # === STEP 4: Determine result ===
    if slippage_pct <= SLIPPAGE_TOLERANCE:
        status = "✅ PASS"
        result_emoji = "✅"
        would_block = "NO - Order would proceed"
        print(f"\n   {status} - Slippage within tolerance")
    else:
        status = "❌ FAIL"
        result_emoji = "🚫"
        would_block = "YES - Order would be BLOCKED"
        print(f"\n   {status} - Slippage exceeds tolerance!")

    print(f"   Would block order: {would_block}")
    print(f"{'='*60}\n")

    # Build Telegram message
    msg = (
        f"{result_emoji} **Slippage Test Result**\n\n"
        f"**Symbol:** {symbol}\n"
        f"**Tiingo (trigger):** ${trigger_price:.4f}\n"
        f"  └─ Bar time: {trigger_time}\n"
        f"**{broker} (market):** ${market_price:.4f}\n\n"
        f"**Slippage:** {slippage_pct:.3f}%\n"
        f"**Tolerance:** {SLIPPAGE_TOLERANCE}%\n\n"
        f"**Result:** {status}\n"
        f"**Would block order:** {would_block}"
    )

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

    # Create appropriate market data provider based on broker
    if context.active_broker == "hyperliquid":
        from services.market_data import HyperliquidDataProvider
        data_provider = HyperliquidDataProvider()
    else:
        from services.tiingo import TiingoService
        data_provider = TiingoService()

    try:
        ohlc_data = await data_provider.get_intraday_ohlc(
            symbol=symbol,
            days=1,
            interval="1min",
            use_cache=False  # Always fresh
        )

        if not ohlc_data or len(ohlc_data) < 2:
            await data_provider.close()
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
        data_info = f"\n📊 Fetched {bar_count} 1min bars"
        data_info += f"\n   Trigger: ${trigger_price:.4f} @ {trigger_time} (completed bar)"

        # Store for slippage guardrail check
        context.latest_prices.set(symbol, trigger_price)

    except Exception as e:
        await data_provider.close()
        return (
            f"🚫 Scheduled BUY BLOCKED\n"
            f"   Failed to fetch OHLC data: {e}\n"
            f"   Cannot verify slippage guardrail"
        )
    finally:
        await data_provider.close()

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
