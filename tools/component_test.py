"""
Component Testing Module

Manual testing tool for validating trading components via Telegram.
Uses strategy ID 999 and fetches 1min OHLC data for guardrail slippage checks.

Behavior (same as other strategies):
- BUY triggers at xx:00s (beginning of new minute) like real strategies
- Fetches 1min OHLC data from Tiingo (~2s delay)
- Uses global shift: trigger price = completed bar's close (iloc[-2])
- Activates all guardrails (slippage, position limits, etc.)
- Logs OHLC trigger to CSV in data/logs/
- Creates position.json tracking file

Enable/disable this feature in services/telegram.py with ENABLE_TESTING_BUTTONS flag.
"""

import time

import context
from services import order_service, pos_manager
from services.tiingo import TiingoService
from services.time_utils import format_iso_to_et, get_et_timestamp
from services.logger import SignalLogger


# === TESTING CONFIGURATION ===

STRATEGY_ID = "888"
TEST_SYMBOL = "QQQ"
TEST_QUANTITY = 1
INTERVAL = 60  # Trigger at every minute boundary (xx:00s)


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
    return {
        "inline_keyboard": [
            [
                {"text": "📈 TEST BUY QQQ", "callback_data": "test_buy_qqq"}
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
    if not context.ibkr_connected.is_set():
        return ("❌ IBKR not connected. Cannot test trading.", None)

    message = (
        "🧪 **Component Test Mode**\n\n"
        f"Strategy ID: {STRATEGY_ID}\n"
        f"Symbol: {TEST_SYMBOL}\n"
        f"Quantity: {TEST_QUANTITY} share\n\n"
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
    if callback_data == "test_buy_qqq":
        return await _execute_test_buy()

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
    # Check local pos_manager
    if pos_manager.get_position(TEST_SYMBOL) is not None:
        return True

    # Check IBKR actual positions
    account_positions = context.get_positions_for_account()
    position = account_positions.get(TEST_SYMBOL)
    return position is not None and position.get("qty", 0) > 0


async def _execute_test_buy() -> str:
    """
    Schedule a component test BUY for next minute boundary.

    Does NOT execute immediately - schedules for xx:00s like real strategies.
    Returns immediately with scheduled time info.
    """
    global _pending_buy

    if not context.ibkr_connected.is_set():
        return "❌ IBKR not connected."

    if has_position():
        return "🚫 Already have position in QQQ. Close first before buying again."

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
        f"   Symbol: {TEST_SYMBOL}\n"
        f"   Quantity: {TEST_QUANTITY} share\n"
        f"   Executes in: ~{int(seconds_until)}s\n\n"
        f"Will fetch Tiingo data, apply guardrails, and track position."
    )


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
    - Fetches 1min OHLC from Tiingo (~2s delay)
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

    if not context.ibkr_connected.is_set():
        return "❌ Scheduled BUY failed: IBKR not connected."

    # Safety check: position may have been opened between schedule and execution
    if has_position():
        return "🚫 Scheduled BUY cancelled: already have position in QQQ."

    # Fetch 1min OHLC (same as other strategies)
    tiingo = TiingoService()
    try:
        ohlc_data = await tiingo.get_intraday_ohlc(
            symbol=TEST_SYMBOL,
            days=1,
            interval="1min",
            use_cache=False  # Always fresh
        )

        if not ohlc_data or len(ohlc_data) < 2:
            await tiingo.close()
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
        data_info += f"\n   Trigger: ${trigger_price:.2f} @ {trigger_time} (completed bar)"

        # Store for slippage guardrail check
        context.latest_prices.set(TEST_SYMBOL, trigger_price)

    except Exception as e:
        await tiingo.close()
        return (
            f"🚫 Scheduled BUY BLOCKED\n"
            f"   Failed to fetch OHLC data: {e}\n"
            f"   Cannot verify slippage guardrail"
        )
    finally:
        await tiingo.close()

    # Submit order through guardrails (same as all strategies)
    success = order_service.submit_order(
        symbol=TEST_SYMBOL,
        action="BUY",
        quantity=TEST_QUANTITY,
        strategy_id=STRATEGY_ID
    )

    # Log OHLC data to CSV (same as other strategies)
    logger = SignalLogger.get_or_create(
        symbol=TEST_SYMBOL,
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
            symbol=TEST_SYMBOL,
            account=context.current_account or "",
            strategy_id=STRATEGY_ID,
            action="LONG",
            quantity=TEST_QUANTITY,
            entry_price=trigger_price,
            take_profit=None,  # Component test doesn't use TP/SL
            stop_loss=None
        )

        msg = (
            f"✅ Scheduled BUY executed @ {get_et_timestamp()}\n"
            f"   Strategy ID: {STRATEGY_ID}\n"
            f"   Symbol: {TEST_SYMBOL}\n"
            f"   Quantity: {TEST_QUANTITY}\n"
            f"   Entry: ${trigger_price:.2f}"
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
