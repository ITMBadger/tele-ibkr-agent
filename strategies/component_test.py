"""
Component Testing Module

Manual testing tool for validating trading components via Telegram.
Uses strategy ID 999 and fetches 1min OHLC data for guardrail slippage checks.

Behavior (same as other strategies):
- Fetches 1min OHLC data from Tiingo
- Logs OHLC trigger to CSV in data/logs/
- Activates all guardrails (slippage, position limits, etc.)
- Creates position.json tracking file

Enable/disable this feature in services/telegram.py with ENABLE_TESTING_BUTTONS flag.
"""

import context
from services import order_service
from services.tiingo import TiingoService
from services.time_utils import format_iso_to_et
from services.logger import SignalLogger


# === TESTING CONFIGURATION ===

STRATEGY_ID = "999"
TEST_SYMBOL = "QQQ"
TEST_QUANTITY = 1


# === INLINE KEYBOARD DEFINITION ===

def get_testing_keyboard() -> dict:
    """
    Returns Telegram InlineKeyboard with single test buy button.

    Returns:
        dict: Telegram reply_markup for inline keyboard
    """
    return {
        "inline_keyboard": [
            [
                {"text": "📈 TEST BUY QQQ", "callback_data": "test_buy_qqq"}
            ],
            [
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

    elif callback_data == "test_close":
        return "Component test menu closed."

    else:
        return f"Unknown test action: {callback_data}"


async def _execute_test_buy() -> str:
    """
    Execute a component test BUY order (async).

    Fetches 1min OHLC data from Tiingo for today (guardrail slippage check),
    then submits order through order_service (activates all guardrails).

    BLOCKS order if OHLC fetch fails (can't verify slippage).
    """
    if not context.ibkr_connected.is_set():
        return "❌ IBKR not connected."

    # Fetch 1min OHLC for today (guardrail slippage check)
    tiingo = TiingoService()
    try:
        ohlc_data = await tiingo.get_intraday_ohlc(
            symbol=TEST_SYMBOL,
            days=1,
            interval="1min",
            use_cache=False  # Always fresh for testing
        )

        # Log the data fetch and store price for slippage check
        bar_count = len(ohlc_data)
        latest_bar = ohlc_data[-1] if ohlc_data else None
        data_info = f"\n📊 Fetched {bar_count} 1min bars"
        if latest_bar:
            latest_price = latest_bar['close']
            et_date = format_iso_to_et(latest_bar['date'])
            data_info += f"\n   Latest: ${latest_price:.2f} @ {et_date}"
            # Store for slippage guardrail check
            context.latest_prices.set(TEST_SYMBOL, latest_price)

    except Exception as e:
        await tiingo.close()
        return (
            f"🚫 Component test BLOCKED\n"
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

    if success:
        return (
            f"✅ Component test BUY order submitted\n"
            f"   Strategy ID: {STRATEGY_ID}\n"
            f"   Symbol: {TEST_SYMBOL}\n"
            f"   Quantity: {TEST_QUANTITY}"
            f"{data_info}"
        )
    else:
        return (
            f"🚫 Component test BUY blocked by guardrails\n"
            f"   Check terminal for details"
            f"{data_info}"
        )


# === UTILITY FUNCTIONS ===

def is_test_command(text: str) -> bool:
    """Check if message is the /test command."""
    return text.strip().lower() == "/test"


def is_test_callback(callback_data: str) -> bool:
    """Check if callback is from testing buttons."""
    return callback_data.startswith("test_")
