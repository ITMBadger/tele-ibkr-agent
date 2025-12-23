"""Strategy management tools."""

from typing import Any

import context
from strategies import get_strategy, format_strategy_list


def list_strategies() -> str:
    """List all available trading strategies."""
    return format_strategy_list()


def activate_strategy(strategy_id: str, symbol: str, chat_id: int | None) -> str:
    """Propose activating a strategy - requires confirmation."""
    symbol = symbol.upper()

    strategy_class = get_strategy(strategy_id)
    if not strategy_class:
        return f"Strategy {strategy_id} not found. Use list_strategies to see options."

    if context.active_strategies.get(symbol):
        return f"A strategy is already running on {symbol}. Stop it first."

    if chat_id is None:
        return "Error: No chat context available"

    current_account = context.current_account

    context.pending_strategies.set(str(chat_id), {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "account": current_account
    })

    account_info = f"**Account**: {current_account}"
    if len(context.all_accounts) > 1:
        account_info += f" (of {len(context.all_accounts)})"

    return (
        f"📋 **Strategy Proposal**\n\n"
        f"{account_info}\n"
        f"**Symbol**: {symbol}\n"
        f"**Strategy**: {strategy_class.NAME}\n"
        f"**Description**: {strategy_class.DESCRIPTION}\n"
        f"**Check Interval**: {strategy_class.INTERVAL}s\n"
        f"**Quantity**: {strategy_class.QUANTITY} shares\n\n"
        f"Reply 'yes' to activate or 'no' to cancel."
    )


def confirm_strategy(chat_id: int | None, tiingo_service: Any) -> str:
    """Confirm and start a pending strategy."""
    if chat_id is None:
        return "Error: No chat context available"

    chat_key = str(chat_id)
    pending = context.pending_strategies.get(chat_key)

    if not pending:
        return "No pending strategy to confirm."

    strategy_id = pending["strategy_id"]
    symbol = pending["symbol"]

    strategy_class = get_strategy(strategy_id)
    if not strategy_class:
        return f"Strategy {strategy_id} not found."

    strategy_instance = strategy_class(symbol, tiingo_service)

    context.active_strategies.set(symbol, {
        "strategy": strategy_instance,
        "strategy_id": strategy_id,
        "name": strategy_class.NAME
    })

    context.pending_strategies.delete(chat_key)

    context.log(f"✅ {strategy_class.NAME} activated on {symbol}", "info")

    return f"✅ **{strategy_class.NAME}** is now running on **{symbol}**!"


def cancel_strategy(chat_id: int | None) -> str:
    """Cancel a pending strategy."""
    if chat_id is None:
        return "Error: No chat context available"

    chat_key = str(chat_id)
    pending = context.pending_strategies.get(chat_key)

    if not pending:
        return "No pending strategy to cancel."

    context.pending_strategies.delete(chat_key)
    return "❌ Strategy proposal cancelled."


def stop_strategy(symbol: str) -> str:
    """Stop a running strategy."""
    symbol = symbol.upper()

    current = context.active_strategies.get(symbol)
    if not current:
        return f"No active strategy for {symbol}"

    name = current.get("name", "Strategy")
    context.active_strategies.delete(symbol)
    context.log(f"🛑 {name} stopped on {symbol}", "info")

    return f"🛑 Stopped {name} on {symbol}"


def list_active_strategies() -> str:
    """List all active strategies."""
    active = context.active_strategies.copy()

    if not active:
        return "No active strategies"

    lines = ["📊 **Active Strategies**", ""]
    for symbol, data in active.items():
        name = data.get("name", "Unknown")
        strategy = data.get("strategy")
        interval = strategy.INTERVAL if strategy else "?"
        lines.append(f"• **{symbol}**: {name} (every {interval}s)")

    return "\n".join(lines)
