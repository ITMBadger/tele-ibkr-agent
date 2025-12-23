"""Admin tools for account and position management."""

from typing import Any
import context
from services import pos_manager


def get_account_info() -> str:
    """Get account balances from cached data."""
    if not context.ibkr_connected.is_set():
        return "❌ IBKR not connected."

    balances = context.all_account_balances
    current = context.current_account

    if not balances:
        return "No account data available. Try 'refresh balances'."

    lines = ["💰 **Account Balances**", ""]

    for acc_id in sorted(balances.keys()):
        data = balances[acc_id]
        net = data.get("net_value", 0)
        cash = data.get("cash", 0)
        avail = data.get("available", 0)

        marker = " ✓ active" if acc_id == current else ""
        lines.append(f"**{acc_id}**{marker}")
        lines.append(f"  Net Value: ${net:,.2f}")
        lines.append(f"  Cash: ${cash:,.2f}")
        lines.append(f"  Available: ${avail:,.2f}")
        lines.append("")

    lines.append("_Data cached at startup. Say 'refresh balances' for latest._")

    return "\n".join(lines)


def refresh_balances() -> str:
    """Request fresh balance data from IBKR."""
    if not context.ibkr_connected.is_set():
        return "❌ IBKR not connected."

    context.request_refresh("balances")
    return "🔄 Refreshing account balances... Ask again in a few seconds."


def switch_account(account_id: str) -> str:
    """Switch to a different account."""
    account_id = account_id.upper()

    if not context.ibkr_connected.is_set():
        return "❌ IBKR not connected."

    if account_id not in context.all_accounts:
        accounts_list = ", ".join(context.all_accounts)
        return f"❌ Account '{account_id}' not found.\n\nAvailable: {accounts_list}"

    if account_id == context.current_account:
        return f"Already using account {account_id}."

    old_account = context.current_account
    context.account_switch_queue.put(account_id)

    return f"✅ Switched from **{old_account}** to **{account_id}**"


def get_positions() -> str:
    """Get all current positions from cached data."""
    positions = context.positions.copy()

    if not positions:
        return "No open positions."

    # Group by account
    by_account: dict[str, list] = {}
    for key, data in positions.items():
        acc = data["account"]
        if acc not in by_account:
            by_account[acc] = []
        by_account[acc].append(data)

    lines = ["📊 **Positions**", ""]

    for acc in sorted(by_account.keys()):
        marker = " ✓" if acc == context.current_account else ""
        lines.append(f"**{acc}**{marker}")
        for p in sorted(by_account[acc], key=lambda x: x["symbol"]):
            lines.append(f"  {p['symbol']}: {p['qty']} @ ${p['avg_cost']:.2f}")
        lines.append("")

    lines.append("_Data cached. Say 'refresh positions' for latest._")

    return "\n".join(lines)


def refresh_positions() -> str:
    """Request fresh position data from IBKR."""
    if not context.ibkr_connected.is_set():
        return "❌ IBKR not connected."

    context.request_refresh("positions")
    return "🔄 Refreshing positions... Ask again in a few seconds."


async def get_price(symbol: str, tiingo_service: Any) -> str:
    """Get current price from Tiingo."""
    symbol = symbol.upper()

    if not tiingo_service:
        return "Price service not available"

    try:
        price = await tiingo_service.get_current_price(symbol)
        context.latest_prices.set(symbol, price)
        return f"{symbol}: ${price:.2f}"
    except Exception as e:
        return f"Could not get price for {symbol}: {e}"


# === POSITION TRACKING ===

def get_tracked_positions() -> str:
    """Get all tracked positions from JSON."""
    positions = pos_manager.get_all_positions()

    if not positions:
        return "No tracked positions."

    lines = ["📋 **Tracked Positions**", ""]

    for key, pos in sorted(positions.items()):
        symbol = pos.get("symbol", "?")
        account = pos.get("account", "?")
        action = pos.get("action", "?")
        qty = pos.get("quantity", 0)
        strategy_id = pos.get("strategy_id", "?")
        entry = pos.get("entry_price")
        tp = pos.get("take_profit")
        sl = pos.get("stop_loss")

        marker = " ✓" if account == context.current_account else ""
        lines.append(f"**{symbol}** ({action}) - Strategy {strategy_id}{marker}")
        lines.append(f"  Account: {account}")
        lines.append(f"  Quantity: {qty}")
        if entry:
            lines.append(f"  Entry: ${entry:.2f}")
        if tp:
            lines.append(f"  Take Profit: ${tp:.2f}")
        if sl:
            lines.append(f"  Stop Loss: ${sl:.2f}")
        lines.append("")

    return "\n".join(lines)


def clear_position_records() -> str:
    """Clear all tracked position records from JSON."""
    positions = pos_manager.get_all_positions()

    if not positions:
        return "No tracked positions to clear."

    count = len(positions)
    success = pos_manager.clear_all_positions()

    if success:
        print(f"   Cleared {count} tracked position(s)")
        return f"✅ Cleared {count} tracked position record(s).\n\n_Note: This only clears tracking data. IBKR positions are not affected._"
    else:
        return "❌ Failed to clear position records."
