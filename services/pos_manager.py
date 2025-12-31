"""
Position Manager - JSON persistence for position tracking.

Handles:
- Saving position records after entry (LONG/SHORT)
- Loading positions on startup
- Reconciling with IBKR positions
- Position management data (TP/SL, strategy, entry price)

File location: data/positions.json
"""

import json
import os
from pathlib import Path
from typing import Any

import context
from services.time_centralize_utils import get_et_timestamp


# === FILE PATH ===

DATA_DIR = Path(__file__).parent.parent / "data"
POSITIONS_FILE = DATA_DIR / "positions.json"


# === DATA STRUCTURE ===

def _empty_data() -> dict:
    """Return empty positions data structure."""
    return {
        "positions": {},
        "last_updated": None
    }


def _get_position_key(account: str, symbol: str) -> str:
    """Generate unique key for a position."""
    return f"{account}:{symbol}"


# === FILE OPERATIONS ===

def load_positions() -> dict:
    """
    Load positions from JSON file.

    """
    if not POSITIONS_FILE.exists():
        return _empty_data()

    try:
        with open(POSITIONS_FILE, "r") as f:
            data = json.load(f)
            return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"   Warning: Could not load positions file: {e}")
        return _empty_data()


def save_positions(data: dict) -> bool:
    """
    Save positions to JSON file.

    """
    try:
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        data["last_updated"] = get_et_timestamp()

        with open(POSITIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"   Error saving positions file: {e}")
        return False


def clear_all_positions() -> bool:
    """
    Clear all position records from JSON.

    """
    return save_positions(_empty_data())


# === POSITION MANAGEMENT ===

def save_position(
    symbol: str,
    account: str,
    strategy_id: str,
    action: str,
    quantity: int,
    entry_price: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None
) -> bool:
    """
    Save a new position record after entry.

    """
    data = load_positions()
    key = _get_position_key(account, symbol)

    data["positions"][key] = {
        "symbol": symbol.upper(),
        "account": account,
        "strategy_id": strategy_id,
        "action": action.upper(),
        "quantity": quantity,
        "entry_price": entry_price,
        "entry_time": get_et_timestamp(),
        "take_profit": take_profit,
        "stop_loss": stop_loss
    }

    success = save_positions(data)

    if success:
        print(f"   Position saved: {action} {quantity} {symbol} (strategy {strategy_id})")

    return success


def remove_position(symbol: str, account: str | None = None) -> bool:
    """
    Remove a position record after exit.

    """
    acc = account or context.current_account
    if not acc:
        return False

    data = load_positions()
    key = _get_position_key(acc, symbol.upper())

    if key in data["positions"]:
        del data["positions"][key]
        success = save_positions(data)

        if success:
            print(f"   Position removed: {symbol}")

        return success

    return True  # Already not present


def get_position(symbol: str, account: str | None = None) -> dict | None:
    """
    Get a position record.

    """
    acc = account or context.current_account
    if not acc:
        return None

    data = load_positions()
    key = _get_position_key(acc, symbol.upper())

    return data["positions"].get(key)


def get_all_positions() -> dict[str, dict]:
    """
    Get all position records.

    """
    data = load_positions()
    return data.get("positions", {})


def update_position(
    symbol: str,
    account: str | None = None,
    **updates
) -> bool:
    """
    Update fields on an existing position.

    """
    acc = account or context.current_account
    if not acc:
        return False

    data = load_positions()
    key = _get_position_key(acc, symbol.upper())

    if key not in data["positions"]:
        return False

    data["positions"][key].update(updates)
    return save_positions(data)


# === STARTUP RECONCILIATION ===

def reconcile_with_ibkr() -> dict:
    """
    Reconcile JSON positions with actual IBKR positions.

    Called on startup after IBKR positions are loaded.

    - If JSON position exists but IBKR doesn't have it:
      Log warning and remove from JSON
    - If IBKR has position but JSON doesn't:
      Just note it (user may have opened manually)

    """
    result = {
        "removed": [],      # Positions in JSON but not IBKR (closed externally)
        "untracked": [],    # Positions in IBKR but not JSON (opened manually)
        "matched": []       # Positions in both
    }

    json_positions = get_all_positions()
    ibkr_positions = context.positions.copy()

    # Check each JSON position against IBKR
    positions_to_remove = []

    for key, json_pos in json_positions.items():
        if key in ibkr_positions:
            # Position exists in both - matched
            result["matched"].append(json_pos["symbol"])
        else:
            # Position in JSON but not IBKR - was closed externally
            positions_to_remove.append(key)
            result["removed"].append({
                "symbol": json_pos["symbol"],
                "account": json_pos["account"],
                "strategy_id": json_pos["strategy_id"]
            })

    # Check for IBKR positions not in JSON
    for key, ibkr_pos in ibkr_positions.items():
        if key not in json_positions:
            result["untracked"].append({
                "symbol": ibkr_pos["symbol"],
                "account": ibkr_pos["account"],
                "qty": ibkr_pos["qty"]
            })

    # Remove closed positions from JSON
    if positions_to_remove:
        data = load_positions()
        for key in positions_to_remove:
            if key in data["positions"]:
                del data["positions"][key]
        save_positions(data)

    return result


def log_reconciliation_result(result: dict) -> None:
    """
    Log the reconciliation result to terminal and Telegram.

    """
    messages = []

    # Log removed positions (closed externally)
    if result["removed"]:
        for pos in result["removed"]:
            msg = f"Position {pos['symbol']} (strategy {pos['strategy_id']}) was closed externally - removed from tracking"
            print(f"   Warning: {msg}")
            messages.append(msg)

    # Log untracked positions (opened manually)
    if result["untracked"]:
        symbols = [p["symbol"] for p in result["untracked"]]
        msg = f"Untracked IBKR positions found: {', '.join(symbols)} (opened manually or before tracking)"
        print(f"   Info: {msg}")

    # Log matched positions
    if result["matched"]:
        print(f"   Tracking {len(result['matched'])} position(s): {', '.join(result['matched'])}")

    # Send warnings to Telegram
    if messages:
        combined = "\n".join([f"Warning: {m}" for m in messages])
        context.log(combined, "warning")


def startup_load_and_reconcile() -> dict:
    """
    Main startup function - load JSON and reconcile with IBKR.

    Call this after IBKR positions are loaded.

    """
    print("\n📂 Loading position records...")

    json_positions = get_all_positions()

    if not json_positions:
        print("   No saved positions to restore")
        return {"removed": [], "untracked": [], "matched": []}

    print(f"   Found {len(json_positions)} saved position(s)")

    result = reconcile_with_ibkr()
    log_reconciliation_result(result)

    return result
