# context.py - Thread-safe communication bridge between services.
"""
This module provides shared state and message queues that allow
the async services (Telegram, Tiingo, Agent) to communicate with
the sync IBKR service running in a separate thread.

This module contains ONLY:
- Queues for inter-thread communication
- Shared state (positions, accounts, prices)
- Simple helper functions

Business logic (order submission, guardrails) is in services/order_service.py
Data classes are in models.py
"""

import queue
import threading
from typing import Any

from models import TradeSignal, LogMessage


class ThreadSafeDict:
    """A simple thread-safe dictionary wrapper."""

    def __init__(self):
        self._dict: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._dict.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._dict[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._dict.pop(key, None)

    def copy(self) -> dict[str, Any]:
        with self._lock:
            return self._dict.copy()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._dict.keys())


# === QUEUES ===

log_queue: queue.Queue[LogMessage] = queue.Queue()
order_queue: queue.Queue[TradeSignal] = queue.Queue()
account_switch_queue: queue.Queue[str] = queue.Queue()
refresh_queue: queue.Queue[str] = queue.Queue()  # "balances", "positions", "all"
price_request_queue: queue.Queue[str] = queue.Queue()  # symbol for market price request


# === SHARED STATE ===

latest_prices = ThreadSafeDict()   # {"QQQ": 485.50} - from Tiingo
market_prices = ThreadSafeDict()   # {"QQQ": 485.30} - from IBKR real-time for slippage check
positions = ThreadSafeDict()       # {"U123:QQQ": {"account": "U123", "symbol": "QQQ", ...}}
account_info = ThreadSafeDict()    # Current account: {"cash": 100000, "net_value": 150000, ...}

# Multi-account support
all_accounts: list[str] = []                    # ["U16279802", "U17002389", ...]
current_account: str | None = None              # Currently selected account
all_account_balances: dict[str, dict] = {}      # {acc_id: {cash, net_value, available, buying_power}}

# Active strategies: {"QQQ": {"strategy": <instance>, "name": "200 EMA Long"}}
active_strategies = ThreadSafeDict()

# Pending strategies: {chat_id: {"strategy_id": "1", "symbol": "QQQ"}}
pending_strategies = ThreadSafeDict()

# Conversation history: {chat_id: [list of message contents]}
conversation_history = ThreadSafeDict()


# === CONNECTION STATE ===

ibkr_connected = threading.Event()
shutdown_event = threading.Event()


# === STRATEGY LOOP ===

# Task reference (set by main.py on startup)
strategy_loop_task: Any = None


# === HELPER FUNCTIONS ===

def log(message: str, level: str = "info") -> None:
    """Add a log message to the queue for Telegram to send."""
    log_queue.put(LogMessage(message=message, level=level))


def request_refresh(what: str = "all") -> None:
    """Request IBKR to refresh data. what: 'balances', 'positions', or 'all'"""
    refresh_queue.put(what)


def get_positions_for_account(account_id: str | None = None) -> dict[str, dict]:
    """Get positions for a specific account (or current account)."""
    acc = account_id or current_account
    if not acc:
        return {}
    
    result = {}
    for key, data in positions.copy().items():
        if data.get("account") == acc:
            result[data["symbol"]] = data
    return result


def clear_conversation_history(chat_id: int | None = None) -> None:
    """Clear conversation history for a specific chat or all chats."""
    if chat_id is not None:
        conversation_history.delete(str(chat_id))
    else:
        conversation_history._dict.clear()
