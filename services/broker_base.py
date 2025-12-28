# broker_base.py - Abstract base class for broker implementations.
"""
This module defines the interface that all broker implementations must follow.
This allows the system to work with multiple brokers (IBKR, Binance, etc.)
without changing the rest of the codebase.

To add a new broker:
1. Create a new file (e.g., services/my_broker.py)
2. Implement a class that inherits from BrokerInterface
3. Add the broker to BROKER_REGISTRY
4. Set BROKER=my_broker in .env
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BrokerCapabilities:
    """Describes what a broker can do."""
    supports_fractional: bool = False      # Can trade 0.001 units?
    supports_short: bool = True            # Can short sell?
    supports_bracket_orders: bool = False  # Parent + SL + TP in one call?
    market_hours_24_7: bool = False        # Trades 24/7? (crypto)
    asset_type: str = "stock"              # "stock", "crypto", "forex"


class BrokerInterface(ABC):
    """
    Abstract base class for all broker implementations.

    Each broker runs in its own thread and communicates via context queues.
    The interface is designed to be simple and consistent across brokers.
    """

    # Override in subclass
    NAME: str = "base"
    CAPABILITIES: BrokerCapabilities = BrokerCapabilities()

    @abstractmethod
    def connect_and_run(self) -> None:
        """
        Connect to the broker and start the main processing loop.

        This method should:
        1. Establish connection to broker API
        2. Load account info and positions
        3. Enter main loop checking context queues
        4. Handle reconnection if needed

        Runs in a dedicated thread (Thread 2).
        """
        pass

    @abstractmethod
    def start_thread(self):
        """
        Start the broker service in a background thread.

        Returns:
            threading.Thread: The started thread
        """
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str = "MKT",
        limit_price: float | None = None,
        strategy_id: str = "manual"
    ) -> Any:
        """
        Place an order with the broker.

        Args:
            symbol: Trading symbol (e.g., "QQQ" for stocks, "BTCUSDT" for crypto)
            action: "BUY" or "SELL"
            quantity: Number of units (can be fractional for crypto)
            order_type: "MKT" (market) or "LMT" (limit)
            limit_price: Required if order_type is "LMT"
            strategy_id: Strategy ID for position tracking

        Returns:
            Order ID or None if failed
        """
        pass

    @abstractmethod
    def get_market_price_sync(self, symbol: str, timeout: float = 5.0) -> float | None:
        """
        Get current market price synchronously.

        Used for slippage checks before order execution.

        Args:
            symbol: Trading symbol
            timeout: Max wait time in seconds

        Returns:
            Current market price or None if unavailable
        """
        pass

    @abstractmethod
    def switch_account(self, account_id: str) -> bool:
        """
        Switch to a different trading account.

        Args:
            account_id: Account identifier

        Returns:
            True if successful, False otherwise
        """
        pass


# Registry of available brokers
# Add new brokers here after implementing them
BROKER_REGISTRY: dict[str, type[BrokerInterface]] = {}


def register_broker(name: str):
    """Decorator to register a broker implementation."""
    def decorator(cls: type[BrokerInterface]):
        BROKER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_broker(name: str) -> type[BrokerInterface]:
    """Get a broker class by name."""
    name_lower = name.lower()
    if name_lower not in BROKER_REGISTRY:
        available = ", ".join(BROKER_REGISTRY.keys())
        raise ValueError(f"Unknown broker: {name}. Available: {available}")
    return BROKER_REGISTRY[name_lower]


def list_brokers() -> list[str]:
    """List all registered broker names."""
    return list(BROKER_REGISTRY.keys())
