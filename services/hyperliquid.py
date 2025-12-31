# hyperliquid.py - Hyperliquid exchange connection service.
"""
This service provides Hyperliquid perpetual trading functionality.
Implements BrokerInterface for multi-broker support.

Hyperliquid is a decentralized perpetual futures exchange.
Uses Ethereum wallet addresses for authentication.

Supports:
- Perpetual futures trading (BTC, ETH, etc.)
- Fractional quantities
- Long and short positions
- 24/7 trading
- Testnet for paper trading
"""

import os
import threading
import time
from typing import Any

import context
from services import pos_manager
from services.broker_base import BrokerInterface, BrokerCapabilities, register_broker

# Hyperliquid SDK import - will be installed via requirements.txt
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from eth_account import Account
    HYPERLIQUID_AVAILABLE = True
except ImportError:
    HYPERLIQUID_AVAILABLE = False
    Info = None
    Exchange = None
    constants = None
    Account = None


@register_broker("hyperliquid")
class HyperliquidService(BrokerInterface):
    """
    Hyperliquid perpetual futures trading service.

    Runs in its own thread and communicates via context queues.
    Uses the official hyperliquid-python-sdk for API access.
    """

    # BrokerInterface attributes
    NAME = "hyperliquid"
    CAPABILITIES = BrokerCapabilities(
        supports_fractional=True,       # Perps trade in fractional units
        supports_short=True,            # Perpetuals support shorting
        supports_bracket_orders=False,  # Handle SL/TP separately
        market_hours_24_7=True,         # Crypto trades 24/7
        asset_type="crypto"
    )

    def __init__(self):
        if not HYPERLIQUID_AVAILABLE:
            raise ImportError(
                "hyperliquid-python-sdk not installed. Run: pip install hyperliquid-python-sdk"
            )

        self.private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        self.testnet = os.getenv("HYPERLIQUID_TESTNET", "false").lower() == "true"
        self.wallet_address = ""  # Will be derived from private key

        self._info: Info | None = None
        self._exchange: Exchange | None = None
        self._thread: threading.Thread | None = None
        self._connected = False

        # Track pending orders for position management
        self._pending_orders: dict[str, dict] = {}  # order_id -> {symbol, action, strategy_id}

        # Cache for mid prices
        self._mid_prices: dict[str, float] = {}

    # === CONNECTION ===

    def connect_and_run(self) -> None:
        """Connect to Hyperliquid and start the main processing loop."""
        try:
            env_label = "Testnet" if self.testnet else "Mainnet"
            print(f"\n[Hyperliquid] Connecting to {env_label}...")

            if not self.private_key:
                print("[Hyperliquid] Private key not configured")
                print("   Set HYPERLIQUID_PRIVATE_KEY in .env")
                return

            # Select API URL based on environment
            # Mainnet: https://api.hyperliquid.xyz
            # Testnet: https://api.hyperliquid-testnet.xyz
            base_url = constants.TESTNET_API_URL if self.testnet else constants.MAINNET_API_URL
            print(f"[Hyperliquid] API URL: {base_url}")

            # Create Info client (for market data and account state)
            self._info = Info(base_url, skip_ws=True)

            # Create wallet from private key
            wallet = Account.from_key(self.private_key)
            print(f"[Hyperliquid] Wallet: {wallet.address}")

            # Create Exchange client (for trading)
            # Exchange(wallet, base_url, ...)
            self._exchange = Exchange(
                wallet=wallet,
                base_url=base_url
            )

            # Use derived wallet address (overrides env if different)
            self.wallet_address = wallet.address

            # Test connection by fetching user state
            user_state = self._info.user_state(self.wallet_address)
            if user_state:
                self._connected = True
                context.broker_connected.set()
                print(f"[Hyperliquid] Connected ({env_label})")

                # Load account info and positions
                self._load_account_info(user_state)
                self._load_positions(user_state)

                # Enter main processing loop
                self._process_loop()
            else:
                print("[Hyperliquid] Failed to connect - no user state returned")

        except Exception as e:
            print(f"[Hyperliquid] Error: {type(e).__name__}: {e}")
        finally:
            context.broker_connected.clear()
            self._connected = False
            print("[Hyperliquid] Disconnected")

    def _load_account_info(self, user_state: dict) -> None:
        """Load account balances from Hyperliquid user state."""
        # User state structure:
        # {
        #   "marginSummary": {"accountValue": "1000.0", "totalMarginUsed": "0.0", ...},
        #   "crossMarginSummary": {...},
        #   "assetPositions": [...]
        # }
        margin_summary = user_state.get("marginSummary", {})

        account_value = float(margin_summary.get("accountValue", 0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        available = account_value - total_margin_used

        # Store in context (map to IBKR-like structure)
        context.account_info.set("cash", available)
        context.account_info.set("net_value", account_value)
        context.account_info.set("available", available)
        context.account_info.set("buying_power", available)

        # Set account ID (use wallet address, truncated for display)
        account_id = f"hl-{self.wallet_address[:8]}"
        context.all_accounts = [account_id]
        context.current_account = account_id

        print(f"[Hyperliquid] Account Value: ${account_value:,.2f}")
        print(f"[Hyperliquid] Available: ${available:,.2f}")

    def _load_positions(self, user_state: dict) -> None:
        """Load open positions from Hyperliquid user state."""
        positions_found = 0
        asset_positions = user_state.get("assetPositions", [])

        for pos in asset_positions:
            position_data = pos.get("position", {})
            coin = position_data.get("coin", "")
            szi = float(position_data.get("szi", 0))  # Signed size (negative = short)

            if szi == 0:
                continue

            # Store position
            account_id = context.current_account or f"hl-{self.wallet_address[:8]}"
            key = f"{account_id}:{coin}"

            entry_px = float(position_data.get("entryPx", 0))
            unrealized_pnl = float(position_data.get("unrealizedPnl", 0))

            context.positions.set(key, {
                "account": account_id,
                "symbol": coin,
                "qty": abs(szi),
                "avg_cost": entry_px,
                "unrealized_pnl": unrealized_pnl,
                "side": "LONG" if szi > 0 else "SHORT"
            })
            positions_found += 1
            side = "LONG" if szi > 0 else "SHORT"
            print(f"[Hyperliquid] Position: {coin} {side} {abs(szi)} @ ${entry_px:.2f}")

        if positions_found > 0:
            print(f"[Hyperliquid] Found {positions_found} position(s)")
        else:
            print("[Hyperliquid] No open positions")

        # Reconcile with saved positions
        pos_manager.startup_load_and_reconcile()

    def _process_loop(self) -> None:
        """Main loop: check queues and refresh data periodically."""
        last_refresh = time.time()
        refresh_interval = 30  # seconds

        while not context.shutdown_event.is_set():
            self._check_order_queue()
            self._check_account_switch_queue()
            self._check_refresh_queue()
            self._check_price_request_queue()

            # Periodic refresh of mid prices
            if time.time() - last_refresh > refresh_interval:
                try:
                    self._refresh_mid_prices()
                    last_refresh = time.time()
                except Exception:
                    pass

            time.sleep(0.1)

    def _refresh_mid_prices(self) -> None:
        """Refresh cached mid prices from Hyperliquid."""
        if not self._info:
            return

        try:
            all_mids = self._info.all_mids()
            if all_mids:
                self._mid_prices = {k: float(v) for k, v in all_mids.items()}
        except Exception:
            pass

    def _check_order_queue(self) -> None:
        """Check and execute any pending orders from the queue."""
        while not context.order_queue.empty():
            try:
                signal = context.order_queue.get_nowait()
                self.place_order(
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    order_type=signal.order_type,
                    limit_price=signal.limit_price,
                    strategy_id=signal.strategy_id
                )
            except Exception as e:
                context.log(f"Error processing order: {e}", "error")

    def _check_account_switch_queue(self) -> None:
        """Check and process any pending account switch requests."""
        while not context.account_switch_queue.empty():
            try:
                account_id = context.account_switch_queue.get_nowait()
                self.switch_account(account_id)
            except Exception as e:
                context.log(f"Error switching account: {e}", "error")

    def _check_refresh_queue(self) -> None:
        """Check and process any refresh requests."""
        while not context.refresh_queue.empty():
            try:
                what = context.refresh_queue.get_nowait()
                print(f"[Hyperliquid] Refreshing {what}...")

                user_state = self._info.user_state(self.wallet_address)
                if what in ("balances", "all"):
                    self._load_account_info(user_state)
                    context.log("[Hyperliquid] Account balances updated", "info")

                if what in ("positions", "all"):
                    self._load_positions(user_state)
                    context.log("[Hyperliquid] Positions updated", "info")

            except Exception as e:
                context.log(f"Error refreshing data: {e}", "error")

    def _check_price_request_queue(self) -> None:
        """Check and process any market price requests."""
        while not context.price_request_queue.empty():
            try:
                symbol = context.price_request_queue.get_nowait()
                price = self._get_mid_price(symbol)
                if price:
                    context.market_prices.set(symbol, price)
            except Exception as e:
                print(f"[Hyperliquid] Error processing price request: {e}")

    def _get_mid_price(self, symbol: str) -> float | None:
        """Get mid price for a symbol."""
        symbol = symbol.upper()

        # Check cache first
        if symbol in self._mid_prices:
            return self._mid_prices[symbol]

        # Fetch fresh
        if not self._info:
            return None

        try:
            all_mids = self._info.all_mids()
            if all_mids and symbol in all_mids:
                price = float(all_mids[symbol])
                self._mid_prices[symbol] = price
                return price
        except Exception:
            pass

        return None

    def start_thread(self) -> threading.Thread:
        """Start the Hyperliquid service in a background thread."""
        self._thread = threading.Thread(target=self.connect_and_run, daemon=True)
        self._thread.start()
        return self._thread

    # === ACCOUNT SWITCHING ===

    def switch_account(self, account_id: str) -> bool:
        """
        Switch to a different account.

        Note: Hyperliquid uses single wallet, so this just validates.
        """
        if account_id not in context.all_accounts:
            return False

        context.current_account = account_id
        print(f"[Hyperliquid] Switched to account: {account_id}")
        context.log(f"Switched to account: {account_id}", "info")
        return True

    # === ORDER EXECUTION ===

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        order_type: str = "MKT",
        limit_price: float | None = None,
        strategy_id: str = "manual"
    ) -> str | None:
        """
        Place an order with Hyperliquid.

        """
        if not self._connected or not self._exchange:
            context.log("Cannot place order: not connected", "error")
            return None

        try:
            # Normalize symbol (Hyperliquid uses simple symbols: BTC, ETH)
            symbol = symbol.upper()
            # Remove common suffixes if present
            for suffix in ["USDT", "USD", "PERP", "-PERP"]:
                if symbol.endswith(suffix):
                    symbol = symbol[:-len(suffix)]
                    break

            is_buy = action.upper() == "BUY"

            # Get current mid price for market orders
            if order_type.upper() == "MKT" or not limit_price:
                mid_price = self._get_mid_price(symbol)
                if not mid_price:
                    context.log(f"Cannot get price for {symbol}", "error")
                    return None
                # Add slippage for market order (5% default)
                slippage = 0.05
                if is_buy:
                    px = mid_price * (1 + slippage)
                else:
                    px = mid_price * (1 - slippage)
            else:
                px = limit_price

            # Place order using exchange.market_open for market orders
            print(f"[Hyperliquid] {action} {quantity} {symbol} @ ~${px:.2f}")
            context.log(f"Placing order: {action} {quantity} {symbol}", "trade")

            if order_type.upper() == "MKT":
                # Use market_open for aggressive market order
                result = self._exchange.market_open(
                    name=symbol,
                    is_buy=is_buy,
                    sz=quantity,
                    px=px,
                    slippage=0.05
                )
            else:
                # Use limit order
                from hyperliquid.utils.types import Limit, Tif
                order_type_obj = {"limit": Limit(tif=Tif.Gtc)}
                result = self._exchange.order(
                    name=symbol,
                    is_buy=is_buy,
                    sz=quantity,
                    limit_px=px,
                    order_type=order_type_obj
                )

            # Parse response
            if result and result.get("status") == "ok":
                response = result.get("response", {})
                if response.get("type") == "order":
                    statuses = response.get("data", {}).get("statuses", [])
                    if statuses and "filled" in statuses[0]:
                        filled = statuses[0]["filled"]
                        fill_price = float(filled.get("avgPx", px))
                        order_id = filled.get("oid", str(time.time()))

                        self._handle_fill(
                            order_id=str(order_id),
                            symbol=symbol,
                            action=action.upper(),
                            quantity=quantity,
                            avg_price=fill_price,
                            strategy_id=strategy_id
                        )
                        return str(order_id)

                    elif statuses and "resting" in statuses[0]:
                        # Order is resting (limit order)
                        resting = statuses[0]["resting"]
                        order_id = resting.get("oid", str(time.time()))
                        print(f"[Hyperliquid] Order resting: {order_id}")
                        return str(order_id)

            # Handle error
            error_msg = f"Order failed: {result}"
            print(f"[Hyperliquid] {error_msg}")
            context.log(error_msg, "error")
            return None

        except Exception as e:
            error_msg = f"Order error: {e}"
            print(f"[Hyperliquid] {error_msg}")
            context.log(error_msg, "error")
            return None

    def _handle_fill(
        self,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        avg_price: float,
        strategy_id: str
    ) -> None:
        """Handle order fill - update position tracking."""
        account = context.current_account or f"hl-{self.wallet_address[:8]}"

        if action == "BUY":
            # Open long position
            pos_manager.save_position(
                symbol=symbol,
                account=account,
                strategy_id=strategy_id,
                action="LONG",
                quantity=quantity,
                entry_price=avg_price
            )
        elif action == "SELL":
            # Close existing position or open short
            existing = pos_manager.get_position(symbol)
            if existing and existing.get("action") == "LONG":
                # Closing long
                pos_manager.remove_position(symbol=symbol, account=account)
            else:
                # Opening short
                pos_manager.save_position(
                    symbol=symbol,
                    account=account,
                    strategy_id=strategy_id,
                    action="SHORT",
                    quantity=quantity,
                    entry_price=avg_price
                )

        # Log fill
        msg = f"Order {order_id}: Filled @ ${avg_price:.4f}"
        print(f"[Hyperliquid] {msg}")
        context.log(msg, "trade")

    def get_market_price_sync(self, symbol: str, timeout: float = 5.0) -> float | None:
        """
        Get market price synchronously.

        """
        if not self._connected or not self._info:
            return None

        # Normalize symbol
        symbol = symbol.upper()
        for suffix in ["USDT", "USD", "PERP", "-PERP"]:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
                break

        price = self._get_mid_price(symbol)
        if price:
            context.market_prices.set(symbol, price)
        return price
