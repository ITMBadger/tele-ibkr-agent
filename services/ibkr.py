# ibkr.py - Interactive Brokers connection service.
"""
This service runs in a dedicated thread (Thread 2) because the IBKR API
uses blocking callbacks. It communicates with the async services via
the context module's thread-safe queues.

Uses reqAccountSummary() for clean multi-account balance data.

Update this file to change broker behavior.
Does NOT affect: telegram.py, agent.py, tiingo.py, strategies.py
"""

import os
import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.ticktype import TickTypeEnum

import context
from services import pos_manager


class IBKRService(EWrapper, EClient):
    """
    IBKR API wrapper that handles connection and order execution.

    Runs in its own thread to prevent blocking the async event loop.
    Communicates via context.log_queue (outbound) and context.order_queue (inbound).
    
    Uses reqAccountSummary for clean, multi-account balance data.
    """

    SUMMARY_TAGS = "NetLiquidation,TotalCashValue,AvailableFunds,BuyingPower"
    SUMMARY_REQ_ID = 9000

    # TEMP: Auto SL/TP Settings
    AUTO_BRACKET_ENABLED = True
    SL_PCT = 0.01  # 1% stop loss
    TP_PCT = 0.02  # 2% take profit

    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))

        self._next_order_id: int | None = None
        self._order_id_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._accounts_received = threading.Event()
        
        # Store all account balances: {account_id: {cash, net_value, ...}}
        self._account_balances: dict[str, dict] = {}

        # Flags to prevent duplicate prints
        self._accounts_printed = False
        self._balance_table_printed = False
        self._positions_printed = False
        self._initial_load_complete = False

        # Market price snapshot (for slippage checks)
        self._price_req_id_base = 5000
        self._price_req_counter = 0
        self._pending_prices: dict[int, dict] = {}  # req_id -> {symbol, price, event}

        # Track pending orders for position management
        self._pending_orders: dict[int, dict] = {}  # order_id -> {symbol, action, strategy_id}

        # Track processed order fills to prevent duplicate messages
        self._processed_fills: set[int] = set()  # order_ids already logged

    # === CONNECTION ===

    def connect_and_run(self) -> None:
        """Connect to TWS/Gateway and start the message loop."""
        try:
            print(f"\n🔌 Connecting to IBKR at {self.host}:{self.port}...")
            self.connect(self.host, self.port, self.client_id)
            
            if not self.isConnected():
                print(f"❌ TCP connection failed to {self.host}:{self.port}")
                return

            # Start message loop for callbacks
            threading.Thread(target=self._message_pump, daemon=True).start()

            # Wait for nextValidId (confirms connection)
            if not self._order_id_event.wait(timeout=10):
                print(f"❌ IBKR handshake timeout - check TWS API settings")
                return

            context.ibkr_connected.set()
            print(f"✅ IBKR connected (port {self.port}, client ID {self.client_id})")

            # Wait for accounts (IBKR sends managedAccounts automatically on connect)
            if self._accounts_received.wait(timeout=5):
                time.sleep(0.5)
                self._request_initial_data()
            else:
                print("⚠️  No accounts received")
            
            # Enter main processing loop
            self._process_loop()

        except ConnectionRefusedError:
            print(f"❌ Connection refused - is TWS/Gateway running on port {self.port}?")
        except Exception as e:
            print(f"❌ IBKR Error: {type(e).__name__}: {e}")
        finally:
            context.ibkr_connected.clear()
            self.disconnect()
            print("📴 Disconnected from IBKR")

    def _request_initial_data(self) -> None:
        """Request initial account summary and positions."""
        self.reqAccountSummary(self.SUMMARY_REQ_ID, "All", self.SUMMARY_TAGS)
        self.reqPositions()

    def _message_pump(self) -> None:
        """Background thread that continuously processes IBKR messages."""
        while not context.shutdown_event.is_set() and self.isConnected():
            self.run()
            time.sleep(0.01)

    def _process_loop(self) -> None:
        """Main loop: check queues and keep connection alive."""
        while not context.shutdown_event.is_set():
            self._check_order_queue()
            self._check_account_switch_queue()
            self._check_refresh_queue()
            self._check_price_request_queue()
            time.sleep(0.1)

    def _check_account_switch_queue(self) -> None:
        """Check and process any pending account switch requests."""
        while not context.account_switch_queue.empty():
            try:
                account_id = context.account_switch_queue.get_nowait()
                self.switch_account(account_id)
            except Exception as e:
                context.log(f"Error switching account: {e}", "error")

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

    def _check_refresh_queue(self) -> None:
        """Check and process any refresh requests."""
        while not context.refresh_queue.empty():
            try:
                what = context.refresh_queue.get_nowait()
                print(f"   🔄 Refreshing {what}...")

                if what in ("balances", "all"):
                    self._account_balances.clear()
                    self._balance_table_printed = False  # Reset flag to allow printing
                    self.reqAccountSummary(self.SUMMARY_REQ_ID, "All", self.SUMMARY_TAGS)

                if what in ("positions", "all"):
                    self._positions_printed = False  # Reset flag to allow printing
                    self.reqPositions()

            except Exception as e:
                context.log(f"Error refreshing data: {e}", "error")

    def _check_price_request_queue(self) -> None:
        """Check and process any market price requests."""
        while not context.price_request_queue.empty():
            try:
                symbol = context.price_request_queue.get_nowait()
                self._request_market_price(symbol)
            except Exception as e:
                print(f"   Error processing price request: {e}")

    def _request_market_price(self, symbol: str) -> None:
        """
        Request market price snapshot using streaming mode.

        1. Call reqMktData with snapshot=False (streaming mode)
        2. Capture the first price tick
        3. Cancel the subscription immediately
        """
        req_id = self._price_req_id_base + self._price_req_counter
        self._price_req_counter += 1

        # Create event for waiting
        event = threading.Event()
        self._pending_prices[req_id] = {
            "symbol": symbol,
            "price": None,
            "event": event
        }

        contract = self._create_stock_contract(symbol)

        # Request streaming market data (snapshot=False for streaming)
        self.reqMktData(req_id, contract, "", False, False, [])

    def get_market_price_sync(self, symbol: str, timeout: float = 5.0) -> float | None:
        """
        Get market price synchronously (called from IBKR thread).

        Uses reqMktData in streaming mode, captures first tick, then cancels.

        Args:
            symbol: Stock symbol
            timeout: Max time to wait for price

        Returns:
            Market price or None if unavailable
        """
        req_id = self._price_req_id_base + self._price_req_counter
        self._price_req_counter += 1

        event = threading.Event()
        self._pending_prices[req_id] = {
            "symbol": symbol,
            "price": None,
            "event": event
        }

        contract = self._create_stock_contract(symbol)
        self.reqMktData(req_id, contract, "", False, False, [])

        # Wait for price tick
        if event.wait(timeout=timeout):
            price = self._pending_prices[req_id].get("price")
            del self._pending_prices[req_id]
            return price
        else:
            # Timeout - cancel and cleanup
            self.cancelMktData(req_id)
            if req_id in self._pending_prices:
                del self._pending_prices[req_id]
            return None

    def start_thread(self) -> threading.Thread:
        """Start the IBKR service in a background thread."""
        self._thread = threading.Thread(target=self.connect_and_run, daemon=True)
        self._thread.start()
        return self._thread

    # === ACCOUNT SWITCHING ===

    def switch_account(self, account_id: str) -> bool:
        """Switch to a different account."""
        if account_id not in context.all_accounts:
            return False
        
        context.current_account = account_id
        self._update_current_account_info()
        
        print(f"   🔄 Switched to account: {account_id}")
        context.log(f"Switched to account: {account_id}", "info")
        
        return True

    def _update_current_account_info(self) -> None:
        """Update context.account_info with current account data."""
        current = context.current_account
        if current and current in self._account_balances:
            data = self._account_balances[current]
            context.account_info.set("cash", data.get("cash", 0))
            context.account_info.set("net_value", data.get("net_value", 0))
            context.account_info.set("available", data.get("available", 0))
            context.account_info.set("buying_power", data.get("buying_power", 0))

    # === ORDER EXECUTION ===

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: float | None = None,
        strategy_id: str = "manual"
    ) -> int | None:
        """Place an order with IBKR."""
        if self._next_order_id is None:
            context.log("Cannot place order: not connected", "error")
            return None

        # TEMP: Auto-bracket logic for BUY orders
        if self.AUTO_BRACKET_ENABLED and action.upper() == "BUY":
            return self._place_bracket_order(
                symbol, action, quantity, order_type, limit_price, strategy_id
            )

        return self._execute_single_order(
            symbol, action, quantity, order_type, limit_price, strategy_id
        )

    def _execute_single_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: float | None,
        strategy_id: str
    ) -> int:
        """Standard single order execution logic."""
        contract = self._create_stock_contract(symbol)
        order = self._create_order(action, quantity, order_type, limit_price)

        order_id = self._next_order_id
        self._next_order_id += 1

        # Track order info for position management
        self._pending_orders[order_id] = {
            "symbol": symbol.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "strategy_id": strategy_id
        }

        print(f"   📤 {action} {quantity} {symbol} ({order_type})")
        context.log(f"Placing order: {action} {quantity} {symbol} @ {order_type}", "trade")
        self.placeOrder(order_id, contract, order)

        return order_id

    def _place_bracket_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: float | None,
        strategy_id: str
    ) -> int:
        """Place a bracket order (Entry + SL + TP)."""
        # 1. Get current price for SL/TP calculation (reuse verified price if possible)
        price = context.market_prices.get(symbol.upper())
        if price is None:
            price = self.get_market_price_sync(symbol)

        if price is None:
            print(f"   ⚠️ Could not get price for {symbol} bracket - falling back to single order")
            return self._execute_single_order(symbol, action, quantity, order_type, limit_price, strategy_id)

        # 2. Calculate Exit Prices
        sl_price = round(price * (1 - self.SL_PCT), 2)
        tp_price = round(price * (1 + self.TP_PCT), 2)

        # 3. Setup IDs (reserve 3 IDs)
        parent_id = self._next_order_id
        self._next_order_id += 3

        contract = self._create_stock_contract(symbol)

        # 4. Parent Order (Entry)
        parent = self._create_order(action, quantity, order_type, limit_price)
        parent.orderId = parent_id
        parent.transmit = False

        # 5. Stop Loss Order (STP)
        sl_order = self._create_order("SELL", quantity, "STP", None)
        sl_order.orderId = parent_id + 1
        sl_order.parentId = parent_id
        sl_order.auxPrice = sl_price
        sl_order.transmit = False

        # 6. Take Profit Order (LMT)
        tp_order = self._create_order("SELL", quantity, "LMT", tp_price)
        tp_order.orderId = parent_id + 2
        tp_order.parentId = parent_id
        tp_order.transmit = True

        # 7. Track all parts for position management
        for i, order_part in enumerate([parent, sl_order, tp_order]):
            self._pending_orders[parent_id + i] = {
                "symbol": symbol.upper(),
                "action": order_part.action,
                "quantity": quantity,
                "strategy_id": strategy_id
            }

        print(f"   📤 BRACKET BUY {quantity} {symbol} (Entry Ref: ${price:.2f})")
        print(f"      ↳ [SL: ${sl_price:.2f}] [TP: ${tp_price:.2f}]")
        context.log(f"Placing bracket: {symbol} SL: {sl_price} TP: {tp_price}", "trade")

        self.placeOrder(parent_id, contract, parent)
        self.placeOrder(parent_id + 1, contract, sl_order)
        self.placeOrder(parent_id + 2, contract, tp_order)

        return parent_id

    def _create_stock_contract(self, symbol: str) -> Contract:
        """Create a US stock contract."""
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract

    def _create_order(
        self,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: float | None
    ) -> Order:
        """Create an order object."""
        order = Order()
        order.action = action.upper()
        order.totalQuantity = quantity
        order.orderType = order_type
        order.account = context.current_account

        if order_type == "LMT" and limit_price is not None:
            order.lmtPrice = limit_price

        order.eTradeOnly = False
        order.firmQuoteOnly = False

        return order

    # === CALLBACKS: Connection ===

    def nextValidId(self, orderId: int) -> None:
        """Called when connection is established with next valid order ID."""
        self._next_order_id = orderId
        self._order_id_event.set()

    def managedAccounts(self, accountsList: str) -> None:
        """Called with list of managed accounts (comma-separated)."""
        if self._accounts_printed:
            return
        
        accounts = [a.strip() for a in accountsList.split(",") if a.strip()]
        context.all_accounts = accounts
        
        if accounts:
            context.current_account = accounts[0]
            
            if len(accounts) == 1:
                print(f"📋 Account: {accounts[0]}")
            else:
                print(f"📋 Accounts: {', '.join(accounts)}")
                print(f"   Active: {accounts[0]}")
            
            self._accounts_printed = True
        
        self._accounts_received.set()

    # === CALLBACKS: Account Summary ===

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str) -> None:
        """Called for each account summary value."""
        if account not in self._account_balances:
            self._account_balances[account] = {}
        
        try:
            float_value = float(value)
        except ValueError:
            return
        
        if tag == "NetLiquidation":
            self._account_balances[account]["net_value"] = float_value
        elif tag == "TotalCashValue":
            self._account_balances[account]["cash"] = float_value
        elif tag == "AvailableFunds":
            self._account_balances[account]["available"] = float_value
        elif tag == "BuyingPower":
            self._account_balances[account]["buying_power"] = float_value

    def accountSummaryEnd(self, reqId: int) -> None:
        """Called when account summary is complete."""
        # Cancel subscription immediately to prevent more callbacks
        self.cancelAccountSummary(self.SUMMARY_REQ_ID)
        
        if len(self._account_balances) == 0:
            return
        
        # Always update context
        self._update_current_account_info()
        context.all_account_balances = self._account_balances.copy()
        
        # Only print if not already printed in this cycle
        if not self._balance_table_printed:
            self._print_balance_table()
            self._balance_table_printed = True
            
            # Notify if this was a refresh (after initial load)
            if self._initial_load_complete:
                context.log("✅ Account balances updated", "info")

    def _print_balance_table(self) -> None:
        """Print formatted balance table to terminal."""
        print(f"\n💰 Account Balances:")
        print(f"   {'Account':<12} {'Net Value':>14} {'Cash':>14} {'Available':>14}")
        print(f"   {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
        
        for acc_id in sorted(self._account_balances.keys()):
            data = self._account_balances[acc_id]
            net = data.get("net_value", 0)
            cash = data.get("cash", 0)
            avail = data.get("available", 0)
            marker = " ◄" if acc_id == context.current_account else ""
            print(f"   {acc_id:<12} ${net:>12,.2f} ${cash:>12,.2f} ${avail:>12,.2f}{marker}")
        
        print()

    # === CALLBACKS: Positions ===

    def position(self, account: str, contract: Contract, pos: float, avgCost: float) -> None:
        """Called with position updates."""
        if pos != 0:
            key = f"{account}:{contract.symbol}"
            context.positions.set(key, {
                "account": account,
                "symbol": contract.symbol,
                "qty": int(pos),
                "avg_cost": avgCost
            })
        else:
            key = f"{account}:{contract.symbol}"
            context.positions.delete(key)

    def positionEnd(self) -> None:
        """Called when all positions have been received."""
        # Only print if not already printed in this cycle
        if self._positions_printed:
            return
        
        self._positions_printed = True
        positions = context.positions.copy()
        
        if not positions:
            print("📊 No open positions\n")
        else:
            by_account: dict[str, list] = {}
            for key, data in positions.items():
                acc = data["account"]
                if acc not in by_account:
                    by_account[acc] = []
                by_account[acc].append(data)
            
            print(f"📊 Positions:")
            for acc in sorted(by_account.keys()):
                marker = " ◄" if acc == context.current_account else ""
                print(f"   [{acc}]{marker}")
                for p in sorted(by_account[acc], key=lambda x: x["symbol"]):
                    print(f"      {p['symbol']}: {p['qty']} @ ${p['avg_cost']:.2f}")
            print()
        
        # Mark initial load as complete after positions are done
        if not self._initial_load_complete:
            self._initial_load_complete = True

            # Reconcile saved positions with IBKR on startup
            pos_manager.startup_load_and_reconcile()

        else:
            # Notify if this was a refresh (not initial load)
            context.log("✅ Positions updated", "info")

    # === CALLBACKS: Orders ===

    def orderStatus(
        self,
        orderId: int,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float = 0.0
    ) -> None:
        """Called when order status changes."""
        if status in ("Filled", "Cancelled", "Inactive"):
            # Skip if already processed (IBKR can send duplicate status updates)
            if orderId in self._processed_fills:
                return
            self._processed_fills.add(orderId)

            msg = f"Order {orderId}: {status}"
            if status == "Filled":
                msg += f" @ ${avgFillPrice:.2f}"

                # Handle position management on fill
                order_info = self._pending_orders.get(orderId)
                if order_info and avgFillPrice > 0:
                    symbol = order_info["symbol"]
                    action = order_info["action"]
                    quantity = order_info["quantity"]
                    strategy_id = order_info["strategy_id"]
                    account = context.current_account

                    if action == "BUY":
                        # Open new position
                        pos_manager.save_position(
                            symbol=symbol,
                            account=account,
                            strategy_id=strategy_id,
                            action="LONG",
                            quantity=quantity,
                            entry_price=avgFillPrice
                        )
                    elif action == "SELL":
                        # Close existing position
                        pos_manager.remove_position(symbol=symbol, account=account)

                # Cleanup tracked order
                if orderId in self._pending_orders:
                    del self._pending_orders[orderId]

            print(f"   📝 {msg}")
            context.log(msg, "trade" if status == "Filled" else "info")

    def openOrder(self, orderId: int, contract: Contract, order: Order, orderState) -> None:
        """Called with open order details."""
        pass

    def openOrderEnd(self) -> None:
        """Called when all open orders have been received."""
        pass

    # === CALLBACKS: Market Data ===

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:
        """
        Called when a price tick arrives from market data subscription.

        We capture LAST, BID, or ASK price and cancel the subscription immediately.
        """
        if reqId not in self._pending_prices:
            return

        # Accept LAST (4), BID (1), or ASK (2) prices - prefer LAST
        # TickTypeEnum.LAST = 4, TickTypeEnum.BID = 1, TickTypeEnum.ASK = 2
        valid_tick_types = {1, 2, 4}  # BID, ASK, LAST

        if tickType in valid_tick_types and price > 0:
            pending = self._pending_prices[reqId]

            # Only capture first valid price
            if pending["price"] is None:
                pending["price"] = price

                # Cancel subscription immediately
                self.cancelMktData(reqId)

                # Store in context for cross-thread access
                context.market_prices.set(pending["symbol"], price)

                # Signal that price is ready
                pending["event"].set()

    # === CALLBACKS: Errors ===

    def error(self, reqId: int, errorTime: int, errorCode: int, errorString: str, 
              advancedOrderRejectJson: str = "") -> None:
        """Handle errors and notifications from IBKR."""
        silent_codes = {
            2104, 2106, 2158, 2103, 2105, 2157,
            2107, 2108, 2119, 2100,
        }
        
        if errorCode in silent_codes:
            return
        
        warning_codes = {2109, 2137}
        
        if errorCode in warning_codes:
            print(f"   ⚠️  [{errorCode}] {errorString}")
            return
        
        if errorCode >= 1000:
            print(f"   ❌ Error [{errorCode}]: {errorString}")
            context.log(f"IBKR Error {errorCode}: {errorString}", "error")

    def connectionClosed(self) -> None:
        """Called when connection is closed."""
        print("   📴 Connection closed by IBKR")
        context.ibkr_connected.clear()