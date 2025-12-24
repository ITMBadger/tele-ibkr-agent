# Telegram-IBKR Trading Bot

A secure, strategy-focused trading bot that connects Telegram, Interactive Brokers, and Gemini AI. Chat naturally to check prices, manage positions, and deploy automated strategies with built-in safety guardrails.

**Key Features**:
- 🤖 Natural language interface via Telegram + Gemini AI
- 🛡️ **Hardcoded guardrails** (account whitelist + max order size)
- 📊 **Position tracking** with JSON persistence across restarts
- 🎯 **Strategy-only trading** - manual orders removed for safety
- 🧪 Testing mode with Telegram inline keyboard buttons
- 🔌 Direct IBKR integration (paper & live trading)
- 📝 Complete terminal logging
- 🎯 Zero framework dependencies for Telegram (native API only)
- 👥 Multi-account support with easy switching

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         THREAD 1 (Async Event Loop)                     │
│                                                                         │
│  ┌──────────────┐      ┌──────────────┐      �┌──────────────────┐      │
│  │              │      │              │      │                  │      │
│  │   Telegram   │◄────►│    Agent     │◄────►│     Tiingo       │      │
│  │     Bot      │      │   (Gemini)   │      │   Market Data    │      │
│  │  + Testing   │      │   + Tools    │      │   + Caching      │      │
│  │              │      │              │      │                  │      │
│  └──────────────┘      └──────────────┘      └──────────────────┘      │
│                                                                         │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         │  Guardrails Enforced
                         │  ─────────────────────►
                         │  order_service.submit_order()
                         │
┌────────────────────────▼────────────────────────────────────────────────┐
│                         THREAD 2 (IBKR Sync Thread)                       │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────┐         │
│  │                                                    ┌─────────▼──────┐ │
│  │      IBKR Service (EClient + EWrapper)             │              │ │
│  │      • Blocking API callbacks                      │  Order       │ │
│  │      • Position tracking                           │  Execution   │ │
│  │      • Account updates                             │              │ │
│  │                                                    └──────────────┘ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                         │
                         │ Position tracking
                         ▼
                   data/positions.json
                   data/cache/*.json  (Tiingo data cache)
```

**Threading Design:**

1. **Thread 1** runs the async event loop with all non-blocking services
2. **Thread 2** handles IBKR's blocking API in isolation
3. **Queues** enable thread-safe communication between threads
4. **Single entry point**: All orders go through `order_service.submit_order()`

### Dependency Layers (Prevents Circular Imports)

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Pure Data (no project dependencies)              │
├─────────────────────────────────────────────────────────────┤
│  • models.py         → TradeSignal, LogMessage             │
│  • guardrails.py     → Validation rules (reads .env only)  │
└────────────────────────────┬────────────────────────────────┘
                             │ imports
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 2: State Management (imports Layer 1)               │
├─────────────────────────────────────────────────────────────┤
│  • context.py        → Queues, shared state, threading     │
└────────────────────────────┬────────────────────────────────┘
                             │ imports
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 3: Business Logic (imports Layers 1 & 2)            │
├─────────────────────────────────────────────────────────────┤
│  • order_service.py  → submit_order(), check_slippage()    │
│  • strategies/*.py   → Trading logic                       │
│  • services/ibkr.py  → Broker integration                  │
│  • services/*.py     → All other services                  │
└─────────────────────────────────────────────────────────────┘
```

### TiingoService Architecture (3-Layer Design)

The Tiingo data service uses a clean 3-layer architecture for optimal performance and code reusability:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Raw API Calls (Private)                          │
├─────────────────────────────────────────────────────────────┤
│  • _fetch_daily_api()     → Daily OHLC from Tiingo         │
│  • _fetch_intraday_api()  → Intraday OHLC from Tiingo IEX  │
│  • _apply_rate_limit()    → Random delay (1.2-2.0s)        │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 2: Cache Logic (Private)                            │
├─────────────────────────────────────────────────────────────┤
│  • _fetch_with_cache()    → Generic two-call strategy      │
│  • _filter_to_market_hours() → NYSE calendar filtering     │
│  • _get_cache_path()      → Cache file naming              │
│  • _load_from_cache()     → Read cached data               │
│  • _save_to_cache()       → Write cached data              │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  LAYER 3: Public API                                        │
├─────────────────────────────────────────────────────────────┤
│  • get_daily_ohlc(use_cache=True)                          │
│  • get_intraday_ohlc(use_cache=True)                       │
│  • get_current_price() - no cache (real-time)              │
│  • get_closes(), get_intraday_closes()                     │
└─────────────────────────────────────────────────────────────┘
```

**Caching Strategy (Two-Call Approach):**

Every data fetch is split into two API calls for optimal performance:

1. **Call 1 - Historical Data** (days ago → yesterday):
   - Cached permanently in `data/cache/{symbol}_{interval}_{start}_{end}.json`
   - No expiry - historical data doesn't change
   - Example: `QQQ_1min_20241213_20241217.json`

2. **Call 2 - Today's Data**:
   - Always fresh, never cached
   - Small dataset (1 day) = fast API response

3. **Merge & Filter**:
   - Combine both results
   - Filter to NYSE market hours (removes pre-market, after-hours, holidays, early closes)
   - Uses `pandas-market-calendars` for accurate NYSE schedule

**Performance:**
- **Cache hit**: ~1.5s (only today's API call + rate limit delay)
- **Cache miss**: ~3s (both API calls + rate limit delays)
- **Rate limiting**: 1.0s base + random 0.2-1.0s to prevent thundering herd

**Market Hours Filtering:**
- Automatically removes pre-market data (before 9:30 AM ET)
- Removes after-hours data (after 4:00 PM ET)
- Filters out holidays and early market closures
- Ensures strategies only see valid market hours data

**Key Design Benefits:**

1. **No circular imports** - Clean DAG structure ensures scalability
2. **models.py** contains pure data classes (no dependencies)
3. **context.py** provides thread-safe state and queues only
4. **order_service.py** enforces guardrails - cannot be bypassed
5. **Position Manager** persists positions to JSON for restart recovery
6. **Strategy-only trading** - no manual order tools for safety
7. **Intelligent caching** - Two-call strategy optimizes performance (historical cached, today fresh)
8. **Market hours filtering** - Automatic NYSE calendar filtering prevents bad data on holidays

---

## Project Structure

```
tele-ibkr-agent/
├── .env                     # Configuration (NEVER commit to git!)
├── requirements.txt         # Dependencies
├── main.py                  # Entry point & orchestration
├── run_backtest.py          # Bar-by-bar backtest entry point
├── run_backtest_vectorized.py # Fast vectorized backtest entry point
├── models.py                # Pure data classes (TradeSignal, LogMessage)
├── context.py               # Thread-safe shared state & queues
├── backtest/                # Backtest engine & components
├── data/                    # Runtime data
│   ├── positions.json       # Position tracking (auto-generated)
│   ├── backtest/            # Backtest data (OHLC, signals, results)
│   └── cache/               # Tiingo data cache (auto-generated)
│       └── *.json           # Cached OHLC data by symbol/interval/date
├── services/                # External service integrations
│   ├── ibkr.py              # IBKR connection (Thread 2)
│   ├── telegram.py          # Telegram Bot API (Thread 1)
│   ├── telegram_testing.py  # Testing buttons (toggle on/off)
│   ├── tiingo.py            # Market data (Thread 1)
│   ├── agent.py             # Gemini AI + tools (Thread 1)
│   ├── order_service.py     # Order submission with guardrails
│   ├── guardrails.py        # Safety validation (accounts, quantity, slippage)
│   └── position_manager.py  # JSON position persistence
├── strategies/              # Trading strategies
│   ├── _base.py             # Base class with position management
│   ├── __init__.py          # Strategy registry
│   └── ...                  # Individual strategies
└── tools/
    ├── admin_tools.py       # Account, position, tracking tools
    └── strategy_tools.py    # Strategy activation tools
```

---

## Guardrails & Safety

### Hardcoded Safety Limits

All orders are validated through `order_service.submit_order()` with **hardcoded** checks:

1. **Allowed Accounts** - Only whitelisted IBKR accounts can trade
2. **Max Order Quantity** - Maximum shares per trade enforced

**Configuration (`.env`):**
```bash
ALLOWED_ACCOUNTS=U18888888,U19999999  # Comma-separated whitelist
MAX_ORDER_QUANTITY=100                 # Max shares per order
```

**Example:**
```
User: "buy 1000 QQQ" (via strategy or testing)
Terminal: ========================================
          🚫 GUARDRAIL BLOCKED: Order quantity 1000
          exceeds maximum allowed (100).
          ========================================
Telegram: 🚫 Order BLOCKED by guardrail: Quantity 1000
          exceeds maximum allowed (100 shares per trade).
```

### Cannot Be Bypassed

Guardrails are enforced in `order_service.submit_order()` - the **single entry point** for all orders:
- Strategies call `order_service.submit_order()`
- Testing buttons call `order_service.submit_order()`
- No other code path can submit orders to IBKR

---

## Position Tracking & Management

### Automatic Position Persistence

Positions are saved to `data/positions.json` when strategies open trades. This allows:
- **Position recovery** after bot restart
- **Take profit / Stop loss** tracking per position
- **Strategy attribution** - which strategy owns which position

### Startup Reconciliation

On bot startup:
```
1. IBKR positions loaded
2. JSON positions loaded
3. Compare & reconcile:
   - Position in JSON but not IBKR → Closed externally (log warning, remove from JSON)
   - Position in IBKR but not JSON → Manual trade (log info, ignore)
   - Position in both → Resume tracking
```

### JSON Structure

```json
{
  "positions": {
    "U18888888:QQQ": {
      "symbol": "QQQ",
      "account": "U18888888",
      "strategy_id": "1",
      "action": "LONG",
      "quantity": 10,
      "entry_price": 485.50,
      "entry_time": "2025-12-17T10:30:00Z",
      "take_profit": 495.00,
      "stop_loss": 480.00
    }
  },
  "last_updated": "2025-12-17T10:30:00Z"
}
```

### Strategy Position Management

Strategies can use position tracking methods from `BaseStrategy`:

```python
class MyStrategy(BaseStrategy):
    TAKE_PROFIT_PRICE = 500.00
    STOP_LOSS_PRICE = 480.00

    async def execute(self):
        price = await self.tiingo.get_current_price(self.symbol)

        # Entry with tracking
        if not self.is_tracked() and should_enter:
            self.open_long(entry_price=price)  # Saves to JSON

        # Management (check TP/SL)
        if self.is_tracked():
            if self.check_take_profit(price):
                self.close_position()  # Removes from JSON
            elif self.check_stop_loss(price):
                self.close_position()
```

**Available Methods:**
- `open_long(entry_price, take_profit, stop_loss)` - Enter LONG with tracking
- `open_short(entry_price, take_profit, stop_loss)` - Enter SHORT with tracking
- `close_position()` - Exit position and remove from JSON
- `get_tracked_position()` - Get position data from JSON
- `is_tracked()` - Check if position exists in JSON
- `update_stops(take_profit, stop_loss)` - Update TP/SL
- `check_take_profit(current_price)` - Check if TP should trigger
- `check_stop_loss(current_price)` - Check if SL should trigger

---

## Testing Mode

### Telegram Inline Keyboard Testing

For testing trades, use `/test` command which shows inline keyboard buttons.

**Enable/Disable:**
```python
# services/telegram.py (line 26)
ENABLE_TESTING_BUTTONS = True  # Set to False to disable
```

**Usage:**
```
User: /test
Bot:  🧪 Testing Mode
      Symbol: QQQ
      Quantity: 1 share
      Current Account: U18888888

      ⚠️ These buttons will place REAL orders!

      [📈 BUY 1 QQQ] [📉 SELL 1 QQQ]
            [❌ Close]
```

**Features:**
- Self-contained in `services/telegram_testing.py`
- Uses same `order_service.submit_order()` as strategies (guardrails enforced)
- Configuration: Edit `TEST_SYMBOL` and `TEST_QUANTITY` in file
- Toggle on/off without affecting main bot

---

## Usage

### Telegram Commands (Natural Language)

All commands are **natural language** - just chat with the bot.

| Say | Action |
|-----|--------|
| **"what's the price of SPY?"** | Get current price from Tiingo |
| **"show my positions"** | View IBKR positions (cached) |
| **"show tracked positions"** | View tracked positions from JSON |
| **"show my balance"** | View account info for all accounts |
| **"refresh balances"** | Fetch fresh account data from IBKR |
| **"refresh positions"** | Fetch fresh position data |
| **"switch account to U18888888"** | Change active trading account |
| **"clear position records"** | Clear all JSON tracking data |
| **"list strategies"** | Show available strategies |
| **"apply strategy 1 to QQQ"** | Activate strategy (requires confirmation) |
| **"stop strategy for QQQ"** | Deactivate running strategy |

**Testing (if enabled):**
| Say | Action |
|-----|--------|
| **"/test"** | Show testing keyboard with buy/sell buttons |

### Important Notes

- ❌ **Manual trading removed** - No `buy`/`sell` commands (strategy-only for safety)
- ✅ **All orders** go through guardrails (cannot be bypassed)
- ✅ **Position tracking** persists across restarts
- ✅ **Testing mode** optional (easy toggle)
- ✅ **Market data caching** - Tiingo data automatically cached (historical permanent, today always fresh)
- ℹ️ **IBKR data refresh** - Use "refresh balances/positions" for fresh IBKR account data (separate from market data cache)

---

## Configuration (.env)

```bash
# APIs
TIINGO_API_KEY=your_tiingo_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash

# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # 7497 = paper trading, 7496 = live trading
IBKR_CLIENT_ID=1

# === GUARDRAILS (HARDCODED SAFETY LIMITS) ===
ALLOWED_ACCOUNTS=U18888888,U19999999  # Whitelist (comma-separated)
MAX_ORDER_QUANTITY=100                 # Max shares per trade
```

**Security Note**: Never commit `.env` to version control. Add it to `.gitignore` immediately.

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd tele-ibkr-agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and guardrail settings
```

## Design Philosophy

### 1. **Security First**
- Hardcoded guardrails prevent oversized orders and unauthorized accounts
- Strategy-only trading eliminates manual order mistakes
- All trades go through a single validated entry point

### 2. **Position Persistence**
- JSON tracking survives bot restarts
- Reconciliation on startup ensures data integrity
- Take profit / Stop loss managed per position

### 3. **Component Independence**
Swap any component (broker, AI, data) without touching others. Clear interfaces via `context.py`.

### 4. **Minimal Dependencies**
Only industry-standard libraries. No frameworks that can break with upgrades.

### 5. **Native APIs**
Direct HTTP calls to Telegram Bot API, Tiingo, etc. No wrappers or middlemen.

### 6. **Thread Safety**
Proper locking and message passing between async and sync threads. No race conditions.

### 7. **Natural Language First**
No slash commands to memorize (except optional `/test`). Just chat naturally with the AI.

### 8. **Testing Flexibility**
Optional testing mode with inline keyboards. Easy toggle on/off.

### 9. **Smart Data Management**
- Two-call caching strategy optimizes API usage (historical cached, today fresh)
- Automatic market hours filtering ensures data quality
- Rate limiting prevents thundering herd when multiple strategies run in parallel

---

## Security Considerations

- **Keep `.env` private**: Contains all API keys
- **Bot token security**: Anyone with token can message your bot
- **First user = admin**: Auto-assigned on first message
- **Guardrails**: Cannot be bypassed - enforced at execution level
- **Strategy-only**: Manual trading disabled by design
- **Position tracking**: JSON file contains trading data - protect it
- **Network security**: TWS/IB Gateway should run locally or on secure network

---

## Known Limitations

- **Natural language only**: No slash commands (except optional `/test`)
- **Single admin**: Only the first user gets notifications
- **Strategy-only trading**: Manual buy/sell commands removed for safety
- **Market hours**: Doesn't check if markets are open (data is filtered to market hours)
- **Full backtesting**: Supports both bar-by-bar and vectorized backtesting
- **No partial fills**: Assumes full order execution

---

## License

MIT License - feel free to modify and distribute.

---

## Disclaimer

**IMPORTANT**: This is trading software. Use at your own risk. Always test thoroughly with paper trading before using real money. The authors are not responsible for any losses incurred while using this software.

**API keys**: You are responsible for securing your API keys and monitoring for unauthorized access. Rotate keys immediately if compromised.

**Guardrails**: While guardrails provide safety limits, they do not guarantee profitable trading or prevent all errors. Always monitor your bot's activity.

---

**Version**: 2.1
**Last Updated**: 2025-12-18
**README Status**: ✅ Verified accurate with code (includes Tiingo caching & market hours filtering)
