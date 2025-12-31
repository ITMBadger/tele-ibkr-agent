# Tele-IBKR Agent

A professional-grade trading bot system bridging Telegram, Gemini AI, and multiple brokers (IBKR, Hyperliquid). It features a sophisticated multi-timeframe strategy engine, dual-engine backtesting (Vectorized & Bar-by-Bar), and modular market data providers.

## Key Features
- **Natural Language Interface**: Chat via Telegram + Gemini AI to execute trades, check positions, and manage strategies.
- **Multi-Broker Support**: Unified interface for **Interactive Brokers (IBKR)** and **Hyperliquid (Crypto Perpetual Perp)**.
- **Advanced MTF Strategy Engine**: Native support for Multi-Timeframe (1m, 5m, 15m, 30m) strategies with Heikin-Ashi, Bollinger Bands, and Stochastic indicators.
- **Dual Backtest Engines**: 
  - **Vectorized Engine**: Ultra-fast performance for testing years of data in seconds.
  - **Multiprocessing Engine**: Realistic bar-by-bar simulation for complex execution and exit logic.
- **Modular Data Layer**: Abstracted market data providers (Tiingo, Hyperliquid) with permanent historical caching and NYSE/Crypto timezone handling.
- **Hardcoded Guardrails**: Execution-level safety checks including account whitelisting, max order sizing, and slippage protection.
- **Position Persistence**: Local tracking in `positions.json` with automatic startup reconciliation against live broker accounts.

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                  THREAD 1: Async Event Loop                  │
│  (Telegram, Gemini AI, Strategy Engine, Data Providers)      │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
       Order Requests (Queues)         Market Data (Async)
                │                               │
┌───────────────▼───────────────────────────────▼──────────────┐
│                  THREAD 2: Broker Sync Thread                │
│  (IBKR EClient / Hyperliquid API, Account/Pos Management)    │
└───────────────────────────────┬──────────────────────────────┘
                                │
                        Execution & PnL
                                │
                    ┌───────────▼───────────┐
                    │      EXCHANGE         │
                    │  (IBKR / Hyperliquid) │
                    └───────────────────────┘
```

- **Thread 1**: Handles all I/O bound tasks and strategy logic. It uses a clock-aligned loop to ensure consistent strategy execution.
- **Thread 2**: Isolates blocking broker API calls to prevent event loop lag.
- **Safety Gate**: All orders must pass through `order_service.submit_order()` where guardrails are strictly enforced before reaching Thread 2.

## Design Principles
1. **Abstraction**: Brokers and Data Providers follow strict `Base` interfaces, allowing for easy expansion.
2. **Safety Shift**: No repainting. Signals are generated on bar $N-1$ and executed at the start of bar $N$.
3. **Time Centralization**: All timestamps are unified to Eastern Time (ET) using `time_centralize_utils.py` for cross-broker consistency.
4. **Decoupled Backtesting**: The backtest suite in `backtest/` is decoupled from live execution but shares the same `compute_signals` logic.

## Project Structure
```text
tele-ibkr-agent/
├── main.py                  # Entry point & modular service orchestration
├── context.py               # Thread-safe global state, queues, and locks
├── models.py                # Shared data models (TradeSignal, OHLCBar, etc.)
├── backtest/                # Sophisticated Backtest Suite
│   ├── engine_vectorized.py # Fast NumPy/Pandas signal generation
│   ├── engine_multiprocessing.py # Bar-by-bar execution simulator
│   ├── config.py            # Centralized backtest parameters
│   └── adapters/            # Data adapters for various formats
├── services/                # Core Integrations (Thread 1 & 2)
│   ├── broker_base.py       # Abstract Broker interface
│   ├── ibkr.py              # Interactive Brokers implementation
│   ├── hyperliquid.py       # Hyperliquid (Crypto) implementation
│   ├── market_data.py       # Data provider abstraction (Tiingo/Hyper)
│   ├── tiingo/              # Multi-layer caching data service
│   ├── order_service.py     # Execution gatekeeper with guardrails
│   ├── pos_manager.py       # JSON-based position tracking & recovery
│   └── telegram.py          # Gemini-powered natural language UI
├── strategies/              # Strategy Library
│   ├── _trading_mech.py     # Base class for live execution logic
│   ├── _template.py         # Multi-timeframe resample-merge template
│   └── strat_*.py           # Concrete strategy implementations
└── data/                    # Logs, permanent cache, and persistent state
```

## Quick Start
1. **Config**: `cp .env.example .env` and configure your API keys. Select broker via `BROKER=ibkr` or `BROKER=hyperliquid`.
2. **Install**: `pip install -r requirements.txt` (install `hyperliquid-python-sdk` if using Hyperliquid).
3. **Run**: `python main.py`
4. **Backtest**: `python -m backtest.engine_vectorized`

---
*Disclaimer: Trading involves risk. This software is for educational purposes only. Always use paper trading before live deployment.*
