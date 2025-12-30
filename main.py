# main.py - Application entry point and orchestration.
"""
This file:
1. Starts the broker service in a background thread (Thread 2)
2. Creates and injects dependencies between services
3. Runs the async event loop with Telegram bot (Thread 1)
4. Strategy loop starts on-demand when user activates a strategy

Supports multiple brokers (IBKR, Binance) via BROKER env var.

Run with: python main.py
"""

import os
import asyncio
import signal
import time
from pathlib import Path
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

import context
from services.broker_base import get_broker, list_brokers
from services.tiingo import TiingoService, TiingoCache
from services.market_data import TiingoDataProvider
from services.agent import GeminiAgent
from services.telegram import TelegramBot, ENABLE_TESTING_BUTTONS
from services.logger import terminal_logger, SignalLogger
from services.time_centralize_utils import get_et_now
from datetime import timedelta

# Conditional import for component testing
if ENABLE_TESTING_BUTTONS:
    from tools import component_test

# Auto-shutdown time (Eastern Time, 24h format)
AUTO_SHUTDOWN_HOUR = 16
AUTO_SHUTDOWN_MINUTE = 15  # 4:15 PM ET (15 min after market close)

# Cache settings
AUTO_CLEAR_CACHE = False  # Set to True to clear cache on startup


async def strategy_loop() -> None:
    """
    Background task that executes active strategies.
    """
    loop_tick = int(os.getenv("STRATEGY_LOOP_TICK", "10"))
    print(f"Strategy loop started (tick: {loop_tick}s, aligned to clock)")

    # Calculate next shutdown time (always the upcoming shutdown, not past)
    et_now = get_et_now()
    next_shutdown = et_now.replace(
        hour=AUTO_SHUTDOWN_HOUR, minute=AUTO_SHUTDOWN_MINUTE, second=0, microsecond=0
    )
    if et_now >= next_shutdown:
        # Already past today's shutdown time, schedule for tomorrow
        next_shutdown += timedelta(days=1)
    print(f"📅 Next auto-shutdown: {next_shutdown.strftime('%Y-%m-%d %H:%M')} ET")

    while not context.shutdown_event.is_set():
        try:
            # Auto-shutdown check
            et_now = get_et_now()
            if et_now >= next_shutdown:
                print(f"🛑 Auto-shutdown triggered at {et_now.strftime('%H:%M')} ET")
                context.log(f"🛑 Auto-shutdown at {et_now.strftime('%H:%M')} ET", "info")
                context.shutdown_event.set()
                break

            active = context.active_strategies.copy()
            tasks = []

            # Component test: add to parallel tasks if pending
            if ENABLE_TESTING_BUTTONS and component_test.should_execute_pending():
                print(f"   ⏰ [{get_et_now().strftime('%H:%M:%S')}] Dispatching: Component test...")
                
                async def run_comp_test():
                    result = await component_test.execute_pending_buy()
                    if result:
                        context.log(result, "trade")
                
                tasks.append(run_comp_test())

            # Collect strategy tasks
            for symbol, data in active.items():
                strategy = data.get("strategy")
                if strategy and strategy.should_run():
                    name = data.get("name", "Unknown")
                    ts = get_et_now().strftime('%H:%M:%S')
                    print(f"   🚀 [{ts}] Executing: {name} on {symbol}")
                    tasks.append(strategy.execute())

            # Run all tasks in parallel (strategies + component test)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Sleep until next aligned tick
            now = time.time()
            sleep_time = loop_tick - (now % loop_tick)
            if sleep_time < 0.5:
                sleep_time = loop_tick
            print(f"   ⏳ [{get_et_now().strftime('%H:%M:%S')}] Tick ({len(active)} active, sleep {sleep_time:.1f}s)")
            await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Strategy loop error: {e}")
            context.log(f"Strategy error: {e}", "error")
            await asyncio.sleep(loop_tick)

    print("Strategy loop stopped")


def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    print("\nShutting down...")
    context.shutdown_event.set()


def clear_cache():
    """Delete all files in the data/cache directory on startup."""
    cache = TiingoCache(CACHE_DIR)
    files = cache.list_files()

    if files:
        print(f"Clearing cache ({len(files)} files)...")
        count = cache.clear_all()
        print(f"  Cleared {count} files.")
    else:
        print("Cache is already clean.")


async def main():
    """Main application entry point."""
    # Start terminal logging FIRST (before any print)
    log_path = terminal_logger.start()

    # Get broker from env
    broker_name = os.getenv("BROKER", "ibkr").lower()
    context.active_broker = broker_name

    print("=" * 50)
    print(f"Telegram Trading Bot ({broker_name.upper()})")
    print("=" * 50)
    print(f"Terminal log: {log_path}")

    # Clear cache on startup (if enabled)
    if AUTO_CLEAR_CACHE:
        clear_cache()
    else:
        print("Cache: Preserved (AUTO_CLEAR_CACHE=False)")

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # 1. Import and start broker in background thread
    try:
        # Conditional import based on selected broker
        if broker_name == "ibkr":
            from services.ibkr import IBKRService  # noqa: F401
        elif broker_name == "hyperliquid":
            from services.hyperliquid import HyperliquidService  # noqa: F401
        else:
            print(f"Unknown broker: {broker_name}")
            print(f"   Available brokers: ibkr, hyperliquid")
            return

        BrokerClass = get_broker(broker_name)
        print(f"Starting {broker_name.upper()} service...")
        broker = BrokerClass()
        broker_thread = broker.start_thread()
        await asyncio.sleep(2)
    except ValueError as e:
        print(f"❌ {e}")
        print(f"   Available brokers: {', '.join(list_brokers())}")
        return
    except ImportError as e:
        print(f"Import error: {e}")
        if broker_name == "hyperliquid":
            print("   Install Hyperliquid support: pip install hyperliquid-python-sdk")
        return

    # 2. Create market data service (always Tiingo for consistent indicators/signals)
    # Both IBKR and Hyperliquid use Tiingo for OHLC data
    # Only difference: Hyperliquid needs crypto symbol translation (BTC → BTCUSD)
    print("Starting Tiingo service...")
    tiingo_service = TiingoService()
    if broker_name == "hyperliquid":
        data_provider = TiingoDataProvider(tiingo_service, crypto_mode=True)
    else:
        data_provider = TiingoDataProvider(tiingo_service)

    print("Starting Gemini agent...")
    agent = GeminiAgent(
        tiingo_service=data_provider,  # Pass data_provider (has symbol translation)
        execution_handler=broker.place_order
    )

    print("Starting Telegram bot...")
    telegram = TelegramBot(agent=agent)

    # 3. Start strategy loop (always runs, handles auto-shutdown)
    context.strategy_loop_task = asyncio.create_task(strategy_loop())

    # 4. Run
    print("Bot is running! Send /start to your Telegram bot.")
    print("Press Ctrl+C to stop.")
    print("=" * 50)

    try:
        await telegram.start()
    except asyncio.CancelledError:
        pass
    finally:
        print("Cleaning up...")
        context.shutdown_event.set()
        await data_provider.close()
        await telegram.stop()
        broker_thread.join(timeout=5)

        # Clean up loggers
        SignalLogger.clear_all()
        print("Shutdown complete.")

        # Stop terminal logging LAST
        terminal_logger.stop()


if __name__ == "__main__":
    asyncio.run(main())
