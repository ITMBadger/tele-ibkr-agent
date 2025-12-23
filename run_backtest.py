#!/usr/bin/env python3
"""
Backtest Runner - 1-click entry point for backtesting strategies.

This script:
1. Automatically downloads missing historical data
2. Generates signals using your strategy (with caching)
3. Runs trade simulation
4. Calculates and displays performance metrics
5. Saves all results

Usage:
    python run_backtest.py

Configuration:
    Edit the CONFIG section below to customize your backtest.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine

# Project paths
PROJECT_ROOT = Path(__file__).parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================
# CONFIGURATION - Edit these values
# ============================================================

CONFIG = BacktestConfig(
    # ---- Symbols to backtest ----
    symbols=["QQQ"],

    # ---- Strategy to test ----
    # Available: "ha_mtf_stoch", "ema_200_long", "ema_100_conservative",
    #            "ema_50_aggressive", "ema_20_scalp", "rsi_bounce"
    strategy="ha_mtf_stoch",

    # ---- Date range ----
    start_date="2025-01-01",
    end_date="2026-01-01",

    # ---- Chunking parameters ----
    chunk_size=0,        # 0 = auto-calculate based on CPU cores
    warmup_size=6_000,   # Warmup bars for indicator convergence

    # ---- Execution simulation ----
    initial_capital=100_000.0,
    slippage_pct=0.001,        # 0.1% slippage
    commission_per_trade=0.5,  # $0.5 per trade

    # ---- Take Profit / Stop Loss ----
    take_profit_pct=2.0,       # +2% take profit (set None to disable)
    stop_loss_pct=1.0,         # -1% stop loss (set None to disable)

    # ---- Parallelism ----
    max_workers=0,  # 0 = auto-use Physical Cores - 1

    # ---- Caching ----
    use_signal_cache=True,
    force_regenerate=False,  # Set True to ignore cache and regenerate signals
)


# ============================================================
# MAIN - Just run this file
# ============================================================

def main():
    """Run the backtest."""
    print("=" * 60)
    print("BACKTEST ENGINE")
    print("=" * 60)
    print(f"Symbols:       {', '.join(CONFIG.symbols)}")
    print(f"Strategy:      {CONFIG.strategy}")
    print(f"Period:        {CONFIG.start_date} to {CONFIG.end_date}")
    print(f"Capital:       ${CONFIG.initial_capital:,.0f}")
    print(f"TP/SL:         {CONFIG.take_profit_pct or 'None'}% / {CONFIG.stop_loss_pct or 'None'}%")
    print(f"Chunk size:    {CONFIG.chunk_size:,} bars")
    print(f"Warmup:        {CONFIG.warmup_size:,} bars")
    print(f"Max workers:   {CONFIG.max_workers}")
    print("=" * 60)

    # Create engine and run backtest
    engine = BacktestEngine(CONFIG)
    metrics = engine.run()

    # Final summary already printed by engine
    print("\nBacktest complete!")

    return metrics


if __name__ == "__main__":
    main()
