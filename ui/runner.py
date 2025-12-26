# ui/runner.py
"""Backtest execution wrapper for UI."""

from pathlib import Path
from typing import Callable, Optional
import traceback

from .state import AppState


def run_backtest(
    state: AppState,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Run backtest with current UI state settings.

    Dynamically patches HAMTFStochEMA.ENABLED_BUY_SIGNALS before run.

    Args:
        state: Current UI state with all settings
        on_progress: Optional callback for progress updates

    Returns:
        Path to results directory (contains dashboard HTML)

    Raises:
        Exception: If backtest fails
    """
    from backtest import BacktestConfig, BacktestEngine, VectorizedBacktestEngine
    from strategies.ha_mtf_stoch_ema import HAMTFStochEMA, SignalType

    def log(msg: str):
        if on_progress:
            on_progress(msg)
        print(msg)

    # Step 1: Patch ENABLED_BUY_SIGNALS based on UI toggles
    log("Configuring signals...")
    enabled_signals = set()
    for sig_name, enabled in state.signals.items():
        if enabled:
            try:
                enabled_signals.add(SignalType[sig_name])
            except KeyError:
                log(f"  Warning: Unknown signal {sig_name}")

    # Dynamically set the class attribute
    HAMTFStochEMA.ENABLED_BUY_SIGNALS = enabled_signals
    log(f"  Enabled: {[s.name for s in enabled_signals]}")

    # Step 2: Create BacktestConfig
    log("Creating config...")
    config = BacktestConfig(
        symbols=[state.symbol],
        strategy="ha_mtf_stoch_ema",
        months_back=state.months_back,
        initial_capital=state.initial_capital,
        slippage_pct=state.slippage_pct,
        commission_per_trade=state.commission_per_trade,
    )
    log(f"  Symbol: {state.symbol}")
    log(f"  Period: {config.start_date} to {config.end_date}")
    log(f"  Capital: ${state.initial_capital:,.0f}")

    # Step 3: Run appropriate engine
    log(f"Starting {state.mode} backtest...")

    if state.mode == "vectorized":
        engine = VectorizedBacktestEngine(config)
    else:
        engine = BacktestEngine(config)

    result = engine.run()

    # Step 4: Get results path
    results_dir = engine._results_dir
    log(f"Results saved to: {results_dir}")

    # Find dashboard HTML
    dashboard_files = list(results_dir.glob("*dashboard*.html"))
    if dashboard_files:
        state.dashboard_path = dashboard_files[0]
        log(f"Dashboard: {state.dashboard_path}")

    state.result_path = results_dir
    return results_dir


def run_backtest_async(
    state: AppState,
    on_progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[Path], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
):
    """
    Run backtest and handle callbacks (for use with NiceGUI's run.cpu_bound).

    This is a wrapper that catches exceptions and calls appropriate callbacks.
    """
    try:
        result_path = run_backtest(state, on_progress)
        if on_complete:
            on_complete(result_path)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if on_error:
            on_error(error_msg)
        raise
