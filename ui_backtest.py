#!/usr/bin/env python3
"""
Strategy Backtest Dashboard - NiceGUI Application

Interactive UI for running strategy backtests with configurable signals.

Usage:
    python ui_backtest.py

Then open http://localhost:8080 in your browser.
"""

import asyncio
import traceback
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from nicegui import ui, run

# Load environment variables
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# CONFIGURATION - Edit these defaults as needed
# ============================================================

# Strategy settings
DEFAULT_STRATEGY = "strat_multi_toggle"
DEFAULT_SYMBOL = "QQQ"
DEFAULT_MONTHS_BACK = 12

# Capital & costs
DEFAULT_INITIAL_CAPITAL = 100_000.0
DEFAULT_SLIPPAGE_PCT = 0.0002  # 0.02%
DEFAULT_COMMISSION_PER_TRADE = 0.5

# Backtest mode
DEFAULT_MODE = "vectorized"  # "vectorized" or "multiprocessing"

# Default signal configuration (mirrors SignalType in strat_multi_toggle.py)
DEFAULT_SIGNALS = {
    "STOCH_RISING": True,      # 30m Stochastic %D rising
    "HA_TWO_GREEN": True,       # Last 2 completed 5m HA bars are green
    "BB_TOUCH": True,           # HA low touched lower Bollinger Band
    "ABOVE_EMA_200": False,     # Price above 200 EMA
}

# UI settings
UI_PORT = 8080
UI_TITLE = "Strategy Backtest"


# ============================================================
# STATE MANAGEMENT
# ============================================================

@dataclass
class AppState:
    """Central state for the backtest UI. All UI components read/write to this state."""

    # Backtest mode
    mode: str = DEFAULT_MODE

    # Signal toggles
    signals: dict = field(default_factory=lambda: DEFAULT_SIGNALS.copy())

    # Backtest config
    strategy: str = DEFAULT_STRATEGY
    symbol: str = DEFAULT_SYMBOL
    months_back: int = DEFAULT_MONTHS_BACK
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT
    commission_per_trade: float = DEFAULT_COMMISSION_PER_TRADE

    # Run state
    running: bool = False
    progress: str = ""
    error: str = ""

    # Results
    result_path: Optional[Path] = None
    dashboard_path: Optional[Path] = None

    def get_enabled_signals(self) -> set:
        """Get set of enabled signal names."""
        return {name for name, enabled in self.signals.items() if enabled}

    def reset_run_state(self):
        """Reset state before a new run."""
        self.running = False
        self.progress = ""
        self.error = ""
        self.result_path = None
        self.dashboard_path = None


# ============================================================
# BACKTEST EXECUTION
# ============================================================

def run_backtest(
    state: AppState,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Run backtest with current UI state settings.

    Dynamically patches StratMultiToggle.ENABLED_BUY_SIGNALS before run.

    Args:
        state: Current UI state with all settings
        on_progress: Optional callback for progress updates

    Returns:
        Path to results directory (contains dashboard HTML)

    Raises:
        Exception: If backtest fails
    """
    from backtest import BacktestConfig, BacktestEngine, VectorizedBacktestEngine
    from strategies.strat_multi_toggle import StratMultiToggle, SignalType

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
    StratMultiToggle.ENABLED_BUY_SIGNALS = enabled_signals
    log(f"  Enabled: {[s.name for s in enabled_signals]}")

    # Step 2: Create BacktestConfig
    log("Creating config...")
    config = BacktestConfig(
        symbols=[state.symbol],
        strategy=state.strategy,
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


# ============================================================
# GLOBAL STATE
# ============================================================

state = AppState()


# ============================================================
# UI COMPONENTS
# ============================================================

def create_header():
    """Create page header."""
    with ui.row().classes("w-full items-center justify-between mb-6"):
        ui.label("HA + MTF Stoch + EMA Backtest").classes(
            "text-2xl font-bold text-gray-800"
        )
        ui.label("Strategy Dashboard").classes("text-gray-500")


def create_mode_selector():
    """Create backtest mode toggle."""
    with ui.card().classes("w-full mb-4"):
        ui.label("Backtest Mode").classes("text-lg font-semibold mb-2")
        ui.toggle(
            ["vectorized", "multiprocessing"],
            value=state.mode,
            on_change=lambda e: setattr(state, "mode", e.value),
        ).classes("w-full")
        with ui.row().classes("text-sm text-gray-500 mt-2"):
            ui.label("vectorized = fast (single pass) | multiprocessing = accurate (bar-by-bar)")


def create_signal_toggles():
    """Create signal toggle switches in two columns."""
    with ui.card().classes("w-full mb-4"):
        ui.label("Signal Configuration").classes("text-lg font-semibold mb-2")
        ui.label("Toggle which signals to include in the strategy:").classes(
            "text-sm text-gray-500 mb-3"
        )

        signal_descriptions = {
            "STOCH_RISING": "30m Stochastic %D rising (momentum)",
            "HA_TWO_GREEN": "Last 2 completed 5m HA bars are green",
            "BB_TOUCH": "HA low touched lower Bollinger Band",
            "ABOVE_EMA_200": "Price above 200 EMA (trend filter)",
        }

        with ui.grid(columns=2).classes("w-full gap-4"):
            for sig_name, description in signal_descriptions.items():
                with ui.card().classes("p-3"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.column().classes("gap-0"):
                            ui.label(sig_name).classes("font-mono font-medium text-sm")
                            ui.label(description).classes("text-xs text-gray-500")
                        ui.switch(
                            value=state.signals.get(sig_name, False),
                            on_change=lambda e, name=sig_name: state.signals.update(
                                {name: e.value}
                            ),
                        )


def create_config_panel():
    """Create configuration inputs."""
    with ui.card().classes("w-full mb-4"):
        ui.label("Backtest Configuration").classes("text-lg font-semibold mb-2")

        with ui.grid(columns=2).classes("w-full gap-4"):
            # Symbol
            ui.input(
                label="Symbol",
                value=state.symbol,
                on_change=lambda e: setattr(state, "symbol", e.value.upper()),
            ).classes("w-full")

            # Months back
            ui.number(
                label="Months of Data",
                value=state.months_back,
                min=1,
                max=60,
                step=1,
                on_change=lambda e: setattr(state, "months_back", int(e.value)),
            ).classes("w-full")

            # Initial capital
            ui.number(
                label="Initial Capital ($)",
                value=state.initial_capital,
                min=1000,
                step=10000,
                format="%.0f",
                on_change=lambda e: setattr(state, "initial_capital", float(e.value)),
            ).classes("w-full")

            # Slippage
            ui.number(
                label="Slippage (%)",
                value=state.slippage_pct * 100,
                min=0,
                max=1,
                step=0.01,
                format="%.3f",
                on_change=lambda e: setattr(state, "slippage_pct", float(e.value) / 100),
            ).classes("w-full")


def create_run_section():
    """Create run button and progress section."""
    progress_label = ui.label("").classes("text-sm text-gray-600 font-mono whitespace-pre-wrap")
    progress_log = ui.log(max_lines=20).classes("w-full h-48 mt-2 hidden")

    async def do_run():
        """Execute backtest."""
        state.reset_run_state()
        state.running = True
        run_btn.disable()
        progress_log.classes(remove="hidden")
        progress_log.clear()

        def on_progress(msg: str):
            progress_log.push(msg)

        try:
            # Run backtest in background thread to not block UI
            await run.io_bound(
                run_backtest,
                state,
                on_progress,
            )

            state.running = False
            progress_label.set_text("Backtest complete!")

            # Show results button
            if state.dashboard_path and state.dashboard_path.exists():
                ui.notify("Backtest complete! Opening dashboard...", type="positive")
                webbrowser.open(f"file://{state.dashboard_path.absolute()}")
            else:
                ui.notify("Backtest complete!", type="positive")

        except Exception as e:
            state.running = False
            state.error = str(e)
            progress_label.set_text(f"Error: {e}")
            ui.notify(f"Backtest failed: {e}", type="negative")

        finally:
            run_btn.enable()

    with ui.card().classes("w-full"):
        ui.label("Run Backtest").classes("text-lg font-semibold mb-2")

        with ui.row().classes("w-full items-center gap-4"):
            run_btn = ui.button(
                "Generate Dashboard",
                on_click=do_run,
                color="primary",
            ).classes("px-8 py-2")

            open_btn = ui.button(
                "Open Last Dashboard",
                on_click=lambda: webbrowser.open(f"file://{state.dashboard_path.absolute()}")
                if state.dashboard_path and state.dashboard_path.exists()
                else ui.notify("No dashboard available", type="warning"),
                color="secondary",
            ).classes("px-4 py-2")

        progress_label
        progress_log


def create_footer():
    """Create footer with info."""
    with ui.row().classes("w-full justify-center mt-8 text-sm text-gray-400"):
        ui.label(f"Strategy: {DEFAULT_STRATEGY} | Engine: backtest/")


# ============================================================
# MAIN PAGE
# ============================================================

@ui.page("/")
def main_page():
    """Main dashboard page."""
    ui.colors(primary="#3b82f6")

    with ui.column().classes("w-full max-w-5xl mx-auto p-6"):
        create_header()
        create_mode_selector()
        create_signal_toggles()
        create_config_panel()
        create_run_section()
        create_footer()


# ============================================================
# RUN
# ============================================================

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title=UI_TITLE,
        port=UI_PORT,
        reload=False,
        show=True,
    )
