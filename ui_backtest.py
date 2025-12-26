#!/usr/bin/env python3
"""
Strategy Backtest Dashboard - NiceGUI Application

Interactive UI for running ha_mtf_stoch_ema backtests with configurable signals.

Usage:
    python ui_backtest.py

Then open http://localhost:8080 in your browser.
"""

import asyncio
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from nicegui import ui, run

# Load environment variables
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")

from ui.state import AppState
from ui.runner import run_backtest


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
        ui.label("Strategy: ha_mtf_stoch_ema | Engine: backtest/")


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
        title="Strategy Backtest",
        port=8080,
        reload=False,
        show=True,
    )
