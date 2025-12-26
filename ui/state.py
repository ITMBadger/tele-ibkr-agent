# ui/state.py
"""Reactive state management for backtest UI."""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class AppState:
    """
    Central state for the backtest UI.

    All UI components read/write to this state.
    """

    # Backtest mode
    mode: str = "vectorized"  # "vectorized" or "multiprocessing"

    # Signal toggles (mirrors SignalType in ha_mtf_stoch_ema.py)
    signals: dict = field(default_factory=lambda: {
        "STOCH_RISING": True,
        "HA_TWO_GREEN": True,
        "BB_TOUCH": True,
        "ABOVE_EMA_200": False,
    })

    # Backtest config
    symbol: str = "QQQ"
    months_back: int = 12
    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0002
    commission_per_trade: float = 0.5

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
