# backtest/engine_vectorized.py
"""
Vectorized Backtest Engine - Fast backtesting using compute_signals().

This engine:
1. Loads historical data once
2. Computes ALL signals in a single vectorized call
3. Runs trade simulation using the shared SimulatedBroker
4. Calculates and displays performance metrics

Much faster than bar-by-bar backtest for simple strategies.

Standalone usage:
    python backtest/engine_vectorized.py
"""

import importlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Type, Optional

from .config import BacktestConfig
from .providers.loader import load_ohlc
from .providers.downloader import ensure_historical_data
from .execution.simulator import SimulatedBroker
from .metrics.calculator import BacktestResult, build_backtest_result
from .metrics.report import save_results, print_summary
from .visualization import generate_html_dashboard
from services.logger import SignalLogger
from services.tiingo import ET_DATETIME_FORMAT


class VectorizedBacktestEngine:
    """
    Fast backtest engine using vectorized signal computation.

    Flow:
    1. Load OHLC data once
    2. Call strategy.compute_signals(df) once (vectorized)
    3. Convert signal column to signal dicts
    4. Pass to SimulatedBroker for TP/SL simulation
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._strategy_class: Optional[Type] = None
        self._signals: Dict[str, List[Dict[str, Any]]] = {}
        self._results: Dict[str, Any] = {}
        self._result: Optional[BacktestResult] = None
        self._df_with_signals: Dict[str, pd.DataFrame] = {}  # Full 1min data with signals
        self._results_dir: Optional[Path] = None

        # Ensure data directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create data directories if they don't exist."""
        for dir_path in [
            self.config.ohlc_dir,
            self.config.results_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _get_strategy_class(self) -> Type:
        """Import and return strategy class."""
        if self._strategy_class is not None:
            return self._strategy_class

        strategy_name = self.config.strategy

        # Map common names to modules
        strategy_map = {
            "ema_only_long": ("strategies.ema_only_long", "EMAOnlyLong"),
            "ha_mtf_stoch": ("strategies.ha_mtf_stoch", "HAMTFStoch"),
            "strat_multi_toggle": ("strategies.strat_multi_toggle", "StratMultiToggle"),
        }

        if strategy_name in strategy_map:
            module_name, class_name = strategy_map[strategy_name]
        else:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Vectorized backtest supports: {list(strategy_map.keys())}"
            )

        try:
            module = importlib.import_module(module_name)
            self._strategy_class = getattr(module, class_name)
            return self._strategy_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import strategy {strategy_name}: {e}")

    def ensure_data(self) -> Dict[str, Path]:
        """Ensure historical data exists for all symbols."""
        print("\n[Step 1] Checking historical data...")

        data_dir = Path(self.config.ohlc_dir)
        data_files = {}

        for symbol in self.config.symbols:
            file_path = ensure_historical_data(
                symbol=symbol,
                months_back=self.config.months_back,
                interval="1min",
                data_dir=data_dir,
            )
            data_files[symbol] = file_path

        return data_files

    def generate_signals_vectorized(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate signals using compute_signals() - single vectorized call.

        This is the key difference from bar-by-bar: we call compute_signals()
        once on the entire dataset and extract all signals at once.
        """
        print("\n[Step 2] Generating signals (VECTORIZED)...")

        strategy_class = self._get_strategy_class()
        all_signals = {}

        for symbol in self.config.symbols:
            print(f"  {symbol}: Loading data...")

            # Load full OHLC data
            df = load_ohlc(
                symbol=symbol,
                data_dir=self.config.ohlc_dir,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            print(f"  {symbol}: {len(df):,} bars loaded")

            # Compute signals in ONE vectorized call
            print(f"  {symbol}: Computing signals (vectorized)...")
            df_with_signals = strategy_class.compute_signals(df)

            # Store full dataframe for later CSV export
            self._df_with_signals[symbol] = df_with_signals.copy()

            # Extract BUY signals (signal == 1)
            buy_mask = df_with_signals["signal"] == 1
            buy_rows = df_with_signals[buy_mask]

            # Get strategy quantity
            qty = getattr(strategy_class, "QUANTITY", 10)

            # Auto-detect signal detail columns (dynamic for all strategies)
            # Exclude base OHLC columns, keep all signal-related columns
            base_cols = {"date", "open", "high", "low", "close", "volume", "signal"}
            detail_cols = [
                col for col in df_with_signals.columns
                if col not in base_cols
            ]

            # Convert to signal dicts with details
            signals = []
            for idx, row in buy_rows.iterrows():
                # Format timestamp (already naive ET from loader)
                ts = pd.to_datetime(row["date"])
                ts_str = ts.strftime(ET_DATETIME_FORMAT)

                signal_dict = {
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": qty,
                    "price": round(row["close"], 3),
                    "timestamp": ts_str,
                    "bar_index": idx,
                }

                # Add signal detail columns if they exist
                for col in detail_cols:
                    if col in row.index:
                        val = row[col]
                        if pd.isna(val):
                            signal_dict[col] = ""
                        elif isinstance(val, bool) or isinstance(val, (int, float)):
                            if isinstance(val, float):
                                signal_dict[col] = round(val, 3)
                            else:
                                signal_dict[col] = val
                        else:
                            signal_dict[col] = val

                signals.append(signal_dict)

            print(f"  {symbol}: {len(signals):,} BUY signals extracted")
            all_signals[symbol] = signals

        self._signals = all_signals
        return all_signals

    def run_simulation(
        self,
        signals: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> BacktestResult:
        """
        Run trade simulation using shared SimulatedBroker.

        This is identical to bar-by-bar backtest - same TP/SL logic.
        """
        print("\n[Step 3] Running simulation...")

        if signals is None:
            signals = self._signals

        if not signals:
            print("  No signals to simulate")
            # Return empty result
            empty_equity = pd.DataFrame({"Equity": [self.config.initial_capital], "Drawdown_Pct": [0.0]})
            return build_backtest_result(
                config=self.config,
                trades=[],
                equity_curve=empty_equity,
                starting_capital=self.config.initial_capital,
                symbol=",".join(self.config.symbols),
            )

        # Combine signals from all symbols
        all_signals = []
        for symbol_signals in signals.values():
            all_signals.extend(symbol_signals)

        print(f"  Total signals: {len(all_signals):,}")

        # Load OHLC for simulation
        symbol = self.config.symbols[0]
        ohlc_df = load_ohlc(
            symbol=symbol,
            data_dir=self.config.ohlc_dir,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        print(f"  OHLC data: {len(ohlc_df):,} bars")

        # Run simulation with shared SimulatedBroker (TP/SL from strategy class)
        strategy_class = self._get_strategy_class()
        broker = SimulatedBroker(
            initial_capital=self.config.initial_capital,
            slippage_pct=self.config.slippage_pct,
            commission_per_trade=self.config.commission_per_trade,
            strategy_class=strategy_class,
        )

        sim_results = broker.run_simulation(all_signals, ohlc_df)

        print(f"  Trades executed: {sim_results['total_trades']}")

        # Build BacktestResult with all statistics
        result = build_backtest_result(
            config=self.config,
            trades=sim_results["trades"],
            equity_curve=sim_results["equity_curve"],
            starting_capital=self.config.initial_capital,
            symbol=",".join(self.config.symbols),
            timeframe="1min",
        )

        self._results = sim_results
        self._result = result
        return result

    def save_results(
        self,
        results_dir: Optional[Path | str] = None,
    ) -> Path:
        """Save all results to disk including HTML dashboard."""
        if results_dir is None:
            results_dir = self._results_dir

        if results_dir is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(self.config.results_dir) / f"vectorized_{run_id}"

        results_dir = Path(results_dir)

        # Gather all signals
        all_signals = []
        for symbol_signals in self._signals.values():
            all_signals.extend(symbol_signals)

        # Save results
        saved_dir = save_results(
            results_dir=results_dir,
            config=self.config,
            result=self._result,
            signals=all_signals,
        )

        # Save complete 1min DataFrames with all signal details using SignalLogger
        for symbol, df in self._df_with_signals.items():
            logger = SignalLogger.get_or_create(symbol, self.config.strategy)
            logger.log_dataframe(df)

        # Generate HTML dashboard
        if self._result is not None:
            # Build signal config string from strategy class
            signal_config = ""
            strategy_class = self._get_strategy_class()
            if hasattr(strategy_class, "get_signal_config_string"):
                try:
                    signal_config = strategy_class.get_signal_config_string()
                except Exception:
                    pass

            generate_html_dashboard(
                result=self._result,
                output_dir=str(results_dir),
                prefix=f"vectorized_{self.config.strategy}",
                verbose=True,
                signal_config=signal_config,
                hide_signals=self.config.hide_signals,
            )

        return saved_dir

    def generate_dashboard_only(self, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Generate dashboard HTML without saving other results."""
        if self._result is None:
            return None

        if output_dir is None:
            # Use temp directory or results dir
            import tempfile
            if self.config.save_results:
                output_dir = self._results_dir
            else:
                # Use a temp location that persists for the session
                output_dir = Path(tempfile.gettempdir()) / "backtest_preview"
                output_dir.mkdir(exist_ok=True)

        output_dir = Path(output_dir)

        # Build signal config string from strategy class
        signal_config = ""
        strategy_class = self._get_strategy_class()
        if hasattr(strategy_class, "get_signal_config_string"):
            try:
                signal_config = strategy_class.get_signal_config_string()
            except Exception:
                pass

        # Generate dashboard
        dashboard_path = generate_html_dashboard(
            result=self._result,
            output_dir=str(output_dir),
            prefix=f"vectorized_{self.config.strategy}",
            verbose=not self.config.save_results,  # Only show message if not saving
            signal_config=signal_config,
            hide_signals=self.config.hide_signals,
        )

        return Path(dashboard_path) if dashboard_path else None

    def run(self) -> BacktestResult:
        """Run complete vectorized backtest pipeline."""
        # Set up run_id and SignalLogger mode
        run_id = f"vectorized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._results_dir = Path(self.config.results_dir) / run_id
        SignalLogger.set_mode("backtest", run_id)

        try:
            # Step 1: Ensure data
            self.ensure_data()

            # Step 2: Generate signals (VECTORIZED)
            self.generate_signals_vectorized()

            # Step 3: Run simulation (uses shared SimulatedBroker)
            result = self.run_simulation()

            # Step 4: Print summary
            print("\n[Step 4] Performance Summary")
            print_summary(result)

            # Step 5: Save results (if enabled)
            if self.config.save_results:
                self.save_results()
                print("\n[Step 5] Results saved")
            else:
                print("\n[Step 5] Skipping file save (save_results=False)")
                # Still generate dashboard for preview
                dashboard_path = self.generate_dashboard_only()
                if dashboard_path:
                    self._result._dashboard_path = dashboard_path  # Store for UI access
                    print(f"  Dashboard preview: {dashboard_path}")

            return result

        finally:
            # Reset SignalLogger to live mode
            SignalLogger.reset()


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path for imports
    PROJECT_ROOT = Path(__file__).parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    # Default config for standalone run
    config = BacktestConfig(
        symbols=["QQQ"],
        strategy="strat_multi_toggle",
        months_back=12,
        initial_capital=100_000.0,
        slippage_pct=0.0002,
        commission_per_trade=0.5,
    )

    print("=" * 60)
    print("VECTORIZED BACKTEST ENGINE")
    print("=" * 60)
    print(f"Symbols:       {', '.join(config.symbols)}")
    print(f"Strategy:      {config.strategy}")
    print(f"Period:        {config.start_date} to {config.end_date}")
    print(f"Capital:       ${config.initial_capital:,.0f}")
    print("=" * 60)

    engine = VectorizedBacktestEngine(config)
    result = engine.run()

    print("\nVectorized backtest complete!")
