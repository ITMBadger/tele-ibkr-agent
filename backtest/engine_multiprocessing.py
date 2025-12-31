# backtest/engine_multiprocessing.py
"""
Multiprocessing Backtest Engine - Bar-by-bar backtesting with parallel chunk processing.

This engine:
1. Loads historical data once
2. Splits data into chunks for parallel processing
3. Generates signals bar-by-bar (simulates live trading)
4. Runs trade simulation using SimulatedBroker
5. Calculates and displays performance metrics

Standalone usage:
    python backtest/engine_multiprocessing.py
"""

from pathlib import Path
from typing import Dict, List, Any, Type, Optional
import importlib
import pandas as pd
import psutil

from services.time_centralize_utils import get_et_now, get_et_timestamp

from .config import BacktestConfig
from .providers.loader import load_ohlc, get_bar_count, get_ohlc_file_path
from .providers.chunker import create_chunks, get_chunk_summary
from .providers.downloader import ensure_historical_data
from .signals.worker import run_parallel_signal_generation, run_sequential_signal_generation
from .signals.cache import SignalCache, get_cache_key
from .execution.simulator import SimulatedBroker
from .metrics.calculator import BacktestResult, build_backtest_result
from .metrics.report import save_results, print_summary
from .visualization import generate_html_dashboard
from services.logger import SignalLogger


class BacktestEngine:
    """
    Main orchestrator for running backtests.

    Coordinates:
    1. Data loading and preparation
    2. Signal generation (parallel, with caching)
    3. Trade simulation
    4. Metrics calculation
    5. Dashboard generation
    """

    def __init__(self, config: BacktestConfig):
        """Initialize backtest engine."""
        self.config = config
        self._strategy_class: Optional[Type] = None
        self._signals: Dict[str, List[Dict[str, Any]]] = {}
        self._result: Optional[BacktestResult] = None
        self._results_dir: Optional[Path] = None  # Set before signal generation

        # Ensure data directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create data directories if they don't exist."""
        for dir_path in [
            self.config.ohlc_dir,
            self.config.signals_dir,
            self.config.results_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Data directories ready:")
        print(f"    OHLC:    {self.config.ohlc_dir}")
        print(f"    Signals: {self.config.signals_dir}")
        print(f"    Results: {self.config.results_dir}")

    def _get_strategy_class(self) -> Type:
        """Import and return strategy class."""
        if self._strategy_class is not None:
            return self._strategy_class

        strategy_name = self.config.strategy

        # Map common names to modules
        strategy_map = {
            "ha_mtf_stoch": ("strategies.ha_mtf_stoch", "HAMTFStoch"),
            "strat_multi_toggle": ("strategies.strat_multi_toggle", "StratMultiToggle"),
            "ema_only_long": ("strategies.ema_only_long", "EMAOnlyLong"),
        }

        if strategy_name in strategy_map:
            module_name, class_name = strategy_map[strategy_name]
        else:
            # Assume format: module.ClassName or just ClassName
            if "." in strategy_name:
                parts = strategy_name.rsplit(".", 1)
                module_name, class_name = parts[0], parts[1]
            else:
                raise ValueError(
                    f"Unknown strategy: {strategy_name}. "
                    f"Known strategies: {list(strategy_map.keys())}"
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
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                interval="1min",
                data_dir=data_dir,
            )
            data_files[symbol] = file_path

        return data_files

    def generate_signals(self, use_cache: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate signals for all symbols.

        Uses parallel chunk processing with optional caching.
        """
        print("\n[Step 2] Generating signals...")

        strategy_class = self._get_strategy_class()
        cache = SignalCache(self.config.signals_dir)

        # Smart calculation of max_workers
        max_workers = self.config.max_workers
        if not max_workers or max_workers <= 0:
            physical_cores = psutil.cpu_count(logical=False) or 1
            max_workers = max(1, physical_cores - 1)

        print(f"  Using {max_workers} workers for signal generation")

        all_signals = {}

        for symbol in self.config.symbols:
            # Check cache
            cache_key = get_cache_key(
                strategy_class,
                symbol,
                self.config.start_date,
                self.config.end_date,
                ignore_position=self.config.ignore_position,
            )

            if use_cache and not self.config.force_regenerate and cache.exists(cache_key):
                print(f"  {symbol}: Loading from cache...")
                signals = cache.load(cache_key)
                print(f"  {symbol}: Loaded {len(signals):,} cached signals")
                all_signals[symbol] = signals
                continue

            # Generate signals
            print(f"  {symbol}: Generating signals...")

            # Get data file and bar count
            ohlc_path = get_ohlc_file_path(
                symbol,
                self.config.ohlc_dir,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            total_bars = get_bar_count(
                symbol,
                self.config.ohlc_dir,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            print(f"  {symbol}: {total_bars:,} total bars")

            # Smart calculation of chunk_size
            chunk_size = self.config.chunk_size
            if not chunk_size or chunk_size <= 0:
                # 1. Total signals to generate
                warmup = self.config.warmup_size
                total_signals = total_bars - warmup

                if total_signals > 0:
                    # 2. Divide signals, rounding to nearest 5 as requested
                    # (This ensures 0-3 are equal, and 4 takes the remainder)
                    raw_signals_per_worker = total_signals / max_workers
                    signals_per_worker = int(round(raw_signals_per_worker / 5) * 5)

                    # 3. Final chunk size includes the warmup
                    chunk_size = signals_per_worker + warmup
                else:
                    chunk_size = total_bars

            print(f"  {symbol}: Using smart balanced chunk_size (nearest 5): {chunk_size:,} (warmup: {self.config.warmup_size:,})")

            # Create chunks (with debug output dir if results_dir is set)
            debug_output_dir = str(self._results_dir) if self._results_dir else ""
            chunks = create_chunks(
                symbol=symbol,
                total_bars=total_bars,
                chunk_size=chunk_size,
                warmup_size=self.config.warmup_size,
                ohlc_path=ohlc_path,
                debug_output_dir=debug_output_dir,
            )

            print(f"  {symbol}: {len(chunks)} chunks created")
            print(get_chunk_summary(chunks))

            # Run parallel signal generation
            if len(chunks) > 1 and max_workers > 1:
                symbol_signals = run_parallel_signal_generation(
                    chunks=chunks,
                    strategy_class=strategy_class,
                    max_workers=max_workers,
                    ignore_position=self.config.ignore_position,
                )
            else:
                symbol_signals = run_sequential_signal_generation(
                    chunks=chunks,
                    strategy_class=strategy_class,
                    ignore_position=self.config.ignore_position,
                )

            signals = symbol_signals.get(symbol, [])
            print(f"  {symbol}: Generated {len(signals):,} signals")

            # Cache signals
            if self.config.use_signal_cache:
                cache.save(
                    cache_key=cache_key,
                    signals=signals,
                    metadata={
                        "symbol": symbol,
                        "strategy": self.config.strategy,
                        "start_date": self.config.start_date,
                        "end_date": self.config.end_date,
                        "generated_at": get_et_timestamp(),
                    },
                )
                print(f"  {symbol}: Signals cached")

            all_signals[symbol] = signals

        self._signals = all_signals
        return all_signals

    def run_simulation(
        self,
        signals: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> BacktestResult:
        """Run trade simulation with signals."""
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

        print(f"  Total signals across all symbols: {len(all_signals):,}")

        # Load OHLC for simulation (need prices for equity tracking)
        # Use first symbol's data for simplicity
        symbol = self.config.symbols[0]

        ohlc_df = load_ohlc(
            symbol=symbol,
            data_dir=self.config.ohlc_dir,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
        )

        print(f"  OHLC data loaded: {len(ohlc_df):,} bars")

        # Run simulation
        # TP/SL: config override > strategy class default
        strategy_class = self._get_strategy_class()
        broker = SimulatedBroker(
            initial_capital=self.config.initial_capital,
            slippage_pct=self.config.slippage_pct,
            commission_per_trade=self.config.commission_per_trade,
            strategy_class=strategy_class,
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )

        sim_results = broker.run_simulation(all_signals, ohlc_df)

        print(f"  Simulation complete: {sim_results['total_trades']} trades executed")

        # Build BacktestResult with all statistics
        result = build_backtest_result(
            config=self.config,
            trades=sim_results["trades"],
            equity_curve=sim_results["equity_curve"],
            starting_capital=self.config.initial_capital,
            symbol=",".join(self.config.symbols),
            timeframe="1min",
        )

        self._result = result
        return result

    def save_results(
        self,
        results_dir: Optional[Path | str] = None,
    ) -> Path:
        """
        Save all results to disk including HTML dashboard.

        Note: Debug CSVs are saved by workers during signal generation.
        """
        if results_dir is None:
            results_dir = self._results_dir

        if results_dir is None:
            run_id = get_et_now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(self.config.results_dir) / run_id

        results_dir = Path(results_dir)

        # Gather all signals for saving
        all_signals = []
        for symbol_signals in self._signals.values():
            all_signals.extend(symbol_signals)

        # Save results (trades, equity, config, etc.)
        saved_dir = save_results(
            results_dir=results_dir,
            config=self.config,
            result=self._result,
            signals=all_signals,
        )

        # Generate HTML dashboard
        if self._result is not None:
            # Build signal config string for strat_multi_toggle strategy
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
                prefix=self.config.strategy,
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
            prefix=self.config.strategy,
            verbose=not self.config.save_results,  # Only show message if not saving
            signal_config=signal_config,
            hide_signals=self.config.hide_signals,
        )

        return Path(dashboard_path) if dashboard_path else None

    def run(self) -> BacktestResult:
        """Run complete backtest pipeline."""
        # Step 0: Create results directory (needed for debug CSV output during signal generation)
        run_id = get_et_now().strftime("%Y%m%d_%H%M%S")
        self._results_dir = Path(self.config.results_dir) / run_id
        self._results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Step 0] Results directory: {self._results_dir}")

        # Set SignalLogger to backtest mode
        SignalLogger.set_mode("backtest", run_id)

        try:
            # Step 1: Ensure data
            self.ensure_data()

            # Step 2: Generate signals (also saves debug CSVs per chunk)
            self.generate_signals(use_cache=self.config.use_signal_cache)

            # Step 3: Run simulation
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

            # Flush any accumulated signal logs
            for symbol in self.config.symbols:
                logger = SignalLogger.get_or_create(symbol, self.config.strategy)
                logger.flush()

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
        from_date="2024-01",  # 1 year of data
        initial_capital=100_000.0,
        slippage_pct=0.0002,
        commission_per_trade=0.5,
    )

    print("=" * 60)
    print("MULTIPROCESSING BACKTEST ENGINE")
    print("=" * 60)
    print(f"Symbols:       {', '.join(config.symbols)}")
    print(f"Strategy:      {config.strategy}")
    print(f"Period:        {config.start_date} to {config.end_date}")
    print(f"Capital:       ${config.initial_capital:,.0f}")
    print("=" * 60)

    engine = BacktestEngine(config)
    result = engine.run()

    print("\nBacktest complete!")
