# strategies/_template.py
"""
Strategy Template Base Class (StrategyTemplate)

A clean, template-based approach for strategies that:
1. Fetch 1min OHLC data (with cache for historical, fresh for today)
2. Resample to higher timeframes (5min, 15min, 30min, etc.)
3. Calculate indicators on each timeframe (with optional HA conversion)
4. Merge indicators back to 1min using forward-fill (merge_asof)
5. Generate signals on 1min data
6. Apply safety shift via finalize_signals() so signal[-1] uses close[-2]

Pattern:
    1min OHLC → resample to HTF → calc indicators → merge back to 1min → signal

Usage:
    class MyStrategy(StrategyTemplate):
        TIMEFRAMES = {"5min": {"use_ha": True}, "30min": {"use_ha": True}}

        def calc_5m(self, df):
            df["ema"] = ema(df["close"].values, 12)
            return df, ["ema"]  # Return df and columns to merge

        def calc_30m(self, df):
            df["stoch_rising"] = ...
            return df, ["stoch_rising"]

        def get_signal_vector(self, df):
            return np.where(df["close"] > df["ema"], 1, 0)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from strategies._trading_mech import BaseStrategy
from strategies._ta import heikin_ashi_vectorized


class StrategyTemplate(BaseStrategy):
    """
    Template base class for strategies using the HTF-merge pattern.

    Pattern: 1min OHLC → resample to HTF → calc indicators → merge back to 1min → signal

    Subclasses only need to:
    1. Define TIMEFRAMES config (which timeframes, use_ha option)
    2. Implement calc_Xm() methods for each timeframe (indicator calculations)
    3. Implement get_signal_vector() for signal logic (vectorized)
    """

    # === OVERRIDE THESE IN SUBCLASS ===

    # Timeframe configuration
    # Keys: pandas resample strings ("5min", "15min", "30min", "1h")
    # Values: dict with options:
    #   - use_ha: bool - auto-convert to Heikin-Ashi before calc method
    TIMEFRAMES: Dict[str, Dict[str, Any]] = {
        # Example: {"5min": {"use_ha": True}, "30min": {"use_ha": False}}
    }

    # OHLC settings
    OHLC_DAYS = 10

    # Resample anchor (market open)
    RESAMPLE_ANCHOR = "09:30:00"

    # === TEMPLATE METHODS (don't override) ===

    @classmethod
    def _get_anchor(cls, df_index) -> pd.Timestamp:
        """Get timezone-aware anchor for resampling."""
        anchor = pd.Timestamp(f"1970-01-01 {cls.RESAMPLE_ANCHOR}")
        if df_index.tz is not None:
            anchor = anchor.tz_localize(df_index.tz)
        return anchor

    @classmethod
    def _resample(cls, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample 1min data to specified timeframe."""
        anchor = cls._get_anchor(df.index)

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
        # Include volume if present
        if "volume" in df.columns:
            agg_dict["volume"] = "sum"

        return df.resample(timeframe, origin=anchor).agg(agg_dict).dropna()

    @classmethod
    def _apply_ha(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Heikin-Ashi conversion."""
        return heikin_ashi_vectorized(df)

    @classmethod
    def _get_calc_method(cls, timeframe: str):
        """Get the calc method for a timeframe (e.g., '5min' -> calc_5m)."""
        # Convert timeframe to method name: "5min" -> "calc_5m", "30min" -> "calc_30m"
        tf_short = timeframe.replace("min", "m").replace("hour", "h")
        method_name = f"calc_{tf_short}"
        return getattr(cls, method_name, None)

    @classmethod
    def compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Template method - handles the full MTF pipeline.

        Flow:
        1. Resample 1min to all configured timeframes
        2. Apply HA conversion if configured
        3. Call calc_Xm() for each timeframe
        4. Merge all indicators back to 1min
        5. Call get_signal_vector() for final signal
        6. Apply finalize_signals() for safety shift
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Store original 1min for final merge
        df_1m = df.copy()

        # Process each timeframe
        merge_data: List[Tuple[pd.DataFrame, List[str]]] = []

        for timeframe, config in cls.TIMEFRAMES.items():
            # 1. Resample
            df_tf = cls._resample(df, timeframe)

            if df_tf.empty:
                continue

            # 2. Apply HA if configured
            if config.get("use_ha", False):
                df_tf = cls._apply_ha(df_tf)

            # 3. Get and call calc method
            calc_method = cls._get_calc_method(timeframe)
            if calc_method is None:
                continue

            # calc_Xm should return (df, cols_to_merge)
            result = calc_method(df_tf)

            if isinstance(result, tuple) and len(result) == 2:
                df_tf, cols = result
            else:
                # Fallback: assume all non-OHLC columns should merge
                df_tf = result
                base_cols = {"open", "high", "low", "close", "volume",
                            "ha_open", "ha_high", "ha_low", "ha_close"}
                cols = [c for c in df_tf.columns if c not in base_cols]

            if cols:
                merge_data.append((df_tf[cols], cols))

        # 4. Merge all back to 1min using merge_asof
        df_1m = df_1m.reset_index()

        for df_tf, cols in merge_data:
            df_tf_ready = df_tf.reset_index().rename(columns={"index": "date"})
            df_1m = pd.merge_asof(
                df_1m.sort_values("date"),
                df_tf_ready[["date"] + cols].sort_values("date"),
                on="date",
                direction="backward"  # Forward fill
            )

        # 5. Generate signals (vectorized)
        signal_result = cls.get_signal_vector(df_1m)

        if isinstance(signal_result, pd.Series):
            df_1m["signal"] = signal_result.values
        else:
            df_1m["signal"] = signal_result

        # Handle NaN signals
        df_1m["signal"] = df_1m["signal"].fillna(0).astype(int)

        # 6. Finalize (safety shift)
        return cls.finalize_signals(df_1m)

    # === OVERRIDE THESE IN SUBCLASS ===

    @classmethod
    def calc_5m(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculate 5min indicators.

        Args:
            df: 5min OHLC data (may be HA-converted if use_ha=True)

        Returns:
            Tuple of (df with indicators, list of column names to merge to 1min)

        Example:
            df["ema"] = ema(df["close"].values, 12)
            return df, ["ema"]
        """
        return df, []

    @classmethod
    def calc_15m(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate 15min indicators."""
        return df, []

    @classmethod
    def calc_30m(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate 30min indicators."""
        return df, []

    @classmethod
    def calc_1h(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate 1hour indicators."""
        return df, []

    @classmethod
    def get_signal_vector(cls, df: pd.DataFrame) -> np.ndarray:
        """
        Generate signals (vectorized) on merged 1min data.

        Args:
            df: 1min data with all indicators merged

        Returns:
            numpy array or pandas Series of signals (1=BUY, 0=HOLD)

        Example:
            return np.where(df["close"] > df["ema"], 1, 0)
        """
        return np.zeros(len(df))

    # === EXECUTE (uses compute_signals) ===

    async def execute(self) -> None:
        """Execute strategy using compute_signals()."""
        if await self.check_and_close_if_stopped():
            return

        try:
            # Fetch 1min OHLC
            ohlc = await self.tiingo.get_ohlc(
                self.symbol,
                days=self.OHLC_DAYS,
                interval="1min"
            )
            if not ohlc or len(ohlc) < 100:
                return

            # Compute signals (uses template)
            df = self.to_df(ohlc)
            df = self.compute_signals(df)

            # Get signal and price
            # signal[-1] is the shifted signal, corresponding to TRIGGER_BAR_INDEX
            current_signal = int(df["signal"].iloc[-1])
            current_price = df["close"].iloc[self.TRIGGER_BAR_INDEX]

            # Update context
            import context
            context.latest_prices.set(self.symbol, current_price)

            # Build indicator columns for logging
            base_cols = {"date", "open", "high", "low", "close", "volume"}
            indicator_cols = {}
            for col in df.columns:
                if col not in base_cols:
                    col_name = "strat_signal" if col == "signal" else col
                    indicator_cols[col_name] = [None if pd.isna(v) else v for v in df[col].tolist()]

            # Execute trade if signal
            if current_signal == 1 and not self.has_position():
                self.log(f"🟢 LONG {self.symbol} @ ${current_price:.2f}")
                triggered = self.buy()
                self.log_strategy_data(ohlc, indicator_cols, "BUY", triggered, "signal")

            elif self.should_log_periodic():
                signal_str = "BUY" if current_signal == 1 else "NONE"
                status = f"{signal_str} (pos: {self.has_position()})"
                self.log_strategy_data(ohlc, indicator_cols, status, False, "periodic")

        except Exception as e:
            import traceback
            print(f"[{self.NAME}] Error: {e}")
            traceback.print_exc()
