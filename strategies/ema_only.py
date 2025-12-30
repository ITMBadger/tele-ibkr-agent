# strategies/ema_only.py
"""
Strategy 1: EMA Only (Long / Short / Both)

Trade when price crosses EMA. Exit via TP/SL only.
A simple, momentum-based trend-following strategy.

Modes:
- LONG_ONLY: Buy when close > EMA
- SHORT_ONLY: Sell when close < EMA
- BOTH: Long above EMA, Short below EMA
"""

from enum import Enum
import numpy as np
import pandas as pd
from typing import Tuple, List

from strategies._template import StrategyTemplate
from strategies._ta import ema


class TradeMode(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"


class EMAOnly(StrategyTemplate):
    """Trade based on EMA crossover. Exit via TP/SL only."""

    # === STRATEGY CONFIG ===
    ID = "1"
    NAME = "EMA Only"
    DESCRIPTION = "Trade when price crosses EMA, exit via TP/SL only"
    INTERVAL = 60  # Check every 1 minute

    # === TRADE MODE TOGGLE ===
    TRADE_MODE = TradeMode.SHORT_ONLY  # Options: LONG_ONLY, SHORT_ONLY, BOTH

    # === MTF CONFIG ===
    TIMEFRAMES = {
        "5min": {"use_ha": False},
    }
    OHLC_DAYS = 5

    # === STRATEGY PARAMS ===
    EMA_PERIOD = 12
    QUANTITY = 0.001

    # Exit strategy (uses defaults from BaseStrategy)
    # STOP_LOSS_PCT = 1.0
    # TAKE_PROFIT_PCT = 2.0

    # === INDICATOR CALCULATION ===
    @classmethod
    def calc_5m(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate EMA on 5min data."""
        df["ema"] = ema(df["close"].values, cls.EMA_PERIOD)
        return df, ["ema"]

    # === SIGNAL LOGIC ===
    @classmethod
    def get_signal_vector(cls, df: pd.DataFrame) -> np.ndarray:
        """
        Generate signals based on TRADE_MODE.

        Returns:
            1 = BUY (long entry)
            -1 = SELL (short entry)
            0 = HOLD
        """
        above_ema = df["close"] > df["ema"]
        below_ema = df["close"] < df["ema"]
        ema_valid = df["ema"].notna()

        if cls.TRADE_MODE == TradeMode.LONG_ONLY:
            # Only long signals
            return np.where(above_ema & ema_valid, 1, 0)

        elif cls.TRADE_MODE == TradeMode.SHORT_ONLY:
            # Only short signals
            return np.where(below_ema & ema_valid, -1, 0)

        else:  # BOTH
            # Long above EMA, Short below EMA
            signal = np.zeros(len(df))
            signal[above_ema & ema_valid] = 1
            signal[below_ema & ema_valid] = -1
            return signal

    # === EXECUTE (override to handle long/short) ===
    async def execute(self) -> None:
        """Execute EMA strategy with long/short support."""
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

            # Compute signals
            df = self.to_df(ohlc)
            df = self.compute_signals(df)

            # Get signal and price
            # signal[-1] is the shifted signal, corresponding to TRIGGER_BAR_INDEX
            current_signal = int(df["signal"].iloc[-1])
            current_price = df["close"].iloc[self.TRIGGER_BAR_INDEX]
            ema_value = df["ema"].iloc[self.TRIGGER_BAR_INDEX]

            # Update context
            import context
            context.latest_prices.set(self.symbol, current_price)

            # Build indicator columns for logging
            ema_series = [None if pd.isna(v) else v for v in df["ema"].tolist()]
            signal_series = [v for v in df["signal"].tolist()]
            indicator_cols = {
                f"ema_{self.EMA_PERIOD}": ema_series,
                "strat_signal": signal_series
            }

            # === LONG SIGNAL ===
            if current_signal == 1:
                if self.TRADE_MODE in (TradeMode.LONG_ONLY, TradeMode.BOTH):
                    if not self.has_position():
                        self.log(
                            f"🟢 LONG {self.symbol} @ ${current_price:.2f} > "
                            f"EMA({self.EMA_PERIOD}) ${ema_value:.2f}"
                        )
                        triggered = self.buy()
                        self.log_strategy_data(ohlc, indicator_cols, "BUY", triggered, "signal")

            # === SHORT SIGNAL ===
            elif current_signal == -1:
                if self.TRADE_MODE in (TradeMode.SHORT_ONLY, TradeMode.BOTH):
                    if not self.has_position():
                        self.log(
                            f"🔴 SHORT {self.symbol} @ ${current_price:.2f} < "
                            f"EMA({self.EMA_PERIOD}) ${ema_value:.2f}"
                        )
                        triggered = self.sell()
                        self.log_strategy_data(ohlc, indicator_cols, "SELL", triggered, "signal")

            # === PERIODIC LOGGING ===
            elif self.should_log_periodic():
                signal_str = "BUY" if current_signal == 1 else ("SELL" if current_signal == -1 else "NONE")
                status = f"{signal_str} (pos: {self.has_position()})"
                self.log_strategy_data(ohlc, indicator_cols, status, False, "periodic")

        except Exception as e:
            import traceback
            print(f"[{self.NAME}] Error: {e}")
            traceback.print_exc()
