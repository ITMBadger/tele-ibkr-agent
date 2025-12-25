"""
Strategy 5: 100 EMA Conservative

Medium-term trend following with 100 EMA.
Balanced between responsiveness and false signals.

ALL PARAMETERS ARE FIXED - no external configuration.
"""

import numpy as np
import pandas as pd

from strategies._base import BaseStrategy
from strategies._ta import ema
import context


class EMA100Conservative(BaseStrategy):
    """Medium-term 100 EMA. Balanced approach with small position."""

    # === FIXED STRATEGY PARAMETERS ===
    ID = "5"
    NAME = "100 EMA Conservative"
    DESCRIPTION = "Medium-term trend following with 100 EMA"
    INTERVAL = 600  # Check every 10 minutes (conservative)

    # Trading parameters - ALL FIXED
    EMA_PERIOD = 100
    QUANTITY = 5  # Small position for conservative approach

    # Exit strategy (uses defaults from BaseStrategy: 1% SL, 2% TP)
    # Uncomment and modify to override:
    # STOP_LOSS_PCT = 1.0
    # TAKE_PROFIT_PCT = 2.0

    @classmethod
    def compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute EMA crossover signals (vectorized).

        Signal logic:
        - 1 (BUY): price > EMA
        - -1 (SELL): price < EMA
        - 0 (HOLD): during warmup period
        """
        df = df.copy()

        # Calculate EMA
        df["ema"] = ema(df["close"].values, cls.EMA_PERIOD)

        # Generate signals: 1=above EMA (BUY), -1=below EMA (SELL)
        df["signal"] = np.where(
            df["close"] > df["ema"],
            1,
            np.where(df["close"] < df["ema"], -1, 0),
        )

        # Handle warmup NaN
        df.loc[df["ema"].isna(), "signal"] = 0

        return df

    async def execute(self) -> None:
        """Execute 100 EMA crossover logic using compute_signals()."""
        # Check TP/SL exits first (skip entry logic if position was closed)
        if await self.check_and_close_if_stopped():
            return

        try:
            ohlc = await self.tiingo.get_daily_ohlc(
                self.symbol, days=self.EMA_PERIOD + 50
            )
            if not ohlc:
                return

            # Convert to DataFrame and compute signals
            df = self.to_df(ohlc)
            df = self.compute_signals(df)

            # Get current values
            current_signal = int(df["signal"].iloc[-1])
            current_price = df["close"].iloc[-1]
            ema_value = df["ema"].iloc[-1]

            # Prepare indicator columns for logging
            ema_series = [
                None if pd.isna(val) else val for val in df["ema"].tolist()
            ]
            indicator_cols = {f"ema_{self.EMA_PERIOD}": ema_series}

            context.latest_prices.set(self.symbol, current_price)

            if current_signal == 1 and not self.has_position():
                self.log(
                    f"🟢 {self.symbol} ${current_price:.2f} > "
                    f"EMA({self.EMA_PERIOD}) ${ema_value:.2f} → BUY"
                )
                triggered = self.buy()
                self.log_strategy_data(ohlc, indicator_cols, "BUY", triggered, "signal")

            elif current_signal == -1 and self.has_position():
                self.log(
                    f"🔴 {self.symbol} ${current_price:.2f} < "
                    f"EMA({self.EMA_PERIOD}) ${ema_value:.2f} → SELL"
                )
                triggered = self.sell()
                self.log_strategy_data(ohlc, indicator_cols, "SELL", triggered, "signal")

            elif self.should_log_periodic():
                signal = "HOLD" if self.has_position() else "NONE"
                self.log_strategy_data(ohlc, indicator_cols, signal, False, "periodic")

        except Exception as e:
            print(f"[{self.NAME}] Error for {self.symbol}: {e}")

