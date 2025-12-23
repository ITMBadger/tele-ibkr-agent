"""
Strategy 1: 200 EMA Long

Buy when price crosses above 200 EMA, sell when crosses below.
A slower, conservative trend-following strategy.

ALL PARAMETERS ARE FIXED - no external configuration.
"""

import numpy as np
import pandas as pd

from strategies._base import BaseStrategy
from strategies._ta import ema
import context


class EMA200Long(BaseStrategy):
    """Buy above 200 EMA, sell below. Conservative trend following."""

    # === FIXED STRATEGY PARAMETERS ===
    ID = "1"
    NAME = "200 EMA Long"
    DESCRIPTION = "Buy when price crosses above 200 EMA, sell when crosses below"
    INTERVAL = 300  # Check every 5 minutes (slow strategy)

    # Trading parameters - ALL FIXED
    EMA_PERIOD = 200
    QUANTITY = 10

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
        """Execute 200 EMA crossover logic using compute_signals()."""
        try:
            # Fetch OHLC data
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

            # Update price in context
            context.latest_prices.set(self.symbol, current_price)

            # Execute based on signal
            if current_signal == 1 and not self.has_position():
                self.log(
                    f"🟢 {self.symbol} ${current_price:.2f} > "
                    f"EMA({self.EMA_PERIOD}) ${ema_value:.2f} → BUY"
                )
                triggered = self.buy()
                self.log_strategy_data(
                    ohlc_bars=ohlc,
                    indicator_columns=indicator_cols,
                    signal="BUY",
                    triggered=triggered,
                    event_type="signal",
                )

            elif current_signal == -1 and self.has_position():
                self.log(
                    f"🔴 {self.symbol} ${current_price:.2f} < "
                    f"EMA({self.EMA_PERIOD}) ${ema_value:.2f} → SELL"
                )
                triggered = self.sell()
                self.log_strategy_data(
                    ohlc_bars=ohlc,
                    indicator_columns=indicator_cols,
                    signal="SELL",
                    triggered=triggered,
                    event_type="signal",
                )

            # Periodic logging (every 30 minutes)
            elif self.should_log_periodic():
                signal = "HOLD" if self.has_position() else "NONE"
                self.log_strategy_data(
                    ohlc_bars=ohlc,
                    indicator_columns=indicator_cols,
                    signal=signal,
                    triggered=False,
                    event_type="periodic",
                )

        except Exception as e:
            print(f"[{self.NAME}] Error for {self.symbol}: {e}")

