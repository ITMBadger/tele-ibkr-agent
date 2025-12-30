"""
Strategy 1: EMA Only Long (Buy Only)

Buy when price crosses above EMA. No sell signal - exit via TP/SL only.
A faster, momentum-based trend-following strategy.

Uses 5min bars by default. EMA period is configurable.
"""

import numpy as np
import pandas as pd

from strategies._base import BaseStrategy
from strategies._ta import ema
import context


class EMAOnlyLong(BaseStrategy):
    """Buy above EMA. Exit via TP/SL only (no sell signal)."""

    # === STRATEGY PARAMETERS ===
    ID = "1"
    NAME = "EMA Only Long"
    DESCRIPTION = "Buy when price crosses above EMA, exit via TP/SL only"
    INTERVAL = 300  # Check every 5 minutes

    # Trading parameters
    EMA_PERIOD = 12
    QUANTITY = 0.001

    # OHLC data parameters
    OHLC_INTERVAL = "5min"  # Bar interval
    OHLC_DAYS = 10  # Days of history (ensures enough bars for EMA warmup)

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
        - 0 (HOLD): price <= EMA or during warmup period
        """
        df = df.copy()

        # Calculate EMA
        df["ema"] = ema(df["close"].values, cls.EMA_PERIOD)

        # Generate signals: 1=above EMA (BUY), 0=below or equal (HOLD)
        df["signal"] = np.where(df["close"] > df["ema"], 1, 0)

        # Handle warmup NaN
        df.loc[df["ema"].isna(), "signal"] = 0

        return cls.finalize_signals(df)

    async def execute(self) -> None:
        """Execute EMA crossover logic using compute_signals()."""
        # Check TP/SL exits first (skip entry logic if position was closed)
        if await self.check_and_close_if_stopped():
            return

        try:
            # Fetch OHLC data
            ohlc = await self.tiingo.get_ohlc(
                self.symbol,
                days=self.OHLC_DAYS,
                interval=self.OHLC_INTERVAL
            )
            if not ohlc:
                return

            # Convert to DataFrame and compute signals
            df = self.to_df(ohlc)
            df = self.compute_signals(df)

            # GET SIGNALS (iloc[-1] contains the shifted signal from the completed bar)
            current_signal = int(df["signal"].iloc[-1])

            # GET TRIGGER PRICE (iloc[-2] is the finalized close of the signal bar)
            # This ensures slippage checks compare against the backtest-identical price.
            current_price = df["close"].iloc[-2]

            ema_value = df["ema"].iloc[-1]

            # Prepare indicator columns for logging
            ema_series = [
                None if pd.isna(val) else val for val in df["ema"].tolist()
            ]
            indicator_cols = {f"ema_{self.EMA_PERIOD}": ema_series}

            # Update price in context
            context.latest_prices.set(self.symbol, current_price)

            # Execute based on signal (BUY ONLY - no sell signal)
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
