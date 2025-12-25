"""
Strategy 3: RSI Oversold Bounce

Buy when RSI drops below 30 (oversold), sell when RSI rises above 70 (overbought).
Mean reversion strategy.

ALL PARAMETERS ARE FIXED - no external configuration.
"""

import numpy as np
import pandas as pd

from strategies._base import BaseStrategy
from strategies._ta import rsi
import context


class RSIOversoldBounce(BaseStrategy):
    """Buy oversold (RSI < 30), sell overbought (RSI > 70)."""

    # === FIXED STRATEGY PARAMETERS ===
    ID = "3"
    NAME = "RSI Oversold Bounce"
    DESCRIPTION = "Buy when RSI < 30, sell when RSI > 70"
    INTERVAL = 60  # Check every minute (RSI changes quickly)

    # Trading parameters - ALL FIXED
    RSI_PERIOD = 14
    BUY_THRESHOLD = 30  # Oversold
    SELL_THRESHOLD = 70  # Overbought
    QUANTITY = 15

    # Exit strategy (uses defaults from BaseStrategy: 1% SL, 2% TP)
    # Uncomment and modify to override:
    # STOP_LOSS_PCT = 1.0
    # TAKE_PROFIT_PCT = 2.0

    @classmethod
    def compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI bounce signals (vectorized).

        Signal logic:
        - 1 (BUY): RSI < BUY_THRESHOLD (oversold)
        - -1 (SELL): RSI > SELL_THRESHOLD (overbought)
        - 0 (HOLD): RSI between thresholds or during warmup
        """
        df = df.copy()

        # Calculate RSI
        df["rsi"] = rsi(df["close"].values, cls.RSI_PERIOD)

        # Generate signals: 1=oversold (BUY), -1=overbought (SELL)
        df["signal"] = np.where(
            df["rsi"] < cls.BUY_THRESHOLD,
            1,
            np.where(df["rsi"] > cls.SELL_THRESHOLD, -1, 0),
        )

        # Handle warmup NaN
        df.loc[df["rsi"].isna(), "signal"] = 0

        return df

    async def execute(self) -> None:
        """Execute RSI bounce logic using compute_signals()."""
        # Check TP/SL exits first (skip entry logic if position was closed)
        if await self.check_and_close_if_stopped():
            return

        try:
            ohlc = await self.tiingo.get_daily_ohlc(
                self.symbol, days=self.RSI_PERIOD + 100
            )
            if not ohlc or len(ohlc) < self.RSI_PERIOD + 1:
                return

            # Convert to DataFrame and compute signals
            df = self.to_df(ohlc)
            df = self.compute_signals(df)

            # Get current values
            current_signal = int(df["signal"].iloc[-1])
            current_price = df["close"].iloc[-1]
            rsi_value = df["rsi"].iloc[-1]

            # Prepare indicator columns for logging
            rsi_series = [
                None if pd.isna(val) else val for val in df["rsi"].tolist()
            ]
            indicator_cols = {f"rsi_{self.RSI_PERIOD}": rsi_series}

            context.latest_prices.set(self.symbol, current_price)

            if current_signal == 1 and not self.has_position():
                self.log(
                    f"🟢 {self.symbol} RSI={rsi_value:.1f} < {self.BUY_THRESHOLD} "
                    f"(Oversold) → BUY"
                )
                triggered = self.buy()
                self.log_strategy_data(ohlc, indicator_cols, "BUY", triggered, "signal")

            elif current_signal == -1 and self.has_position():
                self.log(
                    f"🔴 {self.symbol} RSI={rsi_value:.1f} > {self.SELL_THRESHOLD} "
                    f"(Overbought) → SELL"
                )
                triggered = self.sell()
                self.log_strategy_data(ohlc, indicator_cols, "SELL", triggered, "signal")

            elif self.should_log_periodic():
                signal = "HOLD" if self.has_position() else "NONE"
                self.log_strategy_data(ohlc, indicator_cols, signal, False, "periodic")

        except Exception as e:
            print(f"[{self.NAME}] Error for {self.symbol}: {e}")

