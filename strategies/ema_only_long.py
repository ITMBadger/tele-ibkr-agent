"""
Strategy 1: EMA Only Long (Buy Only)

Buy when price crosses above EMA. No sell signal - exit via TP/SL only.
A faster, momentum-based trend-following strategy.

Uses 5min bars by default. EMA period is configurable.
"""

import numpy as np
import pandas as pd

from strategies._trading_mech import BaseStrategy
from strategies._ta import ema
import context


class EMAOnlyLong(BaseStrategy):
    """Buy above EMA. Exit via TP/SL only (no sell signal)."""

    # === STRATEGY PARAMETERS ===
    ID = "1"
    NAME = "EMA Only Long"
    DESCRIPTION = "Buy when price crosses above EMA, exit via TP/SL only"
    INTERVAL = 60  # Check every 1 minutes

    # Trading parameters
    EMA_PERIOD = 12
    QUANTITY = 0.001

    # OHLC data parameters
    OHLC_DAYS = 5  # Days of history (ensures enough bars for EMA warmup)
    SIGNAL_INTERVAL = "5min"  # Resample 1min data to this interval for signals

    # Exit strategy (uses defaults from BaseStrategy: 1% SL, 2% TP)
    # Uncomment and modify to override:
    # STOP_LOSS_PCT = 1.0
    # TAKE_PROFIT_PCT = 2.0

    @classmethod
    def compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute EMA crossover signals (vectorized).

        Flow:
        1. Resample 1min → 5min
        2. Calculate EMA on 5min data
        3. Map EMA back to 1min using merge_asof (forward fill)
        4. Generate signals on 1min: close > EMA = BUY

        Signal logic:
        - 1 (BUY): 1min close > 5min EMA (forward filled)
        - 0 (HOLD): 1min close <= EMA or during warmup period
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # === Step 1: Resample 1min to 5min ===
        df_5m = df.set_index("date")
        anchor = pd.Timestamp("1970-01-01 09:30:00")
        if df_5m.index.tz is not None:
            anchor = anchor.tz_localize(df_5m.index.tz)

        df_5m = df_5m.resample(cls.SIGNAL_INTERVAL, origin=anchor).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        # === Step 2: Calculate EMA on 5min data ===
        df_5m["ema"] = ema(df_5m["close"].values, cls.EMA_PERIOD)

        # === Step 3: Map EMA back to 1min using merge_asof (forward fill) ===
        df_5m_ema = df_5m[["ema"]].reset_index()
        df = pd.merge_asof(
            df.sort_values("date"),
            df_5m_ema.sort_values("date"),
            on="date",
            direction="backward"  # Forward fill: use most recent 5min EMA
        )

        # === Step 4: Generate signals on 1min data ===
        df["signal"] = np.where(df["close"] > df["ema"], 1, 0)
        df.loc[df["ema"].isna(), "signal"] = 0

        return cls.finalize_signals(df)

    async def execute(self) -> None:
        """Execute EMA crossover logic using compute_signals()."""
        # Check TP/SL exits first (skip entry logic if position was closed)
        if await self.check_and_close_if_stopped():
            return

        try:
            # Fetch 1min OHLC data
            ohlc = await self.tiingo.get_ohlc(
                self.symbol,
                days=self.OHLC_DAYS,
                interval="1min"
            )
            if not ohlc:
                return

            # Convert to DataFrame and compute signals (reuse compute_signals logic)
            df_1m = self.to_df(ohlc)
            df_1m = self.compute_signals(df_1m)

            # GET SIGNALS from 1min data
            # iloc[-1] is the shifted signal (based on previous bar close)
            current_signal = int(df_1m["signal"].iloc[-1])

            # GET TRIGGER PRICE: iloc[-2] is the previous completed bar's close
            current_price = df_1m["close"].iloc[-2]

            ema_value = df_1m["ema"].iloc[-2]

            # Prepare indicator columns for logging
            ema_series = [
                None if pd.isna(val) else val for val in df_1m["ema"].tolist()
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
