# strategies/ha_mtf_stoch_ema.py
"""
Strategy 6: Configurable HA + MTF Stoch + BB + EMA

Multi-timeframe strategy combining 5min Heikin Ashi bars, 30min Stochastic,
and 5min EMA 200 confirmation.

Signals are toggleable via SignalType Enum.
"""

import math
import traceback
from enum import Enum

import numpy as np
import pandas as pd

from strategies._base import BaseStrategy
from strategies._ta import (
    heikin_ashi_vectorized,
    stochastic,
    bbands,
    ema
)
import context


class SignalType(Enum):
    STOCH_RISING = "stoch_rising"
    HA_TWO_GREEN = "ha_two_green"
    BB_TOUCH = "bb_touch"
    ABOVE_EMA_200 = "above_ema_200"


# Detail columns for each signal type
SIGNAL_COLUMNS = {
    SignalType.STOCH_RISING: {
        "30m": ["stoch_rising", "stoch_d", "stoch_d_prev"],
    },
    SignalType.HA_TWO_GREEN: {
        "5m": ["ha_two_green", "ha_green_t1", "ha_green_t2"],
    },
    SignalType.BB_TOUCH: {
        "5m": ["bb_touch", "ha_low_t2", "ha_low_t3", "bb_lower_t2", "bb_lower_t3", "touch_t2", "touch_t3"],
    },
    SignalType.ABOVE_EMA_200: {
        "5m": ["above_ema_200", "ema_200", "close_t1"],
    },
}


class HAMTFStochEMA(BaseStrategy):
    """Configurable 5min HA bars + 30min Stochastic (Raw) + BB touch + EMA 200."""

    ID = "7"
    NAME = "HA + 30m Stoch MTF + BB + EMA"
    DESCRIPTION = "Configurable 5min HA bars + 30min Stochastic (Raw) + BB touch + EMA 200"
    INTERVAL = 60
    QUANTITY = 2

    # Stochastic params (applied to 30min Raw OHLC, anchored to 09:30)
    STOCH_FASTK = 14
    STOCH_SLOWK = 5
    STOCH_SLOWD = 5

    # Bollinger Bands params (applied to 5min HA data)
    BB_LENGTH = 20
    BB_STDDEV = 1.5

    # EMA params
    EMA_PERIOD = 200

    OHLC_DAYS = 10  # Need enough trading days for 200+ 5-min bars

    # Enabled Signals - Toggle these to turn on/off specific triggers
    ENABLED_BUY_SIGNALS = {
        SignalType.STOCH_RISING,
        SignalType.HA_TWO_GREEN,
        SignalType.BB_TOUCH,
        # SignalType.ABOVE_EMA_200,
    }

    @classmethod
    def compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized signal computation for configurable HA + MTF Stoch + BB + EMA strategy.
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # ========== Step 1: Resample to 5m and 30m ==========
        anchor = pd.Timestamp('1970-01-01 09:30:00')
        if df.index.tz is not None:
            anchor = anchor.tz_localize(df.index.tz)

        agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
        df_5m = df.resample('5min', origin=anchor).agg(agg_dict).dropna()
        df_30m = df.resample('30min', origin=anchor).agg(agg_dict).dropna()

        # Check minimum data requirements
        min_5m_bars = max(cls.BB_LENGTH, cls.EMA_PERIOD) + 5
        min_30m_bars = cls.STOCH_FASTK + cls.STOCH_SLOWK + cls.STOCH_SLOWD
        if len(df_5m) < min_5m_bars or len(df_30m) < min_30m_bars:
            df = df.reset_index()
            df['signal'] = 0
            return df

        # ========== Step 2: 30m Stochastic (on RAW OHLC) ==========
        slowk, slowd = stochastic(
            df_30m['high'].values,
            df_30m['low'].values,
            df_30m['close'].values,
            cls.STOCH_FASTK,
            cls.STOCH_SLOWK,
            cls.STOCH_SLOWD,
        )
        stoch_d = pd.Series(slowd, index=df_30m.index)
        df_30m['stoch_d'] = stoch_d.shift(1)  # Completed bar's value
        df_30m['stoch_d_prev'] = stoch_d.shift(2)
        df_30m['stoch_rising'] = df_30m['stoch_d'] > df_30m['stoch_d_prev']

        # ========== Step 3: 5m Indicators (HA, BB, EMA) ==========
        # Heikin Ashi
        df_5m = heikin_ashi_vectorized(df_5m)
        df_5m['ha_green'] = df_5m['ha_close'] > df_5m['ha_open']
        df_5m['ha_green_t1'] = df_5m['ha_green'].shift(1)
        df_5m['ha_green_t2'] = df_5m['ha_green'].shift(2)
        df_5m['ha_two_green'] = df_5m['ha_green_t1'] & df_5m['ha_green_t2']

        # Bollinger Bands (on HA closes)
        _, _, bb_lower = bbands(
            df_5m['ha_close'].values,
            cls.BB_LENGTH,
            cls.BB_STDDEV,
            cls.BB_STDDEV,
        )
        bb_low_series = pd.Series(bb_lower, index=df_5m.index)

        # BB touch at t-2 or t-3 relative to signal bar
        df_5m['ha_low_t2'] = df_5m['ha_low'].shift(3)
        df_5m['ha_low_t3'] = df_5m['ha_low'].shift(4)
        df_5m['bb_lower_t2'] = bb_low_series.shift(3)
        df_5m['bb_lower_t3'] = bb_low_series.shift(4)
        df_5m['touch_t2'] = df_5m['ha_low_t2'] <= df_5m['bb_lower_t2']
        df_5m['touch_t3'] = df_5m['ha_low_t3'] <= df_5m['bb_lower_t3']
        df_5m['bb_touch'] = df_5m['touch_t2'] | df_5m['touch_t3']

        # EMA 200 (on RAW close)
        ema_200_arr = ema(df_5m['close'].values, cls.EMA_PERIOD)
        ema_200_series = pd.Series(ema_200_arr, index=df_5m.index)
        df_5m['ema_200'] = ema_200_series.shift(1)
        df_5m['close_t1'] = df_5m['close'].shift(1)
        df_5m['above_ema_200'] = df_5m['close_t1'] > df_5m['ema_200']

        # ========== Step 4: Build columns for merge (only enabled signals) ==========
        cols_5m = []
        cols_30m = []
        for sig in cls.ENABLED_BUY_SIGNALS:
            sig_cols = SIGNAL_COLUMNS.get(sig, {})
            cols_5m.extend(sig_cols.get("5m", []))
            cols_30m.extend(sig_cols.get("30m", []))

        # ========== Step 5: Map back to 1m using merge_asof ==========
        df = df.reset_index()

        if cols_5m:
            df_5m_ready = df_5m[cols_5m].reset_index().rename(columns={'index': 'date'})
            df = pd.merge_asof(df.sort_values('date'), df_5m_ready.sort_values('date'), on='date', direction='backward')

        if cols_30m:
            df_30m_ready = df_30m[cols_30m].reset_index().rename(columns={'index': 'date'})
            df = pd.merge_asof(df.sort_values('date'), df_30m_ready.sort_values('date'), on='date', direction='backward')

        # ========== Step 6: Final Signal Logic (Toggleable) ==========
        buy_condition = pd.Series(True, index=df.index)
        for sig in SignalType:
            if sig in cls.ENABLED_BUY_SIGNALS:
                col_name = sig.value
                if col_name in df.columns:
                    buy_condition &= df[col_name].fillna(False)
        
        # If no signals enabled, no trades
        if not cls.ENABLED_BUY_SIGNALS:
            buy_condition = pd.Series(False, index=df.index)

        df['signal'] = np.where(buy_condition, 1, 0)

        return df

    async def execute(self) -> None:
        """Execute configurable HA + MTF Stoch + BB + EMA strategy."""
        if await self.check_and_close_if_stopped():
            return

        try:
            ohlc_1m = await self.tiingo.get_intraday_ohlc(
                self.symbol, days=self.OHLC_DAYS, interval="1min"
            )
            if not ohlc_1m or len(ohlc_1m) < 100:
                return

            df = self.to_df(ohlc_1m)
            df = self.compute_signals(df)
            self.capture_debug_row(df)

            current_signal = int(df['signal'].iloc[-1])
            current_price = df['close'].iloc[-1]
            context.latest_prices.set(self.symbol, current_price)

            # Build indicator columns for logging
            indicator_cols = {}
            for col in df.columns:
                if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'signal']:
                    indicator_cols[col] = [None if pd.isna(v) else v for v in df[col].tolist()]

            # Debug logs
            signal_states = {sig.name: (df[sig.value].iloc[-1] if sig.value in df.columns else False) 
                             for sig in SignalType if sig in self.ENABLED_BUY_SIGNALS}
            print(f"   🔍 Signal check: {signal_states}, has_pos={self.has_position()}")

            if current_signal == 1 and not self.has_position():
                enabled_str = ", ".join([f"{k}={v}" for k, v in signal_states.items()])
                self.log(f"🟢 LONG {self.symbol} @ ${current_price:.2f}\n   • {enabled_str}")
                triggered = self.buy()
                self.log_strategy_data(ohlc_1m, indicator_cols, "BUY", triggered, "signal")

            elif self.should_log_periodic():
                status = "HOLD" if self.has_position() else "NONE"
                self.log_strategy_data(ohlc_1m, indicator_cols, status, False, "periodic")

        except Exception as e:
            print(f"[{self.NAME}] Error: {e}")
            traceback.print_exc()
