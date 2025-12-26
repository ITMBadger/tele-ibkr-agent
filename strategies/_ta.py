# strategies/_ta.py
"""
Technical Analysis utilities for strategies.

Provides simple, lowercase indicator functions that return numpy arrays.
All indicators follow the industry-standard naming convention (TA-Lib, pandas-ta).

Contains:
- Heikin Ashi conversion (vectorized for DataFrames)
- Technical indicators (ema, rsi, bbands, stochastic)

Usage:
    from strategies._ta import heikin_ashi_vectorized, ema, rsi, bbands, stochastic

    # All indicators return numpy arrays
    ema_series = ema(closes, period=200)  # Returns array
    current_ema = ema_series[-1]  # Get last value
"""

import numpy as np
import pandas as pd
import talib


# =============================================================================
# HEIKIN ASHI
# =============================================================================

def heikin_ashi_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized Heikin Ashi calculation for DataFrames.

    The HA_Open calculation is inherently sequential (depends on previous bar),
    so we use a fast loop for that part.

    Formulas:
        HA_Close = (O + H + L + C) / 4
        HA_Open  = (prev_HA_Open + prev_HA_Close) / 2  (first bar: (O + C) / 2)
        HA_High  = max(H, HA_Open, HA_Close)
        HA_Low   = min(L, HA_Open, HA_Close)

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Same DataFrame with added 'ha_open', 'ha_high', 'ha_low', 'ha_close' columns
    """
    df = df.copy()

    # HA Close is fully vectorized: (O + H + L + C) / 4
    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0

    # HA Open is recursive: (prev_ha_open + prev_ha_close) / 2
    # First bar: (O + C) / 2
    # Must use a loop since each value depends on the previous
    n = len(df)
    ha_open = np.zeros(n)
    ha_close_arr = df['ha_close'].values
    open_arr = df['open'].values
    close_arr = df['close'].values

    if n > 0:
        ha_open[0] = (open_arr[0] + close_arr[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i - 1] + ha_close_arr[i - 1]) / 2.0

    df['ha_open'] = ha_open

    # HA High/Low are fully vectorized
    df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

    return df


# =============================================================================
# INDICATORS (talib wrappers)
# =============================================================================

def ema(closes: list[float], period: int) -> np.ndarray:
    """Exponential Moving Average."""
    return talib.EMA(np.array(closes, dtype=float), timeperiod=period)


def rsi(closes: list[float], period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    return talib.RSI(np.array(closes, dtype=float), timeperiod=period)


def stochastic(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    fastk_period: int = 14,
    slowk_period: int = 5,
    slowd_period: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Oscillator.

    Returns:
        (slowk, slowd) arrays
    """
    slowk, slowd = talib.STOCH(
        np.array(highs, dtype=float),
        np.array(lows, dtype=float),
        np.array(closes, dtype=float),
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowk_matype=0,  # SMA
        slowd_period=slowd_period,
        slowd_matype=0   # SMA
    )
    return slowk, slowd


def bbands(
    closes: list[float],
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands.

    Returns:
        (upper, middle, lower) arrays
    """
    return talib.BBANDS(
        np.array(closes, dtype=float),
        timeperiod=period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
        matype=0  # SMA
    )
