# strategies/_ta.py
"""
Technical Analysis utilities for strategies.

Provides simple, lowercase indicator functions that return numpy arrays.
All indicators follow the industry-standard naming convention (TA-Lib, pandas-ta).

Contains:
- OHLC resampling (pandas-based, anchored to 09:30 market open)
- Heikin Ashi conversion
- Technical indicators (ema, sma, rsi, bbands, stochastic, macd, atr)
- Utility helpers (extract_ohlcv, last_valid)

Usage:
    from strategies._ta import resample_anchored_to_930, heikin_ashi
    from strategies._ta import ema, rsi, bbands, stochastic

    # Resample to 30min bars anchored to 09:30
    bars_30m = resample_anchored_to_930(bars_5m, source_minutes=5, target_minutes=30)

    # All indicators return numpy arrays
    ema_series = ema(closes, period=200)  # Returns array
    current_ema = ema_series[-1]  # Get last value
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import talib


# =============================================================================
# OHLC RESAMPLING
# =============================================================================

def resample_anchored_to_930(
    bars: list[dict],
    source_minutes: int,
    target_minutes: int
) -> list[dict]:
    """
    Resample OHLC bars to larger timeframe, anchored to 09:30 market open.

    Uses pandas built-in resampling for robustness and correctness.
    Matches TradingView behavior where bars align to 9:30 AM ET.
    Example: 30min bars → 9:30-10:00, 10:00-10:30, etc.

    Args:
        bars: List of OHLC dicts with 'date', 'open', 'high', 'low', 'close', 'volume'
        source_minutes: Source interval (e.g., 5) - not used, kept for compatibility
        target_minutes: Target interval in minutes (e.g., 30)

    Returns:
        List of resampled OHLC bars, oldest first

    Example:
        bars_30m = resample_anchored_to_930(bars_5m, source_minutes=5, target_minutes=30)
    """
    if not bars:
        return []

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(bars)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Build aggregation dict based on available columns
    agg_dict = {}
    if 'open' in df.columns:
        agg_dict['open'] = 'first'
    if 'high' in df.columns:
        agg_dict['high'] = 'max'
    if 'low' in df.columns:
        agg_dict['low'] = 'min'
    if 'close' in df.columns:
        agg_dict['close'] = 'last'
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    if not agg_dict:
        raise ValueError("DataFrame must contain at least one of: open, high, low, close, volume")

    # Anchor to 09:30 market open
    anchor = pd.Timestamp('1970-01-01 09:30:00')
    
    # If index is tz-aware, localize anchor to match
    if df.index.tz is not None:
        anchor = anchor.tz_localize(df.index.tz)

    timeframe = f"{target_minutes}min"

    # Resample using pandas (robust, handles edge cases)
    resampled = df.resample(timeframe, origin=anchor).agg(agg_dict).dropna()

    # Convert back to list of dicts
    result = []
    for timestamp, row in resampled.iterrows():
        bar = {
            'date': timestamp.isoformat(),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
        }
        if 'volume' in row:
            bar['volume'] = row['volume']
        result.append(bar)

    return result


# =============================================================================
# HEIKIN ASHI
# =============================================================================

def heikin_ashi(bars: list[dict]) -> list[dict]:
    """
    Convert OHLC bars to Heikin Ashi candles.

    Formulas:
        HA_Close = (O + H + L + C) / 4
        HA_Open  = (prev_HA_Open + prev_HA_Close) / 2
        HA_High  = max(H, HA_Open, HA_Close)
        HA_Low   = min(L, HA_Open, HA_Close)

    Args:
        bars: List of OHLC dicts

    Returns:
        List of Heikin Ashi candles with keys: date, ha_open, ha_high, ha_low, ha_close, volume
    """
    if not bars:
        return []

    ha = []

    for i, bar in enumerate(bars):
        o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
        ha_close = (o + h + l + c) / 4.0

        if i == 0:
            ha_open = (o + c) / 2.0
        else:
            ha_open = (ha[i-1]['ha_open'] + ha[i-1]['ha_close']) / 2.0

        ha.append({
            'date': bar.get('date', ''),
            'ha_open': ha_open,
            'ha_high': max(h, ha_open, ha_close),
            'ha_low': min(l, ha_open, ha_close),
            'ha_close': ha_close,
            'volume': bar.get('volume', 0)
        })

    return ha


def heikin_ashi_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized Heikin Ashi calculation for DataFrames.

    Same formulas as heikin_ashi() but operates on DataFrames for use
    in vectorized backtesting. The HA_Open calculation is inherently
    sequential (depends on previous bar), so we use a fast loop for that part.

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


def ha_color(ha_bars: list[dict], index: int = -1) -> str:
    """
    Get Heikin Ashi candle color.
    
    Returns:
        'green' if bullish (ha_close > ha_open), 'red' if bearish
    """
    bar = ha_bars[index]
    return 'green' if bar['ha_close'] > bar['ha_open'] else 'red'


def ha_consecutive(ha_bars: list[dict], color: str, count: int = 2) -> bool:
    """
    Check if last N HA candles are same color.
    
    Args:
        ha_bars: Heikin Ashi bars
        color: 'green' or 'red'
        count: Number of consecutive bars to check
        
    Returns:
        True if last `count` bars match `color`
    """
    if len(ha_bars) < count:
        return False
    
    for i in range(-count, 0):
        if ha_color(ha_bars, i) != color:
            return False
    return True


# =============================================================================
# INDICATORS (talib wrappers)
# =============================================================================

def ema(closes: list[float], period: int) -> np.ndarray:
    """Exponential Moving Average."""
    return talib.EMA(np.array(closes, dtype=float), timeperiod=period)


def sma(closes: list[float], period: int) -> np.ndarray:
    """Simple Moving Average."""
    return talib.SMA(np.array(closes, dtype=float), timeperiod=period)


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


def macd(
    closes: list[float],
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD (Moving Average Convergence Divergence).
    
    Returns:
        (macd, signal, histogram) arrays
    """
    return talib.MACD(
        np.array(closes, dtype=float),
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod
    )


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


def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14
) -> np.ndarray:
    """Average True Range."""
    return talib.ATR(
        np.array(highs, dtype=float),
        np.array(lows, dtype=float),
        np.array(closes, dtype=float),
        timeperiod=period
    )


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def extract_ohlcv(bars: list[dict]) -> tuple[list, list, list, list, list]:
    """
    Extract OHLCV arrays from bar list.
    
    Returns:
        (opens, highs, lows, closes, volumes)
    """
    opens = [b['open'] for b in bars]
    highs = [b['high'] for b in bars]
    lows = [b['low'] for b in bars]
    closes = [b['close'] for b in bars]
    volumes = [b.get('volume', 0) for b in bars]
    return opens, highs, lows, closes, volumes


def last_valid(arr: np.ndarray, offset: int = 0) -> float | None:
    """
    Get last valid (non-NaN) value from array.
    
    Args:
        arr: numpy array (may contain NaN)
        offset: 0 = last, 1 = second-to-last, etc.
        
    Returns:
        Value or None if all NaN
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) <= offset:
        return None
    return float(valid[-(offset + 1)])


