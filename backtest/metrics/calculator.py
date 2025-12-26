# backtest/metrics/calculator.py
"""Calculate performance metrics from backtest results."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from backtest.execution.position import Trade, TradeDirection


@dataclass
class DirectionStats:
    """Statistics for a single direction (long, short, or total)."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    cagr: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    final_equity: float = 0.0
    peak_equity: float = 0.0

    # Additional stats
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    avg_trade_duration_minutes: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2),
            "cagr": round(self.cagr, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "calmar_ratio": round(self.calmar_ratio, 2),
            "final_equity": round(self.final_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "largest_winner": round(self.largest_winner, 2),
            "largest_loser": round(self.largest_loser, 2),
            "avg_trade_duration_minutes": round(self.avg_trade_duration_minutes, 2),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
        }


@dataclass
class BacktestResult:
    """Container for backtest results - compatible with dashboard visualization."""

    config: Any  # BacktestConfig
    trades: List[Trade]
    equity_curve: pd.DataFrame

    # Direction-specific equity curves
    equity_curve_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve_short: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Direction-specific stats
    stats_total: DirectionStats = field(default_factory=DirectionStats)
    stats_long: DirectionStats = field(default_factory=DirectionStats)
    stats_short: DirectionStats = field(default_factory=DirectionStats)

    # Monthly returns
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns_long: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns_short: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Metadata
    symbol: str = ""
    timeframe: str = ""

    # Top-level convenience fields (mirrors stats_total)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    final_equity: float = 0.0
    peak_equity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_trades": self.total_trades,
            "stats_total": self.stats_total.to_dict(),
            "stats_long": self.stats_long.to_dict(),
            "stats_short": self.stats_short.to_dict(),
        }


def calculate_direction_stats(
    trades: List[Trade],
    equity_curve: pd.DataFrame,
    starting_capital: float,
) -> DirectionStats:
    """
    Calculate statistics for a set of trades.

    Args:
        trades: List of Trade objects
        equity_curve: Equity curve DataFrame with 'Equity' column
        starting_capital: Initial capital

    Returns:
        DirectionStats with all calculated metrics
    """
    stats = DirectionStats()
    stats.final_equity = starting_capital

    if not trades:
        return stats

    pnls = np.array([t.pnl for t in trades], dtype=float)

    stats.total_trades = len(trades)
    stats.winning_trades = int(np.sum(pnls > 0))
    stats.losing_trades = int(np.sum(pnls <= 0))
    stats.win_rate = (stats.winning_trades / stats.total_trades) * 100 if stats.total_trades > 0 else 0.0

    stats.total_pnl = float(pnls.sum())
    stats.total_pnl_pct = (stats.total_pnl / starting_capital) * 100

    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    stats.avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    stats.avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    stats.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    w_pct = stats.win_rate / 100.0
    stats.expectancy = (w_pct * stats.avg_win) + ((1.0 - w_pct) * stats.avg_loss)

    stats.largest_winner = float(pnls.max()) if len(pnls) > 0 else 0.0
    stats.largest_loser = float(pnls.min()) if len(pnls) > 0 else 0.0

    # Duration
    durations = [t.duration_minutes for t in trades]
    stats.avg_trade_duration_minutes = float(np.mean(durations)) if durations else 0.0

    # Streaks
    stats.consecutive_wins, stats.consecutive_losses = _calculate_streaks(trades)

    # Equity curve metrics
    if not equity_curve.empty and "Equity" in equity_curve.columns:
        eq = equity_curve["Equity"].values
        stats.final_equity = float(eq[-1])
        stats.peak_equity = float(eq.max())

        # Drawdown
        stats.max_drawdown, stats.max_drawdown_pct = _calculate_drawdown_metrics(eq)

        # Returns for Sharpe/Sortino
        returns = np.diff(eq) / eq[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) > 1:
            # Estimate bars per year (5min bars = 252 * 78)
            bars_per_year = 252 * 78 if len(eq) > 1000 else 252

            std_returns = float(np.std(returns))
            if std_returns > 0:
                mean_returns = float(np.mean(returns))
                stats.sharpe_ratio = (mean_returns / std_returns) * np.sqrt(bars_per_year)

                downside = returns[returns < 0]
                if len(downside) > 0:
                    downside_std = float(np.std(downside))
                    if downside_std > 0:
                        stats.sortino_ratio = (mean_returns / downside_std) * np.sqrt(bars_per_year)

        # CAGR
        if len(equity_curve.index) >= 2:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days > 0:
                stats.cagr = ((stats.final_equity / starting_capital) ** (365.0 / days) - 1.0) * 100

        # Calmar
        if stats.max_drawdown_pct > 0:
            stats.calmar_ratio = stats.cagr / stats.max_drawdown_pct
    else:
        stats.final_equity = starting_capital + stats.total_pnl

    return stats


def calculate_monthly_returns(equity_series: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns from equity series.

    Args:
        equity_series: Series with DatetimeIndex and equity values

    Returns:
        DataFrame with Return, Year, Month columns
    """
    if equity_series.empty:
        return pd.DataFrame()

    # Calculate returns
    returns = equity_series.pct_change().fillna(0)

    # Resample to monthly
    monthly_rets = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100

    df = monthly_rets.to_frame("Return")
    df["Year"] = df.index.year
    df["Month"] = df.index.month_name().str[:3]

    return df


def build_separate_equity_curve(
    trades: List[Trade],
    n_bars: int,
    index: pd.DatetimeIndex,
    starting_capital: float,
) -> pd.DataFrame:
    """
    Build equity curve for a subset of trades (e.g., only longs or only shorts).

    Args:
        trades: List of Trade objects
        n_bars: Total number of bars
        index: DatetimeIndex for the curve
        starting_capital: Initial capital

    Returns:
        DataFrame with Equity column
    """
    if not trades:
        return pd.DataFrame()

    equity = np.full(n_bars, starting_capital, dtype=np.float64)
    cumulative_pnl = 0.0

    for trade in trades:
        exit_idx = trade._exit_idx
        cumulative_pnl += trade.pnl
        if exit_idx < n_bars:
            equity[exit_idx:] = starting_capital + cumulative_pnl

    return pd.DataFrame({"Equity": equity}, index=index)


def build_backtest_result(
    config: Any,
    trades: List[Trade],
    equity_curve: pd.DataFrame,
    starting_capital: float,
    symbol: str = "",
    timeframe: str = "",
) -> BacktestResult:
    """
    Build complete BacktestResult from simulation output.

    Args:
        config: BacktestConfig
        trades: List of Trade objects
        equity_curve: Equity curve DataFrame with DatetimeIndex
        starting_capital: Initial capital
        symbol: Symbol name
        timeframe: Timeframe string

    Returns:
        BacktestResult with all statistics calculated
    """
    # Split trades by direction
    long_trades = [t for t in trades if t.direction == TradeDirection.LONG]
    short_trades = [t for t in trades if t.direction == TradeDirection.SHORT]

    # Build direction-specific equity curves
    n_bars = len(equity_curve)
    index = equity_curve.index

    equity_curve_long = build_separate_equity_curve(long_trades, n_bars, index, starting_capital)
    equity_curve_short = build_separate_equity_curve(short_trades, n_bars, index, starting_capital)

    # Calculate stats for each direction
    stats_total = calculate_direction_stats(trades, equity_curve, starting_capital)
    stats_long = calculate_direction_stats(long_trades, equity_curve_long, starting_capital)
    stats_short = calculate_direction_stats(short_trades, equity_curve_short, starting_capital)

    # Calculate monthly returns
    monthly_returns = pd.DataFrame()
    monthly_returns_long = pd.DataFrame()
    monthly_returns_short = pd.DataFrame()

    if not equity_curve.empty and "Equity" in equity_curve.columns:
        monthly_returns = calculate_monthly_returns(equity_curve["Equity"])

    if not equity_curve_long.empty and "Equity" in equity_curve_long.columns:
        monthly_returns_long = calculate_monthly_returns(equity_curve_long["Equity"])

    if not equity_curve_short.empty and "Equity" in equity_curve_short.columns:
        monthly_returns_short = calculate_monthly_returns(equity_curve_short["Equity"])

    # Build result
    result = BacktestResult(
        config=config,
        trades=trades,
        equity_curve=equity_curve,
        equity_curve_long=equity_curve_long,
        equity_curve_short=equity_curve_short,
        stats_total=stats_total,
        stats_long=stats_long,
        stats_short=stats_short,
        monthly_returns=monthly_returns,
        monthly_returns_long=monthly_returns_long,
        monthly_returns_short=monthly_returns_short,
        symbol=symbol,
        timeframe=timeframe,
        # Mirror top-level stats
        total_trades=stats_total.total_trades,
        winning_trades=stats_total.winning_trades,
        losing_trades=stats_total.losing_trades,
        win_rate=stats_total.win_rate,
        total_pnl=stats_total.total_pnl,
        total_pnl_pct=stats_total.total_pnl_pct,
        avg_win=stats_total.avg_win,
        avg_loss=stats_total.avg_loss,
        profit_factor=stats_total.profit_factor,
        expectancy=stats_total.expectancy,
        cagr=stats_total.cagr,
        max_drawdown=stats_total.max_drawdown,
        max_drawdown_pct=stats_total.max_drawdown_pct,
        sharpe_ratio=stats_total.sharpe_ratio,
        sortino_ratio=stats_total.sortino_ratio,
        calmar_ratio=stats_total.calmar_ratio,
        final_equity=stats_total.final_equity,
        peak_equity=stats_total.peak_equity,
    )

    return result


def _calculate_streaks(trades: List[Trade]) -> tuple:
    """Calculate max consecutive wins and losses."""
    if not trades:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trades:
        if trade.is_winner:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)

    return max_wins, max_losses


def _calculate_drawdown_metrics(equity: np.ndarray) -> tuple:
    """Calculate max drawdown and max drawdown percentage."""
    rolling_max = np.maximum.accumulate(equity)
    drawdown = equity - rolling_max
    drawdown_pct = np.divide(drawdown, rolling_max, where=rolling_max > 0) * 100
    max_dd = float(abs(drawdown.min()))
    max_dd_pct = float(abs(drawdown_pct.min()))
    return max_dd, max_dd_pct
