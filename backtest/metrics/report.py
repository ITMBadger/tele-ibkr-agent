# backtest/metrics/report.py
"""Generate reports and save backtest results."""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

from .calculator import BacktestResult
from backtest.config import BacktestConfig
from backtest.utils import format_df_for_csv, round_numeric_columns


def generate_report(
    result: BacktestResult,
    config: BacktestConfig,
) -> str:
    """
    Generate a text report of backtest results.

    Args:
        result: BacktestResult with all statistics
        config: Backtest configuration

    Returns:
        Formatted report string
    """
    stats = result.stats_total

    lines = [
        "=" * 60,
        "BACKTEST RESULTS",
        "=" * 60,
        "",
        "Configuration:",
        f"  Symbols:        {result.symbol}",
        f"  Strategy:       {config.strategy}",
        f"  Period:         {config.start_date} to {config.end_date}",
        f"  Initial Capital: ${config.initial_capital:,.2f}",
        "",
        "-" * 60,
        "Performance Summary:",
        "-" * 60,
        f"  Final Equity:   ${stats.final_equity:,.2f}",
        f"  Total P&L:      ${stats.total_pnl:,.2f}",
        f"  Total Return:   {stats.total_pnl_pct:.2f}%",
        f"  CAGR:           {stats.cagr:.2f}%",
        "",
        "-" * 60,
        "Trade Statistics:",
        "-" * 60,
        f"  Total Trades:   {stats.total_trades}",
        f"  Winning Trades: {stats.winning_trades} ({stats.win_rate:.1f}%)",
        f"  Losing Trades:  {stats.losing_trades}",
        f"  Avg Winner:     ${stats.avg_win:,.2f}",
        f"  Avg Loser:      ${stats.avg_loss:,.2f}",
        f"  Largest Winner: ${stats.largest_winner:,.2f}",
        f"  Largest Loser:  ${stats.largest_loser:,.2f}",
        "",
        "-" * 60,
        "Risk Metrics:",
        "-" * 60,
        f"  Profit Factor:  {stats.profit_factor:.2f}",
        f"  Expectancy:     ${stats.expectancy:,.2f}",
        f"  Max Drawdown:   ${stats.max_drawdown:,.2f} ({stats.max_drawdown_pct:.2f}%)",
        f"  Sharpe Ratio:   {stats.sharpe_ratio:.2f}",
        f"  Sortino Ratio:  {stats.sortino_ratio:.2f}",
        f"  Calmar Ratio:   {stats.calmar_ratio:.2f}",
        "",
        "-" * 60,
        "Trade Duration & Streaks:",
        "-" * 60,
        f"  Avg Duration:           {stats.avg_trade_duration_minutes:.1f} minutes",
        f"  Max Consecutive Wins:   {stats.consecutive_wins}",
        f"  Max Consecutive Losses: {stats.consecutive_losses}",
        "",
    ]

    # Add direction-specific stats if available
    if result.stats_long.total_trades > 0:
        lines.extend([
            "-" * 60,
            "Long Trades:",
            "-" * 60,
            f"  Trades:       {result.stats_long.total_trades}",
            f"  Win Rate:     {result.stats_long.win_rate:.1f}%",
            f"  Total P&L:    ${result.stats_long.total_pnl:,.2f}",
            f"  Profit Factor: {result.stats_long.profit_factor:.2f}",
            "",
        ])

    if result.stats_short.total_trades > 0:
        lines.extend([
            "-" * 60,
            "Short Trades:",
            "-" * 60,
            f"  Trades:       {result.stats_short.total_trades}",
            f"  Win Rate:     {result.stats_short.win_rate:.1f}%",
            f"  Total P&L:    ${result.stats_short.total_pnl:,.2f}",
            f"  Profit Factor: {result.stats_short.profit_factor:.2f}",
            "",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)


def save_results(
    results_dir: Path | str,
    config: BacktestConfig,
    result: Optional[BacktestResult],
    signals: List[Dict[str, Any]] | None = None,
) -> Path:
    """
    Save all backtest results to a directory.

    Args:
        results_dir: Directory to save results
        config: Backtest configuration
        result: BacktestResult with trades, equity, and stats
        signals: Optional list of signals

    Returns:
        Path to results directory
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if result is None:
        print(f"  No results to save")
        return results_dir

    # Save trades
    trades_path = results_dir / "trades.csv"
    if result.trades:
        trades_df = pd.DataFrame([t.to_dict() for t in result.trades])
        trades_df = format_df_for_csv(trades_df)
        trades_df.to_csv(trades_path, index=False)
        print(f"  Trades saved: {trades_path}")
    else:
        pd.DataFrame().to_csv(trades_path, index=False)

    # Save signals if provided
    if signals:
        signals_path = results_dir / "signals.csv"
        signals_df = pd.DataFrame(signals)
        signals_df = round_numeric_columns(signals_df, 3)
        signals_df.to_csv(signals_path, index=False, na_rep="")
        print(f"  Signals saved: {signals_path}")

    # Save text report
    report_path = results_dir / "report.txt"
    report = generate_report(result, config)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    # Save config summary
    config_path = results_dir / "config.txt"
    config_lines = [
        f"Strategy: {config.strategy}",
        f"Symbols: {', '.join(config.symbols)}",
        f"Period: {config.start_date} to {config.end_date}",
        f"Initial Capital: ${config.initial_capital:,.2f}",
        f"Slippage: {config.slippage_pct * 100:.2f}%",
        f"Commission: ${config.commission_per_trade}",
        f"Take Profit/Stop Loss: (from strategy class)",
    ]
    with open(config_path, "w") as f:
        f.write("\n".join(config_lines))

    print(f"  Results saved to: {results_dir}")

    return results_dir


def print_summary(result: BacktestResult):
    """Print a brief summary of results to console."""
    stats = result.stats_total

    # Color codes for terminal
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    pnl_color = GREEN if stats.total_pnl >= 0 else RED

    print("\n" + "=" * 50)
    print("BACKTEST SUMMARY")
    print("=" * 50)
    print(f"Total Trades:   {stats.total_trades}")
    print(f"Win Rate:       {stats.win_rate:.1f}%")
    print(f"Profit Factor:  {stats.profit_factor:.2f}")
    print(f"Total Return:   {pnl_color}{stats.total_pnl_pct:+.2f}%{RESET}")
    print(f"Max Drawdown:   {RED}{stats.max_drawdown_pct:.2f}%{RESET}")
    print(f"Sharpe Ratio:   {stats.sharpe_ratio:.2f}")
    print(f"Final Equity:   {pnl_color}${stats.final_equity:,.2f}{RESET}")

    # Show long/short breakdown if both exist
    if result.stats_long.total_trades > 0 and result.stats_short.total_trades > 0:
        print("-" * 50)
        long_color = GREEN if result.stats_long.total_pnl >= 0 else RED
        short_color = GREEN if result.stats_short.total_pnl >= 0 else RED
        print(f"Long:  {result.stats_long.total_trades} trades, {long_color}${result.stats_long.total_pnl:+,.2f}{RESET}")
        print(f"Short: {result.stats_short.total_trades} trades, {short_color}${result.stats_short.total_pnl:+,.2f}{RESET}")

    print("=" * 50)
