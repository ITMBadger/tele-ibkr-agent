# strategies/__init__.py
"""
strategies/ - Self-contained trading strategies.

Each strategy is a complete unit with ALL parameters fixed inside.
The agent only needs: activate_strategy(strategy_id, symbol)

Usage:
    from strategies import get_strategy, list_strategies, create_strategy
    
    # List available strategies
    for strategy_class in list_strategies():
        print(strategy_class.info())
    
    # Create a strategy instance for a symbol
    strategy = create_strategy("1", "QQQ", tiingo)
    await strategy.execute()
"""

from typing import Type, Any
from strategies._base import BaseStrategy

# Import all strategy classes
from strategies.ema_200_long import EMA200Long
from strategies.ema_50_aggressive import EMA50Aggressive
from strategies.rsi_bounce import RSIOversoldBounce
from strategies.ema_20_scalp import EMA20Scalp
from strategies.ema_100_conservative import EMA100Conservative
from strategies.ha_mtf_stoch import HAMTFStoch


# === STRATEGY REGISTRY ===
# Add new strategies here - this is the ONLY place to register them

STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "1": EMA200Long,
    "2": EMA50Aggressive,
    "3": RSIOversoldBounce,
    "4": EMA20Scalp,
    "5": EMA100Conservative,
    "6": HAMTFStoch,
}


def get_strategy(strategy_id: str) -> Type[BaseStrategy] | None:
    """
    Get a strategy class by ID.
    
    Args:
        strategy_id: Strategy ID (e.g., "1", "2", "3")
        
    Returns:
        Strategy class if found, None otherwise
    """
    return STRATEGY_REGISTRY.get(strategy_id)


def list_strategies() -> list[Type[BaseStrategy]]:
    """
    Get all available strategy classes.
    
    Returns:
        List of strategy classes sorted by ID
    """
    return [
        cls for _, cls in 
        sorted(STRATEGY_REGISTRY.items(), key=lambda x: int(x[0]))
    ]


def create_strategy(strategy_id: str, symbol: str, tiingo: Any) -> BaseStrategy | None:
    """
    Create a strategy instance.
    
    Args:
        strategy_id: Strategy ID
        symbol: Stock symbol to trade
        tiingo: TiingoService instance
        
    Returns:
        Strategy instance if found, None otherwise
    """
    strategy_class = get_strategy(strategy_id)
    if strategy_class is None:
        return None
    return strategy_class(symbol, tiingo)


def format_strategy_list() -> str:
    """
    Format all strategies as a readable list for Telegram display.
    
    Returns:
        Multi-line string with all strategies
    """
    lines = ["📋 **Available Strategies**", ""]
    
    for strategy_class in list_strategies():
        lines.append(strategy_class.info())
        lines.append("")
    
    lines.append("Use: 'apply strategy [ID] to [SYMBOL]'")
    
    return "\n".join(lines)

