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
from strategies._trading_mech import BaseStrategy

# Import all strategy classes
from strategies.ema_only import EMAOnly
from strategies.ha_mtf_stoch import HAMTFStochEMA
from strategies.strat_multi_toggle import StratMultiToggle


# === STRATEGY REGISTRY ===
# Add new strategies here - this is the ONLY place to register them

STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "1": EMAOnly,
    "2": HAMTFStochEMA,
    "999": StratMultiToggle,
}


def get_strategy(strategy_id: str) -> Type[BaseStrategy] | None:
    """
    Get a strategy class by ID.
    
    """
    return STRATEGY_REGISTRY.get(strategy_id)


def list_strategies() -> list[Type[BaseStrategy]]:
    """
    Get all available strategy classes.
    
    """
    return [
        cls for _, cls in 
        sorted(STRATEGY_REGISTRY.items(), key=lambda x: int(x[0]))
    ]


def create_strategy(strategy_id: str, symbol: str, tiingo: Any) -> BaseStrategy | None:
    """
    Create a strategy instance.
    
    """
    strategy_class = get_strategy(strategy_id)
    if strategy_class is None:
        return None
    return strategy_class(symbol, tiingo)


def format_strategy_list() -> str:
    """
    Format all strategies as a readable list for Telegram display.
    
    """
    lines = ["📋 **Available Strategies**", ""]
    
    for strategy_class in list_strategies():
        lines.append(strategy_class.info())
        lines.append("")
    
    lines.append("Use: 'apply strategy [ID] to [SYMBOL]'")
    
    return "\n".join(lines)

