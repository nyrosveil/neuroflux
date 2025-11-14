"""
NeuroFlux Agents Package
Multi-agent trading system components

Built with love by Nyros Veil 🚀
"""

# Lazy imports to avoid import errors during package initialization
__all__ = [
    'trading_agent',
    'risk_agent',
    'sentiment_agent',
    'research_agent',
    'chartanalysis_agent',
    'funding_agent',
    'liquidation_agent',
    'whale_agent',
    'coingecko_agent',
    'copybot_agent',
    'sniper_agent',
    'strategy_agent',
    'swarm_agent',
    'websearch_agent',
    'rbi_agent',
    'backtest_runner',
    'chat_agent',
    'ml_prediction_agent'
]

def __getattr__(name):
    """Lazy import for agents"""
    if name in __all__:
        try:
            import importlib
            module = importlib.import_module(f'.{name}', package=__name__)
            return module
        except ImportError as e:
            # Return a dummy module that won't break imports
            import types
            dummy_module = types.ModuleType(name)
            dummy_module.__file__ = f"<dummy {name}>"
            dummy_module.__name__ = name
            return dummy_module
    raise AttributeError(f"Agent {name} not found")