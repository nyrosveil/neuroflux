# NeuroFlux Agent Development Guide

## Build/Lint/Test Commands
- **Single test**: `python test_trading_agent_simple.py` (or any test_*.py file)
- **All tests**: `for test in test_*.py; do echo "Running $test"; python "$test"; done`
- **React build**: `cd dashboard && npm run build`
- **Full deployment**: `./start.sh` (builds React + starts Flask server)

## Code Style Guidelines

### Imports
```python
# Standard library first
import os, sys, asyncio
from typing import Dict, List, Optional

# Third-party second
import pandas as pd
from termcolor import cprint

# Local imports last
from src.config import config
from .base_agent import BaseAgent
```

### Naming & Types
- **Functions/variables**: `snake_case` (e.g., `process_data()`, `market_data`)
- **Classes**: `PascalCase` (e.g., `TradingAgent`, `MarketData`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES = 3`)
- **Type hints**: Required for all functions (e.g., `def func(data: Dict[str, Any]) -> Optional[str]`)

### Error Handling
```python
try:
    result = await risky_operation()
except ConnectionError as e:
    cprint(f"⚠️  Connection failed: {e}", "yellow")
except Exception as e:
    cprint(f"💥 Error: {e}", "red")
    raise  # Re-raise critical errors
```

### Formatting
- **Line length**: < 100 characters
- **Indentation**: 4 spaces (no tabs)
- **Docstrings**: Required for all classes/functions
- **Async patterns**: Use `async/await` for I/O operations

### File Structure
- Keep files < 800 lines (split if longer)
- Store agent outputs in `src/data/[agent_name]/`
- Use absolute imports: `from src.module import Class`