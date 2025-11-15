# 🤖 NeuroFlux Development Guidelines for Agentic Coding Agents

## Build/Test Commands

### Testing
- **Run all tests**: `python test_runner.py`
- **Run pytest suite**: `python -m pytest`
- **Run specific test file**: `python -m pytest src/tests/test_specific_file.py`
- **Run single test method**: `python -m pytest src/tests/test_specific_file.py::TestClass::test_method -v`
- **Run with coverage**: `python -m pytest --cov=src --cov-report=html`
- **Run integration tests**: `python -m pytest -m integration`
- **Run async tests**: `python -m pytest -m asyncio`

### Code Quality
- **Format code**: `black .`
- **Lint code**: `flake8 .`
- **Type check**: `mypy .`
- **Full quality check**: `black . && flake8 . && mypy .`

### Build & Deploy
- **Build check**: `./scripts/build.sh`
- **Deploy simple**: `./scripts/deploy_simple.sh`
- **Stop services**: `./scripts/stop.sh`
- **Health check**: `./scripts/healthcheck.sh`

## Code Style Guidelines

### Imports
```python
# Standard library imports first
import os
import sys
import json
from typing import Dict, Any, Optional, List

# Third-party imports second
import numpy as np
import pandas as pd
from termcolor import cprint
from flask import Flask

# Local imports last
from models.model_factory import ModelFactory
from config import config
```

### Naming Conventions
- **Functions/Methods/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_CASE`
- **Private methods**: `_leading_underscore`
- **Modules**: `snake_case.py`

### Error Handling
```python
try:
    result = risky_operation()
except SpecificException as e:
    cprint(f"❌ Specific error: {e}", "red")
    logger.error(f"Operation failed: {e}")
    return None
except Exception as e:
    cprint(f"❌ Unexpected error: {e}", "red")
    logger.exception("Unexpected error occurred")
    raise
```

### Type Hints
```python
from typing import Dict, Any, Optional, List

def process_data(data: Dict[str, Any], timeout: Optional[float] = None) -> List[Dict[str, Any]]:
    """Process data with optional timeout."""
    pass
```

### Documentation
```python
"""
🧠 NeuroFlux Component Description
Brief description of what this module/component does.

Built with love by Nyros Veil 🚀

Features:
- Feature 1
- Feature 2
- Feature 3
"""

def function_name(param1: Type, param2: Type) -> ReturnType:
    """Brief description of what function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        SpecificException: When something goes wrong
    """
```

### Async/Await Patterns
```python
import asyncio

async def async_operation(self) -> Dict[str, Any]:
    """Perform async operation with proper error handling."""
    try:
        result = await self.client.request_async(data)
        return result
    except Exception as e:
        cprint(f"❌ Async operation failed: {e}", "red")
        raise

def sync_wrapper(self) -> Dict[str, Any]:
    """Sync wrapper for async operations."""
    loop = event_loop_manager.get_loop()
    return loop.run_until_complete(self.async_operation())
```

## Repository Conventions

### Agent Architecture
- **Base Agent**: Extend `BaseAgent` for all new agents
- **Agent Registration**: Register agents with `AgentRegistry`
- **Communication**: Use `CommunicationBus` for inter-agent messaging
- **Status Tracking**: Implement proper `AgentStatus` lifecycle
- **Metrics**: Use `AgentMetrics` for performance tracking

### Error Context & Logging
- **Error Messages**: Include context and actionable information
- **Logging Levels**: Use appropriate log levels (INFO, WARNING, ERROR)
- **Color Coding**: Use `cprint` for terminal output with colors
- **Exception Chaining**: Preserve original exception context

### Configuration Management
- **Environment Variables**: Load via `python-dotenv`
- **Config Validation**: Implement `validate()` methods
- **Graceful Degradation**: Handle missing optional dependencies
- **Security**: Never log sensitive information

### File Structure
```
src/
├── agents/           # Agent implementations
├── analytics/        # Data analysis components
├── exchanges/        # Exchange integrations
├── models/          # LLM model abstractions
├── strategies/      # Trading strategies
└── tests/           # Unit and integration tests
```

### Performance Considerations
- **Memory Management**: Monitor memory usage in long-running processes
- **Async Operations**: Use async/await for I/O operations
- **Resource Limits**: Implement circuit breakers for external APIs
- **Caching**: Cache expensive operations appropriately

### Security Best Practices
- **API Keys**: Never commit secrets to repository
- **Input Validation**: Validate all external inputs
- **Error Messages**: Don't expose sensitive information in errors
- **Dependencies**: Keep dependencies updated and audited</content>
<parameter name="filePath">/Users/nyrosveil/Development/github/neuroflux/AGENTS.md