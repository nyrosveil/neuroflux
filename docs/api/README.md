# 🧠 NeuroFlux API Documentation

## Overview

Welcome to the comprehensive API documentation for NeuroFlux, the multi-agent trading system with neuro-flux adaptation. This documentation provides detailed technical specifications for all public APIs, methods, and interfaces.

## 📋 Documentation Structure

### Core Framework APIs
- **[Base Agent Framework](base_agent.md)** - Abstract base class for all NeuroFlux agents
- **[Task Orchestrator](task_orchestrator.md)** - Dynamic task assignment and coordination
- **[Model Factory](model_factory.md)** - LLM provider abstraction with flux adaptation
- **[Neural Swarm Network](neural_swarm_network.md)** - Inter-agent communication and learning

### Agent Category APIs
- **[Trading Agent](agents/trading.md)** - Core trading functionality with neuro-flux decisions
- **[Risk Agent](agents/risk.md)** - Risk management and circuit breaker functionality
- **[Market Analysis](agents/analysis.md)** - Research, sentiment, and technical analysis
- **[Specialized Agents](agents/specialized.md)** - Funding, liquidation, whale tracking
- **[Advanced Agents](agents/advanced.md)** - Swarm intelligence, RBI, sniping

### Exchange Integration
- **[Base Exchange](exchanges/base_exchange.md)** - Unified exchange interface
- **[HyperLiquid Adapter](exchanges/hyperliquid.md)** - Perpetual futures integration
- **[Exchange Manager](exchanges/manager.md)** - Multi-exchange coordination

### Supporting Components
- **[Communication Bus](communication_bus.md)** - Inter-agent messaging
- **[Agent Registry](agent_registry.md)** - Agent discovery and management
- **[Conflict Resolution](conflict_resolution.md)** - Decision conflict management

## 🚀 Quick Start

```python
from neuroflux.src.agents.base_agent import BaseAgent
from neuroflux.src.models.model_factory import ModelFactory

# Create a flux-optimized model
model = ModelFactory.create_flux_optimized_model('claude', flux_level=0.7)

# Initialize a custom agent
class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_id="my_agent", flux_level=0.7)

    def _initialize_agent(self) -> bool:
        return True

    def _execute_agent_cycle(self):
        # Your agent logic here
        pass

agent = MyAgent()
agent.initialize()
agent.start()
```

## 📊 API Standards

### Method Signatures
All API methods follow consistent patterns:
- **Return Types**: Explicit type hints for all return values
- **Parameter Validation**: Input validation with descriptive error messages
- **Error Handling**: Comprehensive exception handling with meaningful messages
- **Documentation**: Complete docstrings with examples

### Response Formats
API responses follow standardized structures:
```python
{
    "success": bool,           # Operation success status
    "data": Any,              # Response payload
    "error": Optional[str],   # Error message if applicable
    "metadata": Dict          # Additional context information
}
```

### Error Handling
```python
try:
    result = api_method(param1, param2)
except ValueError as e:
    print(f"Invalid parameter: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔍 Navigation

- **🔗 Cross-References**: All APIs include links to related components
- **📖 Examples**: Practical code examples for common use cases
- **⚡ Performance**: Performance considerations and optimization tips
- **🔒 Security**: Security best practices and considerations

## 📈 Version Information

- **Current Version**: v1.0.0
- **API Stability**: All documented APIs are stable for production use
- **Deprecation Policy**: Deprecated APIs will be marked with warnings and migration guides

## 🤝 Contributing

When adding new APIs:
1. Follow the established documentation format
2. Include comprehensive examples
3. Add cross-references to related APIs
4. Update this index file

---

*Built with ❤️ by Nyros Veil* | [GitHub](https://github.com/nyrosveil/neuroflux) | [Issues](https://github.com/nyrosveil/neuroflux/issues)