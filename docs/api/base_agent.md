# 🧠 Base Agent Framework API

## Overview

The `BaseAgent` class provides the foundation for all NeuroFlux agents, implementing neuro-flux enhanced lifecycle management, performance monitoring, and standardized interfaces.

**Location:** `src/agents/base_agent.py`

**Inheritance:** Abstract base class for all agent implementations

---

## Class Hierarchy

```python
BaseAgent (Abstract)
├── TradingAgent
├── RiskAgent
├── ResearchAgent
├── SentimentAgent
├── SwarmAgent
├── RBI_Agent
└── ... (48+ agent types)
```

---

## Core Methods

### Lifecycle Management

#### `initialize() → bool`
**Description:** Initialize the agent and its dependencies

**Returns:**
- `bool`: True if initialization successful

**Raises:**
- `Exception`: If initialization fails

**Example:**
```python
agent = MyCustomAgent(agent_id="my_agent")
success = agent.initialize()
if success:
    print("Agent initialized successfully")
```

#### `start() → bool`
**Description:** Begin agent execution with threading

**Returns:**
- `bool`: True if started successfully

**Raises:**
- `Exception`: If agent is not in READY state

**Example:**
```python
if agent.initialize():
    success = agent.start()
    if success:
        print("Agent started in background thread")
```

#### `stop(timeout=10.0) → bool`
**Description:** Graceful shutdown with timeout

**Parameters:**
- `timeout` (float, optional): Maximum wait time in seconds (default: 10.0)

**Returns:**
- `bool`: True if stopped successfully

**Example:**
```python
agent.stop(timeout=5.0)  # Wait up to 5 seconds
```

#### `pause() → bool`
**Description:** Suspend execution temporarily

**Returns:**
- `bool`: True if paused successfully

**Example:**
```python
agent.pause()  # Agent enters PAUSED state
```

#### `resume() → bool`
**Description:** Resume from paused state

**Returns:**
- `bool`: True if resumed successfully

**Example:**
```python
agent.resume()  # Agent returns to RUNNING state
```

---

### Neuro-Flux Integration

#### `update_flux_level(new_level: float) → None`
**Description:** Dynamically adjust agent's flux sensitivity

**Parameters:**
- `new_level` (float): New flux level (0.0 to 1.0)

**Example:**
```python
# Increase flux sensitivity during high volatility
agent.update_flux_level(0.8)
```

#### `generate_response(system_prompt: str, user_content: str, **kwargs) → Dict[str, Any]`
**Description:** Generate LLM response with neuro-flux adaptation

**Parameters:**
- `system_prompt` (str): System instruction prompt
- `user_content` (str): User message content
- `temperature` (float, optional): Override temperature
- `**kwargs`: Additional model parameters

**Returns:**
- `Dict[str, Any]`: Response containing content, tokens, etc.

**Example:**
```python
response = agent.generate_response(
    system_prompt="You are a trading analyst",
    user_content="Analyze BTC price action",
    temperature=0.7
)
print(response['content'])
```

#### `add_status_callback(callback: Callable[[AgentStatus], None]) → None`
**Description:** Register callback for status change notifications

**Parameters:**
- `callback` (Callable): Function accepting AgentStatus enum

**Example:**
```python
def on_status_change(status: AgentStatus):
    print(f"Agent status changed to: {status.value}")

agent.add_status_callback(on_status_change)
```

---

### Performance Monitoring

#### `get_status() → Dict[str, Any]`
**Description:** Get comprehensive agent status information

**Returns:**
- `Dict[str, Any]`: Complete status including metrics, config, uptime

**Response Structure:**
```python
{
    "agent_id": "my_agent",
    "agent_type": "MyCustomAgent",
    "status": "running",
    "priority": "medium",
    "flux_level": 0.7,
    "model_provider": "claude",
    "is_running": true,
    "metrics": {
        "start_time": "2024-01-01T00:00:00",
        "total_requests": 150,
        "successful_requests": 145,
        "failed_requests": 5,
        "total_tokens_used": 25000,
        "average_response_time": 2.3,
        "error_count": 5,
        "last_error": null,
        "flux_level": 0.7,
        "model_provider": "claude",
        "success_rate": 96.7,
        "uptime_seconds": 3600
    },
    "config": {...},
    "uptime": 3600
}
```

---

### Data Persistence

#### `save_state(filepath: Optional[str] = None) → bool`
**Description:** Serialize agent state to file

**Parameters:**
- `filepath` (str, optional): Custom save path

**Returns:**
- `bool`: True if saved successfully

**Example:**
```python
agent.save_state("my_agent_backup.json")
```

#### `load_state(filepath: Optional[str] = None) → bool`
**Description:** Restore agent state from file

**Parameters:**
- `filepath` (str, optional): Custom load path

**Returns:**
- `bool`: True if loaded successfully

**Example:**
```python
agent.load_state("my_agent_backup.json")
```

---

## Abstract Methods

### `_initialize_agent() → bool`
**Description:** Agent-specific initialization logic

**Returns:**
- `bool`: True if initialization successful

**Implementation Required:** All concrete agents must implement this method

### `_execute_agent_cycle()`
**Description:** Main agent execution logic

**Implementation Required:** All concrete agents must implement this method

### `_cleanup_agent()`
**Description:** Agent-specific cleanup logic

**Implementation Required:** All concrete agents must implement this method

---

## Agent Status Enum

```python
class AgentStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
```

---

## Agent Priority Enum

```python
class AgentPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

---

## Agent Metrics Class

```python
@dataclass
class AgentMetrics:
    start_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    average_response_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    flux_level: float = 0.5
    model_provider: str = "unknown"

    def record_request(self, success: bool, tokens_used: int = 0,
                      response_time: float = 0.0) → None:
        """Record API request metrics"""

    def get_success_rate(self) → float:
        """Calculate success rate percentage"""
```

---

## Usage Examples

### Basic Agent Implementation

```python
from src.agents.base_agent import BaseAgent, AgentStatus
from typing import Dict, Any

class ExampleAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, flux_level=0.5)

    def _initialize_agent(self) -> bool:
        # Custom initialization logic
        self.data_store = {}
        return True

    def _execute_agent_cycle(self):
        # Main agent logic
        try:
            # Perform agent tasks
            result = self.generate_response(
                system_prompt="You are a helpful assistant",
                user_content="What is the current market sentiment?"
            )

            # Store result
            self.data_store['last_analysis'] = result['content']

        except Exception as e:
            self.logger.error(f"Cycle execution failed: {e}")

    def _cleanup_agent(self):
        # Cleanup resources
        self.data_store.clear()

# Usage
agent = ExampleAgent("example_agent")
agent.initialize()
agent.start()

# Monitor status
status = agent.get_status()
print(f"Agent status: {status['status']}")

# Graceful shutdown
agent.stop()
```

### Flux-Adaptive Agent

```python
class FluxAdaptiveAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, flux_level=0.3)

    def _initialize_agent(self) -> bool:
        self.conservative_mode = False
        return True

    def _execute_agent_cycle(self):
        # Adapt behavior based on flux level
        if self.flux_level > 0.7:
            # High flux: be more conservative
            self._execute_conservative_strategy()
        else:
            # Normal flux: standard operation
            self._execute_normal_strategy()

    def _execute_conservative_strategy(self):
        # Reduced risk operations
        pass

    def _execute_normal_strategy(self):
        # Standard operations
        pass

    def _cleanup_agent(self):
        pass
```

---

## Error Handling

```python
try:
    agent = MyAgent("test_agent")
    agent.initialize()
    agent.start()

except Exception as e:
    print(f"Agent initialization failed: {e}")

finally:
    if 'agent' in locals():
        agent.stop()
```

---

## Performance Considerations

- **Threading:** Agents run in background threads
- **Resource Management:** Implement proper cleanup in `_cleanup_agent()`
- **Flux Adaptation:** Higher flux levels may increase API costs
- **Memory Usage:** Large data stores should be managed carefully
- **Error Recovery:** Implement exponential backoff in cycle logic

---

## Related APIs

- **[Model Factory](model_factory.md)** - LLM provider management
- **[Task Orchestrator](task_orchestrator.md)** - Agent coordination
- **[Trading Agent](agents/trading.md)** - Trading implementation
- **[Risk Agent](agents/risk.md)** - Risk management implementation

---

*Built with ❤️ by Nyros Veil* | [Back to API Index](../README.md)