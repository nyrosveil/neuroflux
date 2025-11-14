# 🧠 Neural Swarm Network API

## Overview

The `NeuralSwarmNetwork` class implements inter-agent communication and learning through neural network-inspired connections, enabling emergent collective intelligence and adaptive behavior patterns.

**Location:** `src/swarm_intelligence/neural_swarm_network.py`

**Key Features:** Synaptic connections, signal propagation, knowledge sharing, adaptive topology

---

## Class Architecture

```python
NeuralSwarmNetwork
├── Network Management
├── Signal Propagation
├── Knowledge Sharing
├── Topology Adaptation
└── Performance Analytics
```

---

## Core Methods

### Network Initialization

#### `__init__(network_id: str = "default_swarm") → None`
**Description:** Initialize a new neural swarm network

**Parameters:**
- `network_id` (str, optional): Unique identifier for the network

**Example:**
```python
from src.swarm_intelligence.neural_swarm_network import NeuralSwarmNetwork

network = NeuralSwarmNetwork(network_id="trading_swarm")
```

### Agent Management

#### `add_agent(agent_id: str, activation_threshold: float = 0.5) → SwarmNeuron`
**Description:** Add an agent as a neuron to the network

**Parameters:**
- `agent_id` (str): Unique agent identifier
- `activation_threshold` (float, optional): Neuron firing threshold (default: 0.5)

**Returns:**
- `SwarmNeuron`: The created neuron instance

**Example:**
```python
neuron = network.add_agent("analysis_agent_1", activation_threshold=0.7)
print(f"Added neuron: {neuron.agent_id}")
```

#### `create_connection(source_id: str, target_id: str, initial_strength: float = 0.5) → Optional[SynapticConnection]`
**Description:** Create a synaptic connection between two agents

**Parameters:**
- `source_id` (str): Source agent ID
- `target_id` (str): Target agent ID
- `initial_strength` (float, optional): Initial connection strength (default: 0.5)

**Returns:**
- `Optional[SynapticConnection]`: The created connection or None if failed

**Example:**
```python
connection = network.create_connection(
    "sentiment_agent",
    "trading_agent",
    initial_strength=0.8
)
if connection:
    print(f"Connected {connection.source_agent_id} -> {connection.target_agent_id}")
```

#### `remove_connection(source_id: str, target_id: str) → bool`
**Description:** Remove a connection between agents

**Parameters:**
- `source_id` (str): Source agent ID
- `target_id` (str): Target agent ID

**Returns:**
- `bool`: True if connection was removed

**Example:**
```python
removed = network.remove_connection("agent_1", "agent_2")
if removed:
    print("Connection removed successfully")
```

### Signal Propagation

#### `propagate_signal(source_agent_id: str, signal_data: Dict[str, Any]) → Dict[str, Any]`
**Description:** Propagate a signal through the network starting from a source agent

**Parameters:**
- `source_agent_id` (str): Agent initiating the signal
- `signal_data` (Dict[str, Any]): Signal payload with strength, type, content

**Returns:**
- `Dict[str, Any]`: Propagation results and network statistics

**Signal Data Structure:**
```python
{
    "strength": 1.0,        # Signal strength (0.0 to 1.0)
    "type": "information",  # Signal type
    "content": {...}        # Signal payload
}
```

**Response Structure:**
```python
{
    "source": "agent_1",
    "signal_type": "information",
    "propagation_results": [...],  # List of firing results
    "total_activated": 5,          # Number of activated neurons
    "network_stats": {...}         # Current network statistics
}
```

**Example:**
```python
signal_data = {
    "strength": 0.9,
    "type": "market_alert",
    "content": {
        "symbol": "BTC",
        "alert_type": "price_spike",
        "severity": "high"
    }
}

result = await network.propagate_signal("market_monitor_agent", signal_data)
print(f"Activated {result['total_activated']} neurons")
```

### Knowledge Sharing

#### `share_knowledge(source_agent_id: str, knowledge_key: str, knowledge_value: Any, target_agents: Optional[List[str]] = None) → int`
**Description:** Share knowledge from one agent to others through the network

**Parameters:**
- `source_agent_id` (str): Knowledge source agent
- `knowledge_key` (str): Knowledge identifier
- `knowledge_value` (Any): Knowledge content
- `target_agents` (List[str], optional): Specific targets, or all connected if None

**Returns:**
- `int`: Number of agents that received the knowledge

**Example:**
```python
# Share trading strategy
shared_count = network.share_knowledge(
    source_agent_id="strategy_agent",
    knowledge_key="rsi_divergence_strategy",
    knowledge_value={
        "description": "RSI divergence trading strategy",
        "parameters": {"rsi_period": 14, "divergence_threshold": 0.05},
        "backtest_results": {"win_rate": 0.68, "profit_factor": 1.4}
    }
)

print(f"Knowledge shared with {shared_count} agents")
```

### Network Analytics

#### `get_network_state() → Dict[str, Any]`
**Description:** Get comprehensive network status and topology

**Returns:**
- `Dict[str, Any]`: Complete network state information

**Response Structure:**
```python
{
    "network_id": "trading_swarm",
    "neurons": {
        "agent_1": {
            "agent_id": "agent_1",
            "activation": 0.3,
            "threshold": 0.5,
            "performance": 0.8,
            "input_connections": 3,
            "output_connections": 2,
            "last_fired": 1640995200.0,
            "knowledge_items": 15
        }
    },
    "connections": {
        "agent_1->agent_2": {
            "source": "agent_1",
            "target": "agent_2",
            "strength": 0.8,
            "last_activation": 0.7,
            "activation_count": 25,
            "age": 3600.0
        }
    },
    "topology": {
        "agent_1": ["agent_2", "agent_3"],
        "agent_2": ["agent_3"]
    },
    "stats": {
        "total_signals": 150,
        "active_connections": 8,
        "average_activation": 0.4,
        "network_density": 0.3
    },
    "global_knowledge": ["strategy_1", "market_data", "risk_params"],
    "learning_enabled": true,
    "adaptive_topology": true
}
```

**Example:**
```python
state = network.get_network_state()
print(f"Network has {len(state['neurons'])} neurons")
print(f"Network density: {state['stats']['network_density']:.2f}")
print(f"Active connections: {state['stats']['active_connections']}")
```

### Network Maintenance

#### `run_network_cycle() → None`
**Description:** Execute one complete network maintenance cycle

**Operations Performed:**
- Connection strength decay
- Topology adaptation (if enabled)
- Network statistics update

**Example:**
```python
await network.run_network_cycle()
print("Network maintenance cycle completed")
```

---

## Data Structures

### SwarmNeuron Class

```python
@dataclass
class SwarmNeuron:
    agent_id: str
    activation_threshold: float = 0.5
    current_activation: float = 0.0
    bias: float = 0.0
    refractory_period: float = 1.0  # seconds
    last_fired: float = 0.0

    # Connection management
    input_connections: Dict[str, SynapticConnection] = field(default_factory=dict)
    output_connections: Dict[str, SynapticConnection] = field(default_factory=dict)

    # Knowledge storage
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.5

    def receive_signal(self, source_agent_id: str, signal_strength: float) → float:
        """Receive and process incoming signal"""

    def should_fire(self) → bool:
        """Check if neuron should fire based on activation"""

    def fire(self) → Dict[str, Any]:
        """Fire neuron and propagate signal to connected agents"""

    def update_performance(self, new_score: float) → None:
        """Update neuron performance score"""

    def get_neuron_state(self) → Dict[str, Any]:
        """Get neuron status information"""
```

### SynapticConnection Class

```python
@dataclass
class SynapticConnection:
    source_agent_id: str
    target_agent_id: str
    strength: float = 1.0
    last_activation: float = 0.0
    activation_count: int = 0
    learning_rate: float = 0.1
    created_at: float = field(default_factory=time.time)

    def activate(self, signal_strength: float) → float:
        """Activate connection with signal and update strength"""

    def decay(self, decay_rate: float = 0.01) → None:
        """Apply time-based decay to connection strength"""

    def get_connection_info(self) → Dict[str, Any]:
        """Get connection metadata"""
```

---

## Usage Examples

### Basic Network Setup

```python
import asyncio
from src.swarm_intelligence.neural_swarm_network import NeuralSwarmNetwork

async def setup_network():
    # Create network
    network = NeuralSwarmNetwork("trading_network")

    # Add agents as neurons
    agents = ["market_data", "sentiment", "technical", "risk", "trading"]
    for agent in agents:
        network.add_agent(f"{agent}_agent")

    # Create connections
    connections = [
        ("market_data_agent", "technical_agent"),
        ("market_data_agent", "sentiment_agent"),
        ("technical_agent", "risk_agent"),
        ("sentiment_agent", "risk_agent"),
        ("risk_agent", "trading_agent")
    ]

    for source, target in connections:
        network.create_connection(source, target, initial_strength=0.6)

    print(f"Network setup complete with {len(agents)} agents")
    return network
```

### Signal Propagation Workflow

```python
async def demonstrate_signal_propagation():
    network = await setup_network()

    # Send market alert signal
    alert_signal = {
        "strength": 0.9,
        "type": "market_alert",
        "content": {
            "event": "large_BTC_buy",
            "size": 1000000,
            "exchange": "binance",
            "impact": "high"
        }
    }

    # Propagate from market data agent
    result = await network.propagate_signal("market_data_agent", alert_signal)

    print(f"Signal propagated to {result['total_activated']} agents")
    print(f"Propagation results: {len(result['propagation_results'])} firings")

    # Check network state after propagation
    state = network.get_network_state()
    print(f"Network activation: {state['stats']['average_activation']:.2f}")
```

### Knowledge Sharing System

```python
async def knowledge_sharing_example():
    network = await setup_network()

    # Technical agent shares analysis method
    analysis_method = {
        "name": "Fibonacci Retracement Analysis",
        "description": "Identify key support/resistance levels using Fibonacci ratios",
        "parameters": {
            "fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
            "lookback_period": 100,
            "min_retracement": 0.1
        },
        "accuracy": 0.72,
        "last_updated": "2024-01-15"
    }

    # Share with all connected agents
    shared_count = network.share_knowledge(
        source_agent_id="technical_agent",
        knowledge_key="fibonacci_analysis",
        knowledge_value=analysis_method
    )

    print(f"Knowledge shared with {shared_count} agents")

    # Risk agent can now access this knowledge
    risk_neuron = network.neurons["risk_agent"]
    if "fibonacci_analysis" in risk_neuron.knowledge_base:
        fib_data = risk_neuron.knowledge_base["fibonacci_analysis"]
        print(f"Risk agent learned: {fib_data['name']}")
```

### Adaptive Network Evolution

```python
async def adaptive_network_example():
    network = await setup_network()

    # Enable adaptive features
    network.learning_enabled = True
    network.adaptive_topology = True

    # Simulate network activity over time
    for cycle in range(10):
        # Send various signals
        signals = [
            {"strength": 0.7, "type": "price_data", "content": {"symbol": "BTC"}},
            {"strength": 0.8, "type": "sentiment", "content": {"score": 0.6}},
            {"strength": 0.5, "type": "volume_spike", "content": {"multiplier": 3.2}}
        ]

        for signal in signals:
            source = "market_data_agent" if signal["type"] == "price_data" else "sentiment_agent"
            await network.propagate_signal(source, signal)

        # Run maintenance cycle
        await network.run_network_cycle()

        # Monitor evolution
        state = network.get_network_state()
        print(f"Cycle {cycle + 1}:")
        print(f"  - Active connections: {state['stats']['active_connections']}")
        print(f"  - Network density: {state['stats']['network_density']:.3f}")
        print(f"  - Average activation: {state['stats']['average_activation']:.3f}")

        await asyncio.sleep(0.1)
```

---

## Network Properties

### Topology Metrics
- **Density**: Ratio of actual to possible connections
- **Clustering**: Local connectivity patterns
- **Centrality**: Agent importance in signal flow
- **Path Length**: Average signal propagation distance

### Learning Mechanisms
- **Hebbian Learning**: Connections strengthen with correlated activity
- **Decay**: Unused connections weaken over time
- **Performance-Based Adaptation**: Successful agents get stronger connections

### Emergent Behaviors
- **Consensus Formation**: Collective decision making
- **Pattern Recognition**: Distributed signal processing
- **Adaptive Routing**: Dynamic signal path optimization

---

## Performance Considerations

- **Async Operations:** All propagation methods are async
- **Memory Usage:** Knowledge bases grow over time
- **Connection Limits:** Large networks may need connection pruning
- **Signal Flooding:** High connectivity can cause signal storms
- **Learning Overhead:** Adaptive features add computational cost

---

## Integration with Base Agent

```python
from src.agents.base_agent import BaseAgent
from src.swarm_intelligence.neural_swarm_network import NeuralSwarmNetwork

class SwarmEnabledAgent(BaseAgent):
    def __init__(self, agent_id: str, network: NeuralSwarmNetwork):
        super().__init__(agent_id=agent_id)
        self.network = network
        self.neuron = network.add_agent(agent_id)

    def _initialize_agent(self) -> bool:
        # Connect to other agents in network
        self.network.create_connection(self.agent_id, "coordinator_agent")
        return True

    def _execute_agent_cycle(self):
        # Share findings with network
        if self.has_new_insights():
            insight = self.generate_insight()
            self.network.share_knowledge(
                self.agent_id,
                f"insight_{self.agent_id}",
                insight
            )

        # Process incoming signals
        # (Network handles signal propagation automatically)
```

---

## Related APIs

- **[Base Agent](base_agent.md)** - Individual agent lifecycle
- **[Task Orchestrator](task_orchestrator.md)** - Agent coordination
- **[Collective Decision Engine](swarm_intelligence/collective_decision_engine.md)** - Group decision making

---

*Built with ❤️ by Nyros Veil* | [Back to API Index](../README.md)