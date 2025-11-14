# Agent Registry API Reference

The Agent Registry provides dynamic agent registration, service discovery, and health monitoring capabilities for NeuroFlux's multi-agent system.

## Overview

The Agent Registry enables agents to:
- Register themselves with the system
- Discover other agents by capabilities and performance
- Monitor agent health and status
- Auto-scale based on load and demand
- Track performance metrics and analytics

## Core Classes

### AgentStatus Enum

```python
class AgentStatus(Enum):
    REGISTERING = "registering"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    DEREGISTERING = "deregistering"
    OFFLINE = "offline"
```

### AgentCapability Enum

```python
class AgentCapability(Enum):
    TRADING = "trading"
    RESEARCH = "research"
    RISK_MANAGEMENT = "risk_management"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CHART_ANALYSIS = "chart_analysis"
    FUNDING_ANALYSIS = "funding_analysis"
    LIQUIDATION_MONITORING = "liquidation_monitoring"
    WHALE_TRACKING = "whale_tracking"
    NEWS_ANALYSIS = "news_analysis"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    BACKTESTING = "backtesting"
    EXECUTION = "execution"
```

### AgentInfo

```python
@dataclass
class AgentInfo:
    agent_id: str
    agent_type: str
    capabilities: Set[AgentCapability]
    status: AgentStatus = AgentStatus.REGISTERING
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0  # 0.0 to 1.0
    load_factor: float = 0.0  # 0.0 to 1.0
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
```

### ServiceQuery

```python
@dataclass
class ServiceQuery:
    capabilities: Optional[Set[AgentCapability]] = None
    agent_type: Optional[str] = None
    min_health_score: float = 0.5
    max_load_factor: float = 0.8
    tags: Optional[Set[str]] = None
    limit: int = 10
    sort_by: str = "performance"  # performance, health, load, random
```

## AgentRegistry Class

### Initialization

```python
registry = AgentRegistry(communication_bus: CommunicationBus)
```

### Lifecycle Methods

#### `async def start() -> None`
Start the agent registry system and begin health monitoring.

#### `async def stop() -> None`
Stop the agent registry system and clean up resources.

### Agent Registration

#### `async def register_agent(agent_info: Dict[str, Any]) -> str`
Register a new agent with the system.

**Parameters:**
- `agent_info` (Dict[str, Any]): Agent registration information containing:
  - `agent_type` (str): Type of agent (required)
  - `capabilities` (List[str]): List of agent capabilities (required)
  - `agent_id` (str): Optional custom agent ID
  - `metadata` (Dict[str, Any]): Additional metadata
  - `version` (str): Agent version
  - `tags` (List[str]): Agent tags

**Returns:**
- Agent ID assigned to the registered agent

**Raises:**
- `ValueError`: If agent information is invalid

**Example:**
```python
agent_info = {
    'agent_type': 'trading_agent',
    'capabilities': ['trading', 'risk_management'],
    'metadata': {'exchange': 'hyperliquid', 'strategy': 'momentum'},
    'version': '2.1.0',
    'tags': ['high_frequency', 'crypto']
}

agent_id = await registry.register_agent(agent_info)
```

#### `async def deregister_agent(agent_id: str) -> bool`
Deregister an agent from the system.

**Parameters:**
- `agent_id` (str): ID of the agent to deregister

**Returns:**
- `True` if deregistration successful, `False` otherwise

### Service Discovery

#### `async def discover_agents(query: ServiceQuery) -> List[AgentInfo]`
Discover agents matching the given criteria.

**Parameters:**
- `query` (ServiceQuery): Service discovery query

**Returns:**
- List of matching agents, ranked by relevance

**Example:**
```python
from neuroflux.orchestration.agent_registry import ServiceQuery, AgentCapability

query = ServiceQuery(
    capabilities={AgentCapability.TRADING, AgentCapability.RISK_MANAGEMENT},
    agent_type="trading_agent",
    min_health_score=0.8,
    max_load_factor=0.7,
    limit=5,
    sort_by="performance"
)

agents = await registry.discover_agents(query)
for agent in agents:
    print(f"Found agent: {agent.agent_id} ({agent.agent_type})")
```

### Health Monitoring

#### `async def update_agent_health(agent_id: str, health_data: Dict[str, Any]) -> None`
Update agent health information.

**Parameters:**
- `agent_id` (str): Agent ID
- `health_data` (Dict[str, Any]): Health metrics containing:
  - `health_score` (float): Overall health score (0.0-1.0)
  - `load_factor` (float): Current load factor (0.0-1.0)
  - `response_time` (float): Response time in seconds
  - Additional metrics

**Example:**
```python
await registry.update_agent_health("trading_agent_001", {
    'health_score': 0.95,
    'load_factor': 0.3,
    'response_time': 0.15,
    'active_connections': 5,
    'memory_usage': 0.6
})
```

#### `async def update_agent_performance(agent_id: str, metrics: Dict[str, float]) -> None`
Update agent performance metrics.

**Parameters:**
- `agent_id` (str): Agent ID
- `metrics` (Dict[str, float]): Performance metrics like:
  - `requests_served`: Number of requests processed
  - `avg_response_time`: Average response time
  - `success_rate`: Success rate (0.0-1.0)
  - `uptime`: Uptime percentage

### Information Retrieval

#### `def get_agent_info(agent_id: str) -> Optional[AgentInfo]`
Get information about a specific agent.

**Parameters:**
- `agent_id` (str): Agent ID

**Returns:**
- AgentInfo object or None if not found

#### `def get_registry_stats() -> Dict[str, Any]`
Get registry statistics and analytics.

**Returns:**
```python
{
    'total_registrations': int,
    'active_agents': int,
    'degraded_agents': int,
    'health_checks_performed': int,
    'failed_health_checks': int,
    'auto_scaling_events': int,
    'service_discovery_requests': int,
    'registered_agents': int,
    'capabilities_distribution': Dict[str, int],
    'health_distribution': Dict[str, int]
}
```

## Usage Examples

### Basic Agent Registration

```python
from neuroflux.orchestration import AgentRegistry
from neuroflux.orchestration.communication_bus import CommunicationBus

# Initialize components
bus = CommunicationBus()
registry = AgentRegistry(bus)

await bus.start()
await registry.start()

# Register a trading agent
agent_config = {
    'agent_type': 'trading_agent',
    'capabilities': ['trading', 'execution'],
    'metadata': {
        'exchange': 'hyperliquid',
        'strategy_type': 'arbitrage',
        'risk_level': 'medium'
    },
    'tags': ['crypto', 'perpetuals']
}

agent_id = await registry.register_agent(agent_config)
print(f"Registered agent: {agent_id}")
```

### Service Discovery

```python
from neuroflux.orchestration.agent_registry import ServiceQuery, AgentCapability

# Find healthy trading agents with low load
query = ServiceQuery(
    capabilities={AgentCapability.TRADING},
    min_health_score=0.9,
    max_load_factor=0.5,
    sort_by="performance",
    limit=3
)

available_agents = await registry.discover_agents(query)

for agent in available_agents:
    print(f"Available: {agent.agent_id}")
    print(f"  Health: {agent.health_score}")
    print(f"  Load: {agent.load_factor}")
    print(f"  Capabilities: {list(agent.capabilities)}")
```

### Health Monitoring Integration

```python
# Update agent health periodically
async def health_check_loop():
    while True:
        health_data = {
            'health_score': calculate_health_score(),
            'load_factor': get_current_load(),
            'response_time': measure_response_time(),
            'memory_usage': get_memory_usage(),
            'cpu_usage': get_cpu_usage()
        }

        await registry.update_agent_health(my_agent_id, health_data)
        await asyncio.sleep(30)  # Check every 30 seconds
```

### Performance Tracking

```python
# Track performance metrics
performance_metrics = {
    'requests_served': requests_count,
    'avg_response_time': avg_response_time,
    'success_rate': success_rate,
    'uptime': uptime_percentage
}

await registry.update_agent_performance(agent_id, performance_metrics)
```

## Auto-Scaling Features

The Agent Registry includes automatic scaling capabilities:

- **Load-based Scaling**: Scales up/down based on agent load factors
- **Health-based Scaling**: Replaces unhealthy agents automatically
- **Capability Balancing**: Maintains optimal distribution of capabilities
- **Event-driven Triggers**: Broadcasts scaling events for orchestration

### Scaling Configuration

```python
# Configure auto-scaling parameters
registry.auto_scaling_enabled = True
registry.min_agents_per_capability = 2
registry.max_agents_per_capability = 10
registry.scaling_thresholds = {
    'high_load': 0.8,      # Scale up when load > 80%
    'low_load': 0.3,       # Scale down when load < 30%
    'health_degraded': 0.6 # Replace agents with health < 60%
}
```

## Error Handling

The Agent Registry includes comprehensive error handling:

- **Validation Errors**: Invalid agent registration data
- **Duplicate Registration**: Attempting to register existing agent
- **Not Found Errors**: Operations on non-existent agents
- **Health Check Failures**: Automatic handling of failed health checks
- **Timeout Handling**: Heartbeat timeout detection and handling

## Performance Considerations

- **Efficient Discovery**: Fast agent lookup with multiple indexing strategies
- **Health Monitoring**: Lightweight periodic health checks
- **Statistics Tracking**: Minimal overhead performance metrics
- **Memory Management**: Automatic cleanup of deregistered agents
- **Concurrent Operations**: Thread-safe operations for multi-agent environments

## Integration with Other Components

The Agent Registry integrates with:

- **Communication Bus**: For agent messaging and health checks
- **Task Orchestrator**: For agent assignment and load balancing
- **Conflict Resolution**: For agent capability conflict detection
- **Neural Swarm Network**: For swarm intelligence coordination

## Cross-References

- See [Communication Bus API](communication_bus.md) for messaging integration
- See [Task Orchestrator API](task_orchestrator.md) for agent assignment
- See [Conflict Resolution API](conflict_resolution.md) for capability conflicts
- See [Base Agent Framework](../base_agent.md) for agent lifecycle integration</content>
<parameter name="filePath">neuroflux/docs/api/agent_registry.md