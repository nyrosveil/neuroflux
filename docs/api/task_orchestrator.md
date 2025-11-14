# 🧠 Task Orchestrator API

## Overview

The `TaskOrchestrator` class provides dynamic task assignment and coordination across multiple agents, implementing load balancing, dependency management, and failure recovery.

**Location:** `src/orchestration/task_orchestrator.py`

**Dependencies:** Requires `CommunicationBus` for inter-agent messaging

---

## Class Architecture

```python
TaskOrchestrator
├── Task Management
├── Agent Coordination
├── Performance Analytics
├── Failure Recovery
└── Load Balancing
```

---

## Core Methods

### Initialization & Lifecycle

#### `__init__(communication_bus: CommunicationBus) → None`
**Description:** Initialize the task orchestrator with communication infrastructure

**Parameters:**
- `communication_bus` (CommunicationBus): Inter-agent communication system

**Example:**
```python
from src.orchestration.task_orchestrator import TaskOrchestrator
from src.orchestration.communication_bus import CommunicationBus

bus = CommunicationBus()
orchestrator = TaskOrchestrator(bus)
```

#### `start() → None`
**Description:** Start the orchestration loop and task processing

**Raises:**
- `Exception`: If communication bus is not available

**Example:**
```python
await orchestrator.start()  # Async operation
```

#### `stop() → None`
**Description:** Gracefully stop orchestration and cleanup resources

**Example:**
```python
await orchestrator.stop()
```

---

### Agent Management

#### `register_agent(agent_id: str, capabilities: List[str], max_concurrent_tasks: int = 5) → None`
**Description:** Register an agent with its capabilities and capacity limits

**Parameters:**
- `agent_id` (str): Unique agent identifier
- `capabilities` (List[str]): List of agent capabilities (e.g., ['trading', 'analysis'])
- `max_concurrent_tasks` (int, optional): Maximum concurrent tasks (default: 5)

**Example:**
```python
await orchestrator.register_agent(
    agent_id="trading_agent_1",
    capabilities=["trading", "market_data", "risk_management"],
    max_concurrent_tasks=3
)
```

#### `unregister_agent(agent_id: str) → None`
**Description:** Remove an agent and reassign its tasks

**Parameters:**
- `agent_id` (str): Agent to remove

**Example:**
```python
await orchestrator.unregister_agent("failed_agent_1")
```

---

### Task Management

#### `submit_task(name: str, description: str, task_type: str, payload: Dict[str, Any], ...) → str`
**Description:** Submit a new task for execution

**Parameters:**
- `name` (str): Human-readable task name
- `description` (str): Detailed task description
- `task_type` (str): Task category (e.g., 'analysis', 'trading')
- `payload` (Dict[str, Any]): Task-specific data
- `priority` (TaskPriority, optional): Task priority level
- `dependencies` (List[str], optional): Task IDs this task depends on
- `required_capabilities` (List[str], optional): Required agent capabilities
- `estimated_duration` (int, optional): Expected duration in seconds
- `timeout` (int, optional): Maximum execution time in seconds

**Returns:**
- `str`: Unique task ID

**Example:**
```python
from src.orchestration.task_orchestrator import TaskPriority

task_id = await orchestrator.submit_task(
    name="BTC Analysis",
    description="Analyze BTC price action and sentiment",
    task_type="analysis",
    payload={
        "symbol": "BTC",
        "timeframe": "1H",
        "indicators": ["rsi", "macd"]
    },
    priority=TaskPriority.HIGH,
    required_capabilities=["technical_analysis", "sentiment_analysis"],
    estimated_duration=300,
    timeout=600
)
```

#### `cancel_task(task_id: str) → bool`
**Description:** Cancel a pending or assigned task

**Parameters:**
- `task_id` (str): Task to cancel

**Returns:**
- `bool`: True if successfully cancelled

**Example:**
```python
success = await orchestrator.cancel_task("task_123")
if success:
    print("Task cancelled successfully")
```

#### `get_task_status(task_id: str) → Optional[Dict[str, Any]]`
**Description:** Retrieve current task status and metadata

**Parameters:**
- `task_id` (str): Task to query

**Returns:**
- `Optional[Dict]`: Task status information or None if not found

**Response Structure:**
```python
{
    "task_id": "task_123",
    "name": "BTC Analysis",
    "status": "running",
    "assigned_agent": "analysis_agent_1",
    "created_at": 1640995200.0,
    "started_at": 1640995260.0,
    "priority": "high",
    "progress": 0.75,
    "result": {...},  # Available when completed
    "error_message": None
}
```

---

### Performance Analytics

#### `get_stats() → Dict[str, Any]`
**Description:** Get comprehensive orchestrator statistics

**Returns:**
- `Dict[str, Any]`: Performance and utilization metrics

**Response Structure:**
```python
{
    "tasks_created": 150,
    "tasks_completed": 142,
    "tasks_failed": 8,
    "avg_completion_time": 245.3,
    "pending_tasks": 5,
    "running_tasks": 3,
    "completed_tasks": 142,
    "registered_agents": 8,
    "agent_utilization": {
        "trading_agent_1": 0.8,
        "analysis_agent_1": 0.6
    },
    "overall_utilization": 0.72
}
```

---

## Task Priority Enum

```python
class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

---

## Task Status Enum

```python
class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
```

---

## Task Data Structure

```python
@dataclass
class Task:
    task_id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: int = 300
    max_retries: int = 3
    timeout: int = 600

    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
```

---

## Agent Capability Structure

```python
@dataclass
class AgentCapability:
    agent_id: str
    capabilities: List[str]
    performance_score: float = 1.0
    current_load: int = 0
    max_concurrent_tasks: int = 5
    specialization_score: Dict[str, float] = field(default_factory=dict)
```

---

## Usage Examples

### Basic Task Submission

```python
import asyncio
from src.orchestration.task_orchestrator import TaskOrchestrator, TaskPriority

async def main():
    # Initialize orchestrator
    orchestrator = TaskOrchestrator(communication_bus)
    await orchestrator.start()

    # Submit analysis task
    task_id = await orchestrator.submit_task(
        name="Market Sentiment Analysis",
        description="Analyze current market sentiment across major cryptocurrencies",
        task_type="sentiment_analysis",
        payload={
            "symbols": ["BTC", "ETH", "SOL"],
            "sources": ["twitter", "news", "reddit"],
            "timeframe": "24h"
        },
        priority=TaskPriority.HIGH,
        required_capabilities=["sentiment_analysis", "social_media"],
        estimated_duration=600
    )

    print(f"Task submitted: {task_id}")

    # Monitor task progress
    while True:
        status = await orchestrator.get_task_status(task_id)
        if status:
            print(f"Task status: {status['status']}")
            if status['status'] in ['completed', 'failed']:
                break
        await asyncio.sleep(5)

    await orchestrator.stop()
```

### Complex Workflow with Dependencies

```python
async def create_trading_workflow():
    # Step 1: Market analysis
    analysis_task = await orchestrator.submit_task(
        name="Market Analysis",
        description="Comprehensive market analysis",
        task_type="analysis",
        payload={"symbols": ["BTC"], "indicators": ["rsi", "macd", "volume"]},
        priority=TaskPriority.HIGH
    )

    # Step 2: Risk assessment (depends on analysis)
    risk_task = await orchestrator.submit_task(
        name="Risk Assessment",
        description="Evaluate trading risks",
        task_type="risk_analysis",
        payload={"analysis_task_id": analysis_task},
        dependencies=[analysis_task],
        priority=TaskPriority.CRITICAL
    )

    # Step 3: Execute trade (depends on risk assessment)
    trade_task = await orchestrator.submit_task(
        name="Execute Trade",
        description="Execute trading decision",
        task_type="trading",
        payload={"risk_task_id": risk_task},
        dependencies=[risk_task],
        priority=TaskPriority.CRITICAL
    )

    return [analysis_task, risk_task, trade_task]
```

### Agent Registration and Load Balancing

```python
# Register multiple agents with different capabilities
agents = [
    ("analysis_agent_1", ["technical_analysis", "sentiment_analysis"], 5),
    ("trading_agent_1", ["trading", "order_execution"], 3),
    ("risk_agent_1", ["risk_management", "portfolio_analysis"], 2),
]

for agent_id, capabilities, max_tasks in agents:
    await orchestrator.register_agent(
        agent_id=agent_id,
        capabilities=capabilities,
        max_concurrent_tasks=max_tasks
    )

# Submit tasks - orchestrator will automatically balance load
for i in range(10):
    await orchestrator.submit_task(
        name=f"Analysis Task {i}",
        description=f"Market analysis job {i}",
        task_type="analysis",
        payload={"task_number": i},
        required_capabilities=["technical_analysis"]
    )
```

---

## Error Handling

```python
try:
    task_id = await orchestrator.submit_task(...)
except ValueError as e:
    print(f"Invalid task parameters: {e}")
except ConnectionError as e:
    print(f"Communication bus error: {e}")
except Exception as e:
    print(f"Unexpected orchestrator error: {e}")
```

---

## Performance Considerations

- **Async Operations:** All methods are async - use proper event loops
- **Load Balancing:** Tasks are automatically distributed based on agent capacity
- **Dependency Resolution:** Tasks with dependencies are queued until prerequisites complete
- **Failure Recovery:** Failed tasks are automatically retried or reassigned
- **Resource Limits:** Monitor agent utilization to prevent overload

---

## Integration with Communication Bus

The Task Orchestrator integrates with the Communication Bus for:
- **Task Assignment:** Sending tasks to assigned agents
- **Result Collection:** Receiving task completion notifications
- **Status Updates:** Real-time task progress monitoring
- **Error Propagation:** Failure notifications and recovery

---

## Related APIs

- **[Communication Bus](orchestration/communication_bus.md)** - Inter-agent messaging
- **[Base Agent](base_agent.md)** - Agent lifecycle management
- **[Agent Registry](orchestration/agent_registry.md)** - Agent discovery

---

*Built with ❤️ by Nyros Veil* | [Back to API Index](../README.md)