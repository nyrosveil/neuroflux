# Communication Bus API Reference

The Communication Bus is NeuroFlux's central messaging infrastructure, providing asynchronous inter-agent communication with priority routing, pub/sub patterns, and request/response capabilities.

## Overview

The Communication Bus enables agents to communicate through standardized messages with features like:
- Priority-based message queuing
- Topic-based pub/sub messaging
- Request/response patterns
- Broadcast capabilities
- Message persistence and retry mechanisms

## Core Classes

### Message

Standardized message format for inter-agent communication.

```python
@dataclass
class Message:
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcasts
    message_type: MessageType
    priority: MessagePriority
    topic: str
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    ttl: int = 300
    retry_count: int = 0
    max_retries: int = 3
```

### MessageType Enum

```python
class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    COMMAND = "command"
    NOTIFICATION = "notification"
```

### MessagePriority Enum

```python
class MessagePriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

## CommunicationBus Class

### Initialization

```python
bus = CommunicationBus(max_queue_size: int = 1000)
```

### Lifecycle Methods

#### `async def start() -> None`
Start the communication bus and begin message processing.

#### `async def stop() -> None`
Stop the communication bus and clean up resources.

### Agent Management

#### `async def register_agent(agent_id: str) -> None`
Register an agent with the communication bus.

**Parameters:**
- `agent_id` (str): Unique identifier for the agent

#### `async def unregister_agent(agent_id: str) -> None`
Unregister an agent from the communication bus.

**Parameters:**
- `agent_id` (str): ID of the agent to unregister

### Message Operations

#### `async def send_message(message: Message) -> None`
Send a message through the communication bus.

**Parameters:**
- `message` (Message): The message to send

#### `async def publish_event(sender_id: str, topic: str, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM) -> None`
Publish an event to all subscribers of a topic.

**Parameters:**
- `sender_id` (str): ID of the publishing agent
- `topic` (str): Event topic
- `payload` (Dict[str, Any]): Event data
- `priority` (MessagePriority): Message priority level

#### `async def send_request(sender_id: str, recipient_id: str, topic: str, payload: Dict[str, Any], timeout: float = 30.0, priority: MessagePriority = MessagePriority.MEDIUM) -> Dict[str, Any]`
Send a request and wait for response.

**Parameters:**
- `sender_id` (str): ID of the requesting agent
- `recipient_id` (str): ID of the target agent
- `topic` (str): Request topic
- `payload` (Dict[str, Any]): Request data
- `timeout` (float): Response timeout in seconds
- `priority` (MessagePriority): Message priority level

**Returns:**
- Response payload from the recipient agent

**Raises:**
- `TimeoutError`: If response times out

#### `async def send_response(sender_id: str, recipient_id: str, correlation_id: str, payload: Dict[str, Any]) -> None`
Send a response to a request.

**Parameters:**
- `sender_id` (str): ID of the responding agent
- `recipient_id` (str): ID of the original requester
- `correlation_id` (str): Correlation ID from the original request
- `payload` (Dict[str, Any]): Response data

#### `async def broadcast_message(sender_id: str, topic: str, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.MEDIUM) -> None`
Broadcast a message to all registered agents.

**Parameters:**
- `sender_id` (str): ID of the broadcasting agent
- `topic` (str): Broadcast topic
- `payload` (Dict[str, Any]): Broadcast data
- `priority` (MessagePriority): Message priority level

#### `async def get_agent_messages(agent_id: str) -> List[Message]`
Get pending messages for a specific agent.

**Parameters:**
- `agent_id` (str): ID of the agent

**Returns:**
- List of pending messages for the agent

### Statistics and Monitoring

#### `def get_stats() -> Dict[str, Any]`
Get communication bus statistics.

**Returns:**
```python
{
    'messages_processed': int,
    'messages_failed': int,
    'messages_retried': int,
    'avg_processing_time': float,
    'registered_agents': int,
    'active_subscriptions': int,
    'pending_responses': int
}
```

## Usage Examples

### Basic Message Sending

```python
from neuroflux.orchestration import CommunicationBus, Message, MessageType, MessagePriority

# Initialize bus
bus = CommunicationBus()
await bus.start()

# Register agents
await bus.register_agent("trading_agent_1")
await bus.register_agent("analysis_agent_1")

# Send a request
try:
    response = await bus.send_request(
        sender_id="trading_agent_1",
        recipient_id="analysis_agent_1",
        topic="market_analysis",
        payload={"symbol": "BTC/USD", "timeframe": "1h"},
        timeout=30.0
    )
    print(f"Analysis result: {response}")
except TimeoutError:
    print("Analysis request timed out")
```

### Event Publishing

```python
# Publish market data event
await bus.publish_event(
    sender_id="data_feed_agent",
    topic="market_data",
    payload={
        "symbol": "BTC/USD",
        "price": 45000.0,
        "volume": 100.5,
        "timestamp": time.time()
    },
    priority=MessagePriority.HIGH
)
```

### Broadcasting Commands

```python
# Broadcast system shutdown command
await bus.broadcast_message(
    sender_id="system_controller",
    topic="system_command",
    payload={
        "command": "shutdown",
        "reason": "maintenance",
        "graceful": True
    },
    priority=MessagePriority.CRITICAL
)
```

## Error Handling

The Communication Bus includes comprehensive error handling:

- **Message Expiration**: Messages are automatically discarded after TTL
- **Retry Logic**: Failed messages are retried up to `max_retries` times
- **Timeout Handling**: Request/response operations support configurable timeouts
- **Route Failures**: Invalid recipient IDs are logged and handled gracefully

## Performance Considerations

- **Priority Queuing**: Critical messages are processed before lower priority ones
- **Async Processing**: All operations are non-blocking and use asyncio
- **Message Batching**: Multiple messages can be processed concurrently
- **Resource Limits**: Configurable queue sizes prevent memory exhaustion

## Integration with Other Components

The Communication Bus integrates with:

- **Agent Registry**: For agent discovery and health monitoring
- **Task Orchestrator**: For coordinating complex multi-agent workflows
- **Conflict Resolution**: For consensus-based decision making
- **Neural Swarm Network**: For distributed intelligence coordination

## Cross-References

- See [Base Agent Framework](../base_agent.md) for agent lifecycle integration
- See [Agent Registry API](agent_registry.md) for agent discovery patterns
- See [Task Orchestrator API](task_orchestrator.md) for workflow coordination
- See [Conflict Resolution API](conflict_resolution.md) for consensus communication</content>
<parameter name="filePath">neuroflux/docs/api/communication_bus.md