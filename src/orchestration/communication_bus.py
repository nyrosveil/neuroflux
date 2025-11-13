"""
🧠 NeuroFlux Agent Communication Bus
Asynchronous message passing system for inter-agent communication.

Built with love by Nyros Veil 🚀

Features:
- Async message queues with priority routing
- Pub/Sub pattern for event-driven communication
- Request/Response pattern for direct queries
- Broadcast capabilities for swarm coordination
- Message persistence and retry mechanisms
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MessagePriority(Enum):
    """Message priority levels for routing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MessageType(Enum):
    """Types of messages that can be sent between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    BROADCAST = "broadcast"
    COMMAND = "command"
    NOTIFICATION = "notification"

@dataclass
class Message:
    """Standardized message format for inter-agent communication."""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcasts
    message_type: MessageType
    priority: MessagePriority
    topic: str
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None  # For request/response correlation
    ttl: int = 300  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return time.time() - self.timestamp > self.ttl

class MessageQueue:
    """Priority-based message queue with async operations."""

    def __init__(self, max_size: int = 1000):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.max_size = max_size
        self.pending_messages: Dict[str, Message] = {}
        self.lock = asyncio.Lock()

    def _get_priority_value(self, priority: MessagePriority) -> int:
        """Convert priority to numeric value for queue ordering."""
        priority_map = {
            MessagePriority.LOW: 4,
            MessagePriority.MEDIUM: 3,
            MessagePriority.HIGH: 2,
            MessagePriority.CRITICAL: 1
        }
        return priority_map[priority]

    async def put(self, message: Message) -> None:
        """Add message to queue with priority ordering."""
        priority_value = self._get_priority_value(message.priority)
        await self.queue.put((priority_value, message.timestamp, message.message_id, message))

        # Store for potential retry
        async with self.lock:
            self.pending_messages[message.message_id] = message

    async def get(self) -> Optional[Message]:
        """Get next message from queue."""
        try:
            _, _, _, message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None

    async def acknowledge(self, message_id: str) -> None:
        """Acknowledge successful processing of message."""
        async with self.lock:
            if message_id in self.pending_messages:
                del self.pending_messages[message_id]

    async def retry_message(self, message: Message) -> None:
        """Retry a failed message."""
        if message.retry_count < message.max_retries:
            message.retry_count += 1
            message.timestamp = time.time()  # Reset timestamp
            await self.put(message)
        else:
            cprint(f"❌ Message {message.message_id} exceeded max retries", "red")

class SubscriptionManager:
    """Manages topic subscriptions for pub/sub pattern."""

    def __init__(self):
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> [agent_ids]
        self.agent_subscriptions: Dict[str, List[str]] = {}  # agent_id -> [topics]
        self.lock = asyncio.Lock()

    async def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to a topic."""
        async with self.lock:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = []
            if agent_id not in self.subscriptions[topic]:
                self.subscriptions[topic].append(agent_id)

            if agent_id not in self.agent_subscriptions:
                self.agent_subscriptions[agent_id] = []
            if topic not in self.agent_subscriptions[agent_id]:
                self.agent_subscriptions[agent_id].append(topic)

    async def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from a topic."""
        async with self.lock:
            if topic in self.subscriptions and agent_id in self.subscriptions[topic]:
                self.subscriptions[topic].remove(agent_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]

            if agent_id in self.agent_subscriptions and topic in self.agent_subscriptions[agent_id]:
                self.agent_subscriptions[agent_id].remove(topic)
                if not self.agent_subscriptions[agent_id]:
                    del self.agent_subscriptions[agent_id]

    async def get_subscribers(self, topic: str) -> List[str]:
        """Get all agents subscribed to a topic."""
        async with self.lock:
            return self.subscriptions.get(topic, []).copy()

    async def get_agent_subscriptions(self, agent_id: str) -> List[str]:
        """Get all topics an agent is subscribed to."""
        async with self.lock:
            return self.agent_subscriptions.get(agent_id, []).copy()

class CommunicationBus:
    """
    Central communication hub for inter-agent messaging.

    Features:
    - Async message routing with priority queues
    - Pub/Sub pattern for event-driven communication
    - Request/Response pattern for direct queries
    - Broadcast capabilities for swarm coordination
    - Message persistence and retry mechanisms
    """

    def __init__(self, max_queue_size: int = 1000):
        self.message_queue = MessageQueue(max_queue_size)
        self.subscription_manager = SubscriptionManager()
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.agent_queues: Dict[str, MessageQueue] = {}
        self.running = False
        self.message_processor_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

        # Message processing statistics
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'messages_retried': 0,
            'avg_processing_time': 0.0
        }

    async def start(self) -> None:
        """Start the communication bus."""
        cprint("🚌 Starting Agent Communication Bus...", "cyan")
        self.running = True
        self.message_processor_task = asyncio.create_task(self._process_messages())
        cprint("✅ Communication Bus started", "green")

    async def stop(self) -> None:
        """Stop the communication bus."""
        cprint("🛑 Stopping Agent Communication Bus...", "yellow")
        self.running = False
        if self.message_processor_task:
            self.message_processor_task.cancel()
            try:
                await self.message_processor_task
            except asyncio.CancelledError:
                pass
        cprint("✅ Communication Bus stopped", "green")

    async def register_agent(self, agent_id: str) -> None:
        """Register an agent with the communication bus."""
        async with self.lock:
            if agent_id not in self.agent_queues:
                self.agent_queues[agent_id] = MessageQueue()
                cprint(f"📝 Agent {agent_id} registered with communication bus", "blue")

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the communication bus."""
        async with self.lock:
            if agent_id in self.agent_queues:
                del self.agent_queues[agent_id]
                # Clean up subscriptions
                topics = await self.subscription_manager.get_agent_subscriptions(agent_id)
                for topic in topics:
                    await self.subscription_manager.unsubscribe(agent_id, topic)
                cprint(f"📝 Agent {agent_id} unregistered from communication bus", "blue")

    async def send_message(self, message: Message) -> None:
        """Send a message through the communication bus."""
        await self.message_queue.put(message)

    async def publish_event(self, sender_id: str, topic: str, payload: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.MEDIUM) -> None:
        """Publish an event to all subscribers of a topic."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.EVENT,
            priority=priority,
            topic=topic,
            payload=payload,
            timestamp=time.time()
        )
        await self.send_message(message)

    async def send_request(self, sender_id: str, recipient_id: str, topic: str,
                          payload: Dict[str, Any], timeout: float = 30.0,
                          priority: MessagePriority = MessagePriority.MEDIUM) -> Dict[str, Any]:
        """Send a request and wait for response."""
        correlation_id = str(uuid.uuid4())
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.REQUEST,
            priority=priority,
            topic=topic,
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id
        )

        # Create future for response
        response_future = asyncio.Future()
        self.response_handlers[correlation_id] = response_future

        await self.send_message(message)

        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {recipient_id} timed out")
        finally:
            # Clean up
            if correlation_id in self.response_handlers:
                del self.response_handlers[correlation_id]

    async def send_response(self, sender_id: str, recipient_id: str, correlation_id: str,
                          payload: Dict[str, Any]) -> None:
        """Send a response to a request."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.HIGH,  # Responses are high priority
            topic="response",
            payload=payload,
            timestamp=time.time(),
            correlation_id=correlation_id
        )
        await self.send_message(message)

    async def broadcast_message(self, sender_id: str, topic: str, payload: Dict[str, Any],
                              priority: MessagePriority = MessagePriority.MEDIUM) -> None:
        """Broadcast a message to all registered agents."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=None,  # Broadcast to all
            message_type=MessageType.BROADCAST,
            priority=priority,
            topic=topic,
            payload=payload,
            timestamp=time.time()
        )
        await self.send_message(message)

    async def get_agent_messages(self, agent_id: str) -> List[Message]:
        """Get pending messages for a specific agent."""
        if agent_id not in self.agent_queues:
            return []

        messages = []
        while True:
            message = await self.agent_queues[agent_id].get()
            if message is None:
                break
            messages.append(message)

        return messages

    async def _process_messages(self) -> None:
        """Main message processing loop."""
        while self.running:
            message = None
            try:
                message = await self.message_queue.get()
                if message is None:
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.time()
                await self._route_message(message)
                processing_time = time.time() - start_time

                # Update statistics
                self.stats['messages_processed'] += 1
                self.stats['avg_processing_time'] = (
                    (self.stats['avg_processing_time'] * (self.stats['messages_processed'] - 1)) +
                    processing_time
                ) / self.stats['messages_processed']

                await self.message_queue.acknowledge(message.message_id)

            except Exception as e:
                cprint(f"❌ Message processing error: {str(e)}", "red")
                self.stats['messages_failed'] += 1
                if message:
                    await self.message_queue.retry_message(message)

    async def _route_message(self, message: Message) -> None:
        """Route message to appropriate recipients."""
        if message.is_expired():
            cprint(f"⏰ Message {message.message_id} expired", "yellow")
            return

        # Handle broadcasts
        if message.recipient_id is None:
            await self._handle_broadcast(message)
            return

        # Handle responses
        if message.message_type == MessageType.RESPONSE and message.correlation_id:
            await self._handle_response(message)
            return

        # Route to specific agent
        if message.recipient_id in self.agent_queues:
            await self.agent_queues[message.recipient_id].put(message)
        else:
            cprint(f"⚠️ No route to agent {message.recipient_id}", "yellow")

    async def _handle_broadcast(self, message: Message) -> None:
        """Handle broadcast messages."""
        if message.message_type == MessageType.EVENT:
            # Route to topic subscribers
            subscribers = await self.subscription_manager.get_subscribers(message.topic)
            for agent_id in subscribers:
                if agent_id in self.agent_queues:
                    await self.agent_queues[agent_id].put(message)
        else:
            # Route to all agents
            for agent_id, queue in self.agent_queues.items():
                if agent_id != message.sender_id:  # Don't send to self
                    await queue.put(message)

    async def _handle_response(self, message: Message) -> None:
        """Handle response messages."""
        if message.correlation_id in self.response_handlers:
            future = self.response_handlers[message.correlation_id]
            if not future.done():
                future.set_result(message.payload)

    def get_stats(self) -> Dict[str, Any]:
        """Get communication bus statistics."""
        return {
            **self.stats,
            'registered_agents': len(self.agent_queues),
            'active_subscriptions': len(self.subscription_manager.subscriptions),
            'pending_responses': len(self.response_handlers)
        }