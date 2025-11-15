"""
🧠 NeuroFlux Real-Time Agent Bus
Event-driven agent coordination and message broadcasting system.

Built with love by Nyros Veil 🚀

Provides real-time communication between agents and dashboard clients:
- Event-driven agent messaging
- Real-time signal broadcasting
- Agent coordination through WebSocket channels
- Message routing and filtering
- Performance monitoring and analytics

Features:
- Pub/Sub messaging for agents
- Real-time event broadcasting
- Message prioritization and routing
- Agent health monitoring
- Performance metrics collection
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading

from termcolor import cprint


class MessagePriority(Enum):
    """Message priority levels for real-time routing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MessageType(Enum):
    """Types of real-time messages."""
    AGENT_STATUS = "agent_status"
    TRADING_SIGNAL = "trading_signal"
    MARKET_DATA = "market_data"
    RISK_ALERT = "risk_alert"
    SYSTEM_EVENT = "system_event"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"
    PREDICTION_UPDATE = "prediction_update"
    PRICE_PREDICTION = "price_prediction"
    VOLATILITY_UPDATE = "volatility_update"
    SENTIMENT_UPDATE = "sentiment_update"
    MODEL_PERFORMANCE = "model_performance"


@dataclass
class RealTimeMessage:
    """Real-time message format for agent communication."""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    sender: str
    recipient: Optional[str]  # None for broadcasts
    topic: str
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    ttl: int = 30  # Time to live in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket transmission."""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'sender': self.sender,
            'recipient': self.recipient,
            'topic': self.topic,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RealTimeMessage':
        """Create from dictionary."""
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            priority=MessagePriority(data['priority']),
            sender=data['sender'],
            recipient=data.get('recipient'),
            topic=data['topic'],
            payload=data['payload'],
            timestamp=data['timestamp'],
            correlation_id=data.get('correlation_id')
        )


class RealTimeAgentBus:
    """
    Real-time agent bus for event-driven coordination and broadcasting.

    Acts as a bridge between the orchestrator's communication system
    and WebSocket clients, enabling real-time agent interactions.
    """

    def __init__(self, orchestrator=None, socketio_instance=None):
        self.orchestrator = orchestrator
        self.socketio = socketio_instance

        # Message routing
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> set of subscriber_ids
        self.agent_subscriptions: Dict[str, Set[str]] = {}  # agent_id -> set of topics

        # Message queues
        self.message_queue = asyncio.Queue()
        self.pending_messages: Dict[str, RealTimeMessage] = {}

        # Performance tracking
        self.message_counts: Dict[str, int] = {}
        self.processing_times: List[float] = []
        self.error_counts: Dict[str, int] = {}

        # Control
        self.running = False
        self._processing_task: Optional[asyncio.Task] = None

        # Callbacks
        self.message_callbacks: List[Callable] = []
        self.event_callbacks: Dict[str, List[Callable]] = {}

    async def start(self) -> bool:
        """Start the real-time agent bus."""
        cprint("🚀 Starting Real-Time Agent Bus...", "cyan")

        try:
            self.running = True
            self._processing_task = asyncio.create_task(self._process_messages())

            # Subscribe to orchestrator events if available
            if self.orchestrator:
                await self._setup_orchestrator_integration()

            cprint("✅ Real-Time Agent Bus started", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to start Real-Time Agent Bus: {e}", "red")
            return False

    async def stop(self) -> bool:
        """Stop the real-time agent bus."""
        cprint("🛑 Stopping Real-Time Agent Bus...", "yellow")

        try:
            self.running = False

            if self._processing_task:
                self._processing_task.cancel()
                try:
                    await self._processing_task
                except asyncio.CancelledError:
                    pass

            cprint("✅ Real-Time Agent Bus stopped", "green")
            return True

        except Exception as e:
            cprint(f"❌ Error stopping Real-Time Agent Bus: {e}", "red")
            return False

    async def _setup_orchestrator_integration(self):
        """Set up integration with the NeuroFlux orchestrator."""
        if not self.orchestrator:
            return

        # Set up callbacks for orchestrator events
        # This would integrate with the communication bus
        pass

    async def publish_message(self, message: RealTimeMessage) -> bool:
        """Publish a real-time message to the bus."""
        try:
            await self.message_queue.put(message)
            self.message_counts[message.topic] = self.message_counts.get(message.topic, 0) + 1
            return True

        except Exception as e:
            cprint(f"❌ Failed to publish message: {e}", "red")
            self.error_counts['publish'] = self.error_counts.get('publish', 0) + 1
            return False

    async def send_agent_message(self, agent_id: str, topic: str, payload: Dict[str, Any],
                               priority: MessagePriority = MessagePriority.MEDIUM) -> bool:
        """Send a message to a specific agent."""
        message = RealTimeMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COORDINATION,
            priority=priority,
            sender="realtime_bus",
            recipient=agent_id,
            topic=topic,
            payload=payload,
            timestamp=time.time()
        )

        return await self.publish_message(message)

    async def broadcast_event(self, topic: str, payload: Dict[str, Any],
                            priority: MessagePriority = MessagePriority.MEDIUM) -> bool:
        """Broadcast an event to all subscribers."""
        message = RealTimeMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            priority=priority,
            sender="realtime_bus",
            recipient=None,
            topic=topic,
            payload=payload,
            timestamp=time.time()
        )

        return await self.publish_message(message)

    async def broadcast_prediction_update(self, prediction_data: Dict[str, Any],
                                        priority: MessagePriority = MessagePriority.HIGH) -> bool:
        """Broadcast prediction update to all dashboard clients."""
        message = RealTimeMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            priority=priority,
            sender="orchestrator",
            recipient=None,
            topic="prediction_update",
            payload={
                "type": "prediction_update",
                "data": prediction_data,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=time.time()
        )

        success = await self.publish_message(message)
        if success:
            cprint(f"📊 Broadcasted prediction update: {prediction_data.get('task_name', 'unknown')}", "green")
        return success

    async def broadcast_sentiment_update(self, sentiment_data: Dict[str, Any],
                                        priority: MessagePriority = MessagePriority.MEDIUM) -> bool:
        """Broadcast sentiment analysis update to all dashboard clients."""
        message = RealTimeMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.SENTIMENT_UPDATE,
            priority=priority,
            sender="orchestrator",
            recipient=None,
            topic="sentiment_update",
            payload={
                "type": "sentiment_update",
                "data": sentiment_data,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=time.time()
        )

        success = await self.publish_message(message)
        if success:
            cprint(f"📊 Broadcasted sentiment update: {sentiment_data.get('token', 'unknown')}", "green")
        return success

    async def broadcast_volatility_update(self, volatility_data: Dict[str, Any],
                                         priority: MessagePriority = MessagePriority.MEDIUM) -> bool:
        """Broadcast volatility/risk analysis update to all dashboard clients."""
        message = RealTimeMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.VOLATILITY_UPDATE,
            priority=priority,
            sender="orchestrator",
            recipient=None,
            topic="volatility_update",
            payload={
                "type": "volatility_update",
                "data": volatility_data,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=time.time()
        )

        success = await self.publish_message(message)
        if success:
            cprint(f"📊 Broadcasted volatility update: {volatility_data.get('volatility_pct', 0):.2f}%", "green")
        return success

    async def subscribe_topic(self, subscriber_id: str, topic: str) -> bool:
        """Subscribe to a topic."""
        try:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()

            self.subscriptions[topic].add(subscriber_id)

            if subscriber_id not in self.agent_subscriptions:
                self.agent_subscriptions[subscriber_id] = set()

            self.agent_subscriptions[subscriber_id].add(topic)

            return True

        except Exception as e:
            cprint(f"❌ Failed to subscribe to topic {topic}: {e}", "red")
            return False

    async def unsubscribe_topic(self, subscriber_id: str, topic: str) -> bool:
        """Unsubscribe from a topic."""
        try:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(subscriber_id)

            if subscriber_id in self.agent_subscriptions:
                self.agent_subscriptions[subscriber_id].discard(topic)

            return True

        except Exception as e:
            cprint(f"❌ Failed to unsubscribe from topic {topic}: {e}", "red")
            return False

    async def _process_messages(self):
        """Main message processing loop."""
        while self.running:
            try:
                message = await self.message_queue.get()

                if message is None:
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.time()

                # Route the message
                await self._route_message(message)

                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

                # Keep only last 1000 processing times
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-1000:]

            except Exception as e:
                cprint(f"❌ Message processing error: {e}", "red")
                self.error_counts['processing'] = self.error_counts.get('processing', 0) + 1

    async def _route_message(self, message: RealTimeMessage):
        """Route message to appropriate recipients."""
        try:
            # Check if message is expired
            if time.time() - message.timestamp > message.ttl:
                return

            # Route based on message type and recipients
            if message.recipient is None:
                # Broadcast message
                await self._handle_broadcast(message)
            else:
                # Direct message
                await self._handle_direct_message(message)

            # Call message callbacks
            for callback in self.message_callbacks:
                try:
                    await callback(message)
                except Exception as e:
                    cprint(f"❌ Error in message callback: {e}", "red")

        except Exception as e:
            cprint(f"❌ Error routing message: {e}", "red")

    async def _handle_broadcast(self, message: RealTimeMessage):
        """Handle broadcast messages."""
        # Send to all subscribers of the topic
        subscribers = self.subscriptions.get(message.topic, set())

        # Also send to WebSocket clients if available
        if self.socketio:
            try:
                # Convert message for WebSocket transmission
                ws_message = message.to_dict()
                ws_message['event_type'] = 'broadcast'

                # Emit to WebSocket clients
                self.socketio.emit('realtime_broadcast', ws_message)

            except Exception as e:
                cprint(f"❌ Error sending broadcast to WebSocket: {e}", "red")

        # Send to event callbacks
        if message.topic in self.event_callbacks:
            for callback in self.event_callbacks[message.topic]:
                try:
                    await callback(message)
                except Exception as e:
                    cprint(f"❌ Error in event callback for {message.topic}: {e}", "red")

    async def _handle_direct_message(self, message: RealTimeMessage):
        """Handle direct messages to specific agents."""
        # In a full implementation, this would route to specific agents
        # For now, we'll just log and potentially forward to WebSocket

        if self.socketio:
            try:
                ws_message = message.to_dict()
                ws_message['event_type'] = 'direct_message'

                # Emit to WebSocket (could be filtered by recipient)
                self.socketio.emit('realtime_message', ws_message)

            except Exception as e:
                cprint(f"❌ Error sending direct message to WebSocket: {e}", "red")

    def add_message_callback(self, callback: Callable) -> None:
        """Add a callback for all messages."""
        self.message_callbacks.append(callback)

    def remove_message_callback(self, callback: Callable) -> None:
        """Remove a message callback."""
        if callback in self.message_callbacks:
            self.message_callbacks.remove(callback)

    def add_event_callback(self, topic: str, callback: Callable) -> None:
        """Add a callback for specific event topics."""
        if topic not in self.event_callbacks:
            self.event_callbacks[topic] = []

        self.event_callbacks[topic].append(callback)

    def remove_event_callback(self, topic: str, callback: Callable) -> None:
        """Remove an event callback."""
        if topic in self.event_callbacks and callback in self.event_callbacks[topic]:
            self.event_callbacks[topic].remove(callback)

    def get_bus_stats(self) -> Dict[str, Any]:
        """Get real-time bus statistics."""
        return {
            'messages_processed': sum(self.message_counts.values()),
            'messages_by_topic': self.message_counts.copy(),
            'active_subscriptions': len(self.subscriptions),
            'subscribed_topics': list(self.subscriptions.keys()),
            'avg_processing_time': sum(self.processing_times) / max(len(self.processing_times), 1),
            'error_counts': self.error_counts.copy(),
            'queue_size': self.message_queue.qsize() if hasattr(self.message_queue, 'qsize') else 0,
            'uptime': time.time() - (self._processing_task.get_coro().cr_frame.f_locals.get('start_time', time.time()) if self._processing_task else time.time())
        }

    async def send_trading_signal(self, signal_type: str, symbol: str,
                                confidence: float, metadata: Dict[str, Any] = None) -> bool:
        """Send a trading signal through the real-time bus."""
        payload = {
            'signal_type': signal_type,
            'symbol': symbol,
            'confidence': confidence,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }

        return await self.broadcast_event('trading_signal', payload, MessagePriority.HIGH)

    async def send_risk_alert(self, alert_type: str, severity: str,
                            message: str, agent_id: str = None) -> bool:
        """Send a risk alert through the real-time bus."""
        payload = {
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat()
        }

        return await self.broadcast_event('risk_alert', payload, MessagePriority.CRITICAL)

    async def send_system_event(self, event_type: str, message: str,
                              details: Dict[str, Any] = None) -> bool:
        """Send a system event through the real-time bus."""
        payload = {
            'event_type': event_type,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }

        return await self.broadcast_event('system_event', payload, MessagePriority.MEDIUM)