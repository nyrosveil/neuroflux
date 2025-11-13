"""
🧠 NeuroFlux Orchestration Package
Agent orchestration and communication system for Phase 3.2

Built with love by Nyros Veil 🚀

Inter-agent communication, task orchestration, conflict resolution, and agent registry.
"""

from .communication_bus import CommunicationBus, Message, MessageType, MessagePriority
from .task_orchestrator import TaskOrchestrator
from .conflict_resolution import ConflictResolutionEngine
from .agent_registry import AgentRegistry, AgentInfo, AgentStatus, AgentCapability, ServiceQuery, HealthCheck

__all__ = [
    'CommunicationBus',
    'Message',
    'MessageType',
    'MessagePriority',
    'TaskOrchestrator',
    'ConflictResolutionEngine',
    'AgentRegistry',
    'AgentInfo',
    'AgentStatus',
    'AgentCapability',
    'ServiceQuery',
    'HealthCheck'
]