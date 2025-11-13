"""
🧠 NeuroFlux Orchestration Package
Agent orchestration and communication system for Phase 3.2

Built with love by Nyros Veil 🚀

Inter-agent communication, task orchestration, and swarm intelligence.
"""

from .communication_bus import CommunicationBus
from .task_orchestrator import TaskOrchestrator

__all__ = [
    'CommunicationBus',
    'TaskOrchestrator'
]