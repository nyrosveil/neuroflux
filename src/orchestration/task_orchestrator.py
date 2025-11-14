"""
🧠 NeuroFlux Task Orchestrator
Dynamic task assignment and orchestration system for multi-agent coordination.

Built with love by Nyros Veil 🚀

Features:
- Dynamic task assignment based on agent capabilities
- Load balancing across available agents
- Dependency management and sequencing
- Failure recovery and task reassignment
- Performance-based agent selection
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

from .communication_bus import CommunicationBus, Message, MessageType, MessagePriority

# Load environment variables
load_dotenv()

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Task:
    """Represents a task to be executed by agents."""
    task_id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # Task IDs this task depends on
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: int = 300  # Estimated duration in seconds
    max_retries: int = 3
    timeout: int = 600  # Timeout in seconds

    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'priority': self.priority.value,
            'payload': self.payload,
            'dependencies': self.dependencies,
            'required_capabilities': self.required_capabilities,
            'estimated_duration': self.estimated_duration,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'status': self.status.value,
            'assigned_agent': self.assigned_agent,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'retry_count': self.retry_count,
            'result': self.result,
            'error_message': self.error_message
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        data['priority'] = TaskPriority(data['priority'])
        data['status'] = TaskStatus(data['status'])
        return cls(**data)

    def is_expired(self) -> bool:
        """Check if task has timed out."""
        if self.started_at and self.status == TaskStatus.RUNNING:
            return time.time() - self.started_at > self.timeout
        return False

    def can_start(self, completed_tasks: List[str]) -> bool:
        """Check if task can start based on dependencies."""
        return all(dep in completed_tasks for dep in self.dependencies)

@dataclass
class AgentCapability:
    """Represents an agent's capabilities and performance."""
    agent_id: str
    capabilities: List[str]
    performance_score: float = 1.0
    current_load: int = 0
    max_concurrent_tasks: int = 5
    specialization_score: Dict[str, float] = field(default_factory=dict)

    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle a specific task."""
        # Check capabilities
        if not all(cap in self.capabilities for cap in task.required_capabilities):
            return False

        # Check load
        if self.current_load >= self.max_concurrent_tasks:
            return False

        return True

    def get_task_score(self, task: Task) -> float:
        """Calculate suitability score for a task."""
        base_score = self.performance_score

        # Specialization bonus
        task_type_score = self.specialization_score.get(task.task_type, 1.0)
        base_score *= task_type_score

        # Load penalty
        load_factor = 1.0 - (self.current_load / self.max_concurrent_tasks)
        base_score *= load_factor

        # Priority bonus
        priority_multipliers = {
            TaskPriority.LOW: 1.0,
            TaskPriority.MEDIUM: 1.1,
            TaskPriority.HIGH: 1.2,
            TaskPriority.CRITICAL: 1.5
        }
        base_score *= priority_multipliers[task.priority]

        return base_score

class TaskOrchestrator:
    """
    Orchestrates task assignment and execution across multiple agents.

    Features:
    - Dynamic task assignment based on agent capabilities
    - Load balancing and performance optimization
    - Dependency management and task sequencing
    - Failure recovery and automatic reassignment
    - Real-time monitoring and analytics
    """

    def __init__(self, communication_bus: CommunicationBus, agent_registry, conflict_engine):
        self.communication_bus = communication_bus
        self.agent_registry = agent_registry
        self.conflict_engine = conflict_engine
        self.tasks: Dict[str, Task] = {}
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.completed_tasks: set = set()
        self.running = False
        self.orchestration_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

        # Statistics
        self.stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_completion_time': 0.0,
            'agent_utilization': {}
        }

    async def start(self) -> None:
        """Start the task orchestrator."""
        cprint("🎯 Starting Task Orchestrator...", "cyan")
        self.running = True
        self.orchestration_task = asyncio.create_task(self._orchestration_loop())
        cprint("✅ Task Orchestrator started", "green")

    async def stop(self) -> None:
        """Stop the task orchestrator."""
        cprint("🛑 Stopping Task Orchestrator...", "yellow")
        self.running = False
        if self.orchestration_task:
            self.orchestration_task.cancel()
            try:
                await self.orchestration_task
            except asyncio.CancelledError:
                pass
        cprint("✅ Task Orchestrator stopped", "green")

    async def register_agent(self, agent_id: str, capabilities: List[str],
                           max_concurrent_tasks: int = 5) -> None:
        """Register an agent with its capabilities."""
        async with self.lock:
            capability = AgentCapability(
                agent_id=agent_id,
                capabilities=capabilities,
                max_concurrent_tasks=max_concurrent_tasks
            )
            self.agent_capabilities[agent_id] = capability

            # Register with communication bus
            await self.communication_bus.register_agent(agent_id)

            cprint(f"📝 Agent {agent_id} registered with orchestrator", "blue")

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        async with self.lock:
            if agent_id in self.agent_capabilities:
                del self.agent_capabilities[agent_id]

            # Unregister from communication bus
            await self.communication_bus.unregister_agent(agent_id)

            # Reassign any tasks assigned to this agent
            await self._reassign_agent_tasks(agent_id)

            cprint(f"📝 Agent {agent_id} unregistered from orchestrator", "blue")

    async def submit_task(self, task_or_name, description=None, task_type=None,
                         payload=None, priority=TaskPriority.MEDIUM,
                         dependencies=None, required_capabilities=None,
                         estimated_duration=300, timeout=600) -> str:
        """Submit a new task for execution."""

        # Check if first argument is a Task object
        if isinstance(task_or_name, Task):
            task = task_or_name
            task_id = task.task_id
        else:
            # Create new task from parameters
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                name=str(task_or_name),
                description=str(description) if description else "",
                task_type=str(task_type) if task_type else "",
                priority=priority,
                payload=payload or {},
                dependencies=dependencies or [],
                required_capabilities=required_capabilities or [],
                estimated_duration=estimated_duration,
                timeout=timeout
            )

        async with self.lock:
            self.tasks[task_id] = task
            self.stats['tasks_created'] += 1

        cprint(f"📋 Task {task_id} submitted: {task.name}", "blue")
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's not completed."""
        async with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()

                    # Release agent load
                    if task.assigned_agent and task.assigned_agent in self.agent_capabilities:
                        self.agent_capabilities[task.assigned_agent].current_load -= 1

                    cprint(f"❌ Task {task_id} cancelled", "yellow")
                    return True

        return False

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task."""
        async with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
        return None

    async def _orchestration_loop(self) -> None:
        """Main orchestration loop."""
        while self.running:
            try:
                await self._process_pending_tasks()
                await self._check_running_tasks()
                await self._update_agent_performance()

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                cprint(f"❌ Orchestration loop error: {str(e)}", "red")
                await asyncio.sleep(5)

    async def _process_pending_tasks(self) -> None:
        """Process pending tasks and assign them to agents."""
        async with self.lock:
            pending_tasks = [
                task for task in self.tasks.values()
                if task.status == TaskStatus.PENDING and task.can_start(list(self.completed_tasks))
            ]

            # Sort by priority (higher priority first)
            pending_tasks.sort(key=lambda t: self._get_priority_value(t.priority), reverse=True)

            for task in pending_tasks:
                agent_id = await self._select_best_agent(task)
                if agent_id:
                    await self._assign_task_to_agent(task, agent_id)

    async def _select_best_agent(self, task: Task) -> Optional[str]:
        """Select the best agent for a task based on capabilities and performance."""
        candidates = []

        for agent_id, capability in self.agent_capabilities.items():
            if capability.can_handle_task(task):
                score = capability.get_task_score(task)
                candidates.append((agent_id, score))

        if not candidates:
            return None

        # Select agent with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def _assign_task_to_agent(self, task: Task, agent_id: str) -> None:
        """Assign a task to an agent."""
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent_id
        self.agent_capabilities[agent_id].current_load += 1

        # Send task assignment message
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id="orchestrator",
            recipient_id=agent_id,
            message_type=MessageType.COMMAND,
            priority=self._task_priority_to_message_priority(task.priority),
            topic="task_assignment",
            payload={
                'task': task.to_dict(),
                'action': 'execute_task'
            },
            timestamp=time.time()
        )

        await self.communication_bus.send_message(message)
        cprint(f"👤 Task {task.task_id} assigned to agent {agent_id}", "green")

    async def _check_running_tasks(self) -> None:
        """Check status of running tasks and handle timeouts/failures."""
        async with self.lock:
            running_tasks = [
                task for task in self.tasks.values()
                if task.status == TaskStatus.RUNNING
            ]

            for task in running_tasks:
                # Check for timeout
                if task.is_expired():
                    await self._handle_task_timeout(task)
                    continue

                # Check for completion/failure messages
                await self._check_task_messages(task)

    async def _check_task_messages(self, task: Task) -> None:
        """Check for task completion/failure messages from agents."""
        if not task.assigned_agent:
            return

        # Get messages for the assigned agent
        messages = await self.communication_bus.get_agent_messages(task.assigned_agent)

        for message in messages:
            if (message.topic == "task_result" and
                message.payload.get('task_id') == task.task_id):

                if message.payload.get('status') == 'completed':
                    await self._handle_task_completion(task, message.payload.get('result', {}))
                elif message.payload.get('status') == 'failed':
                    await self._handle_task_failure(task, message.payload.get('error', 'Unknown error'))

    async def _handle_task_completion(self, task: Task, result: Dict[str, Any]) -> None:
        """Handle successful task completion."""
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result

        # Update agent load and performance
        if task.assigned_agent and task.assigned_agent in self.agent_capabilities:
            agent = self.agent_capabilities[task.assigned_agent]
            agent.current_load -= 1

            # Update performance score based on completion time
            if task.started_at:
                actual_duration = task.completed_at - task.started_at
                efficiency = task.estimated_duration / actual_duration if actual_duration > 0 else 1.0
                agent.performance_score = (agent.performance_score + efficiency) / 2

        # Add to completed tasks
        self.completed_tasks.add(task.task_id)

        # Update statistics
        self.stats['tasks_completed'] += 1
        if task.started_at and task.completed_at:
            completion_time = task.completed_at - task.started_at
            self.stats['avg_completion_time'] = (
                (self.stats['avg_completion_time'] * (self.stats['tasks_completed'] - 1)) +
                completion_time
            ) / self.stats['tasks_completed']

        cprint(f"✅ Task {task.task_id} completed by {task.assigned_agent}", "green")

    async def _handle_task_failure(self, task: Task, error_message: str) -> None:
        """Handle task failure."""
        task.error_message = error_message

        # Update agent load
        if task.assigned_agent and task.assigned_agent in self.agent_capabilities:
            agent = self.agent_capabilities[task.assigned_agent]
            agent.current_load -= 1
            # Penalize performance score for failure
            agent.performance_score *= 0.9

        # Retry or mark as failed
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.assigned_agent = None
            cprint(f"🔄 Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})", "yellow")
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            self.stats['tasks_failed'] += 1
            cprint(f"❌ Task {task.task_id} failed permanently: {error_message}", "red")

    async def _handle_task_timeout(self, task: Task) -> None:
        """Handle task timeout."""
        task.status = TaskStatus.TIMEOUT
        task.completed_at = time.time()
        task.error_message = "Task timed out"

        # Update agent load and penalize performance
        if task.assigned_agent and task.assigned_agent in self.agent_capabilities:
            agent = self.agent_capabilities[task.assigned_agent]
            agent.current_load -= 1
            agent.performance_score *= 0.8  # Heavy penalty for timeout

        self.stats['tasks_failed'] += 1
        cprint(f"⏰ Task {task.task_id} timed out", "red")

    async def _reassign_agent_tasks(self, agent_id: str) -> None:
        """Reassign tasks from a failed agent."""
        async with self.lock:
            affected_tasks = [
                task for task in self.tasks.values()
                if task.assigned_agent == agent_id and task.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]
            ]

            for task in affected_tasks:
                task.assigned_agent = None
                task.status = TaskStatus.PENDING
                cprint(f"🔄 Task {task.task_id} reassigned due to agent {agent_id} failure", "yellow")

    async def _update_agent_performance(self) -> None:
        """Update agent utilization statistics."""
        total_load = 0
        total_capacity = 0

        for agent_id, capability in self.agent_capabilities.items():
            total_load += capability.current_load
            total_capacity += capability.max_concurrent_tasks
            utilization = capability.current_load / capability.max_concurrent_tasks if capability.max_concurrent_tasks > 0 else 0
            self.stats['agent_utilization'][agent_id] = utilization

        # Update overall utilization
        self.stats['overall_utilization'] = total_load / total_capacity if total_capacity > 0 else 0

    def _get_priority_value(self, priority: TaskPriority) -> int:
        """Convert priority to numeric value."""
        priority_map = {
            TaskPriority.LOW: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.CRITICAL: 4
        }
        return priority_map[priority]

    def _task_priority_to_message_priority(self, task_priority: TaskPriority) -> MessagePriority:
        """Convert task priority to message priority."""
        mapping = {
            TaskPriority.LOW: MessagePriority.LOW,
            TaskPriority.MEDIUM: MessagePriority.MEDIUM,
            TaskPriority.HIGH: MessagePriority.HIGH,
            TaskPriority.CRITICAL: MessagePriority.CRITICAL
        }
        return mapping[task_priority]

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        async def _get_stats():
            async with self.lock:
                return {
                    **self.stats,
                    'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                    'running_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
                    'completed_tasks': len(self.completed_tasks),
                    'registered_agents': len(self.agent_capabilities)
                }

        # Since this is called from sync context, we need to handle it differently
        # For now, return a copy of current stats
        return self.stats.copy()