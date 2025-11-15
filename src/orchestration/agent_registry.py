"""
🗂️ NeuroFlux Agent Registry & Discovery System
Dynamic agent registration, service discovery, and health monitoring.

Built with love by Nyros Veil 🚀

Features:
- Dynamic agent registration and deregistration
- Service discovery by agent type, capabilities, and performance
- Health monitoring with heartbeat system
- Performance metrics tracking and analytics
- Auto-scaling integration with load balancing
- Fault-tolerant agent lifecycle management
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

from .communication_bus import CommunicationBus, Message, MessageType, MessagePriority

# Load environment variables
load_dotenv()


class AgentStatus(Enum):
    """Agent lifecycle states."""
    REGISTERING = "registering"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    DEREGISTERING = "deregistering"
    OFFLINE = "offline"


class AgentCapability(Enum):
    """Agent capability types."""
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
    ML_PREDICTION = "ml_prediction"
    TIME_SERIES_ANALYSIS = "time_series_analysis"


@dataclass
class AgentInfo:
    """Information about a registered agent."""
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


@dataclass
class ServiceQuery:
    """Query for finding agents by criteria."""
    capabilities: Optional[Set[AgentCapability]] = None
    agent_type: Optional[str] = None
    min_health_score: float = 0.5
    max_load_factor: float = 0.8
    tags: Optional[Set[str]] = None
    limit: int = 10
    sort_by: str = "performance"  # performance, health, load, random


@dataclass
class HealthCheck:
    """Health check result for an agent."""
    agent_id: str
    timestamp: float
    status: str  # healthy, degraded, unhealthy
    response_time: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Central registry for agent discovery and management.

    Features:
    - Agent registration/deregistration with validation
    - Service discovery with intelligent ranking
    - Health monitoring and automatic status updates
    - Performance metrics collection and analysis
    - Auto-scaling triggers and load balancing
    """

    def __init__(self, communication_bus: CommunicationBus):
        self.communication_bus = communication_bus
        self.agents: Dict[str, AgentInfo] = {}
        self.health_history: List[HealthCheck] = []
        self.performance_stats: Dict[str, Dict[str, float]] = {}

        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.heartbeat_timeout = 90  # seconds
        self.health_monitoring_task: Optional[asyncio.Task] = None

        # Auto-scaling
        self.auto_scaling_enabled = True
        self.min_agents_per_capability = 2
        self.max_agents_per_capability = 10
        self.scaling_thresholds = {
            'high_load': 0.8,
            'low_load': 0.3,
            'health_degraded': 0.6
        }

        # Statistics
        self.stats = {
            'total_registrations': 0,
            'active_agents': 0,
            'health_checks_performed': 0,
            'failed_health_checks': 0,
            'auto_scaling_events': 0,
            'service_discovery_requests': 0
        }

        self.running = False

    async def start(self) -> None:
        """Start the agent registry system."""
        cprint("🗂️ Starting Agent Registry & Discovery System...", "cyan")
        self.running = True
        self.health_monitoring_task = asyncio.create_task(self._health_monitoring_loop())
        cprint("✅ Agent Registry & Discovery System started", "green")

    async def stop(self) -> None:
        """Stop the agent registry system."""
        cprint("🛑 Stopping Agent Registry & Discovery System...", "yellow")
        self.running = False
        if self.health_monitoring_task:
            self.health_monitoring_task.cancel()
            try:
                await self.health_monitoring_task
            except asyncio.CancelledError:
                pass
            self.health_monitoring_task = None
        cprint("✅ Agent Registry & Discovery System stopped", "green")

    async def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """
        Register a new agent with the system.

        Args:
            agent_info: Agent registration information

        Returns:
            Agent ID assigned to the registered agent
        """
        agent_id = agent_info.get('agent_id', str(uuid.uuid4()))

        # Validate agent info
        if not self._validate_agent_info(agent_info):
            raise ValueError("Invalid agent registration information")

        # Create agent record
        agent = AgentInfo(
            agent_id=agent_id,
            agent_type=agent_info['agent_type'],
            capabilities=set(agent_info.get('capabilities', [])),
            status=AgentStatus.ACTIVE,  # Agents start as active after successful registration
            metadata=agent_info.get('metadata', {}),
            version=agent_info.get('version', '1.0.0'),
            tags=set(agent_info.get('tags', []))
        )

        # Register agent
        self.agents[agent_id] = agent
        self.stats['total_registrations'] += 1
        self.stats['active_agents'] += 1

        # Initialize performance tracking
        self.performance_stats[agent_id] = {
            'requests_served': 0,
            'avg_response_time': 0.0,
            'success_rate': 1.0,
            'uptime': 0.0,
            'last_updated': time.time()
        }

        cprint(f"✅ Agent {agent_id} ({agent.agent_type}) registered successfully", "green")

        # Broadcast registration event
        await self.communication_bus.broadcast_message(
            sender_id="agent_registry",
            topic="agent_registered",
            payload={
                'agent_id': agent_id,
                'agent_type': agent.agent_type,
                'capabilities': list(agent.capabilities),
                'timestamp': time.time()
            }
        )

        return agent_id

    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the system.

        Args:
            agent_id: ID of the agent to deregister

        Returns:
            True if deregistration successful, False otherwise
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        agent.status = AgentStatus.DEREGISTERING

        # Clean up resources
        if agent_id in self.performance_stats:
            del self.performance_stats[agent_id]

        # Remove from registry
        del self.agents[agent_id]
        self.stats['active_agents'] -= 1

        cprint(f"✅ Agent {agent_id} ({agent.agent_type}) deregistered successfully", "yellow")

        # Broadcast deregistration event
        await self.communication_bus.broadcast_message(
            sender_id="agent_registry",
            topic="agent_deregistered",
            payload={
                'agent_id': agent_id,
                'agent_type': agent.agent_type,
                'timestamp': time.time()
            }
        )

        return True

    async def discover_agents(self, query: ServiceQuery) -> List[AgentInfo]:
        """
        Discover agents matching the given criteria.

        Args:
            query: Service discovery query

        Returns:
            List of matching agents, ranked by relevance
        """
        self.stats['service_discovery_requests'] += 1

        # Find matching agents
        candidates = []
        for agent in self.agents.values():
            if agent.status != AgentStatus.ACTIVE:
                continue

            # Check capabilities
            if query.capabilities and not query.capabilities.issubset(agent.capabilities):
                continue

            # Check agent type
            if query.agent_type and agent.agent_type != query.agent_type:
                continue

            # Check health score
            if agent.health_score < query.min_health_score:
                continue

            # Check load factor
            if agent.load_factor > query.max_load_factor:
                continue

            # Check tags
            if query.tags and not query.tags.issubset(agent.tags):
                continue

            candidates.append(agent)

        # Sort candidates by query criteria
        candidates = self._rank_agents(candidates, query.sort_by)

        # Apply limit
        result = candidates[:query.limit]

        cprint(f"🔍 Service discovery: found {len(result)} agents matching criteria", "blue")

        return result

    async def update_agent_health(self, agent_id: str, health_data: Dict[str, Any]) -> None:
        """
        Update agent health information.

        Args:
            agent_id: Agent ID
            health_data: Health metrics and status
        """
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]

        # Update health score
        agent.health_score = health_data.get('health_score', agent.health_score)
        agent.load_factor = health_data.get('load_factor', agent.load_factor)
        agent.last_heartbeat = time.time()

        # Update status based on health
        if agent.health_score >= 0.8:
            agent.status = AgentStatus.ACTIVE
        elif agent.health_score >= 0.5:
            agent.status = AgentStatus.DEGRADED
        else:
            agent.status = AgentStatus.SUSPENDED

        # Record health check
        health_check = HealthCheck(
            agent_id=agent_id,
            timestamp=time.time(),
            status="healthy" if agent.health_score >= 0.5 else "unhealthy",
            response_time=health_data.get('response_time', 0.0),
            metrics=health_data
        )
        self.health_history.append(health_check)
        self.stats['health_checks_performed'] += 1

        # Check for auto-scaling triggers
        await self._check_auto_scaling_triggers()

    async def update_agent_performance(self, agent_id: str, metrics: Dict[str, float]) -> None:
        """
        Update agent performance metrics.

        Args:
            agent_id: Agent ID
            metrics: Performance metrics
        """
        if agent_id not in self.performance_stats:
            return

        stats = self.performance_stats[agent_id]

        # Update metrics
        for key, value in metrics.items():
            if key in stats:
                # Calculate running average for some metrics
                if key.startswith('avg_'):
                    old_value = stats[key]
                    count = max(stats.get('requests_served', 1), 1)  # Ensure count is at least 1
                    if old_value == 0.0:  # First update
                        stats[key] = value
                    else:
                        stats[key] = (old_value * (count - 1) + value) / count
                else:
                    stats[key] = value

        stats['last_updated'] = time.time()

    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get information about a specific agent."""
        return self.agents.get(agent_id)

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self.stats,
            'registered_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
            'degraded_agents': len([a for a in self.agents.values() if a.status == AgentStatus.DEGRADED]),
            'capabilities_distribution': self._get_capabilities_distribution(),
            'health_distribution': self._get_health_distribution()
        }

    def _validate_agent_info(self, agent_info: Dict[str, Any]) -> bool:
        """Validate agent registration information."""
        required_fields = ['agent_type', 'capabilities']
        for field in required_fields:
            if field not in agent_info:
                return False

        # Validate capabilities (case-insensitive)
        try:
            capabilities = [AgentCapability(cap.lower()) for cap in agent_info['capabilities']]
            agent_info['capabilities'] = capabilities
        except ValueError:
            return False

        return True

    def _rank_agents(self, agents: List[AgentInfo], sort_by: str) -> List[AgentInfo]:
        """Rank agents based on the specified criteria."""
        if sort_by == "performance":
            # Rank by health score and inverse load factor
            def score(agent):
                perf_stats = self.performance_stats.get(agent.agent_id, {})
                success_rate = perf_stats.get('success_rate', 0.5)
                return (agent.health_score * 0.4) + ((1 - agent.load_factor) * 0.3) + (success_rate * 0.3)

        elif sort_by == "health":
            def score(agent):
                return agent.health_score

        elif sort_by == "load":
            def score(agent):
                return -agent.load_factor  # Lower load is better

        else:  # random or unknown
            import random
            random.shuffle(agents)
            return agents

        return sorted(agents, key=score, reverse=True)

    async def _health_monitoring_loop(self) -> None:
        """Monitor agent health and handle timeouts."""
        while self.running:
            try:
                current_time = time.time()

                # Check for heartbeat timeouts
                timeout_agents = []
                for agent_id, agent in self.agents.items():
                    if current_time - agent.last_heartbeat > self.heartbeat_timeout:
                        timeout_agents.append(agent_id)

                # Mark timed out agents as offline
                for agent_id in timeout_agents:
                    agent = self.agents[agent_id]
                    agent.status = AgentStatus.OFFLINE
                    self.stats['failed_health_checks'] += 1

                    cprint(f"⚠️ Agent {agent_id} marked offline due to heartbeat timeout", "yellow")

                    # Broadcast offline event
                    await self.communication_bus.broadcast_message(
                        sender_id="agent_registry",
                        topic="agent_offline",
                        payload={
                            'agent_id': agent_id,
                            'reason': 'heartbeat_timeout',
                            'timestamp': time.time()
                        }
                    )

                # Perform active health checks
                await self._perform_health_checks()

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                cprint(f"❌ Health monitoring error: {str(e)}", "red")
                await asyncio.sleep(30)

    async def _perform_health_checks(self) -> None:
        """Perform active health checks on registered agents."""
        for agent_id, agent in list(self.agents.items()):
            if agent.status in [AgentStatus.OFFLINE, AgentStatus.DEREGISTERING]:
                continue

            try:
                # Send health check request
                start_time = time.time()
                response = await self.communication_bus.send_request(
                    sender_id="agent_registry",
                    recipient_id=agent_id,
                    topic="health_check",
                    payload={'timestamp': start_time},
                    timeout=10
                )
                response_time = time.time() - start_time

                if response and 'status' in response:
                    # Update health based on response
                    health_score = response.get('health_score', 0.5)
                    load_factor = response.get('load_factor', 0.0)

                    await self.update_agent_health(agent_id, {
                        'health_score': health_score,
                        'load_factor': load_factor,
                        'response_time': response_time,
                        'status': response['status']
                    })
                else:
                    # Health check failed
                    self.stats['failed_health_checks'] += 1
                    await self.update_agent_health(agent_id, {
                        'health_score': max(0, agent.health_score - 0.1),
                        'response_time': response_time
                    })

            except Exception as e:
                # Health check failed
                self.stats['failed_health_checks'] += 1
                cprint(f"⚠️ Health check failed for agent {agent_id}: {str(e)}", "yellow")

    async def _check_auto_scaling_triggers(self) -> None:
        """Check for auto-scaling triggers and initiate scaling actions."""
        if not self.auto_scaling_enabled:
            return

        # Analyze load distribution by capability
        capability_loads = {}
        for agent in self.agents.values():
            if agent.status != AgentStatus.ACTIVE:
                continue

            for capability in agent.capabilities:
                if capability not in capability_loads:
                    capability_loads[capability] = []
                capability_loads[capability].append(agent.load_factor)

        # Check scaling triggers
        for capability, loads in capability_loads.items():
            if not loads:
                continue

            avg_load = sum(loads) / len(loads)
            agent_count = len(loads)

            # Scale up if high load and not at max
            if (avg_load > self.scaling_thresholds['high_load'] and
                agent_count < self.max_agents_per_capability):

                await self._trigger_scale_up(capability, avg_load)

            # Scale down if low load and above minimum
            elif (avg_load < self.scaling_thresholds['low_load'] and
                  agent_count > self.min_agents_per_capability):

                await self._trigger_scale_down(capability, avg_load)

    async def _trigger_scale_up(self, capability: AgentCapability, current_load: float) -> None:
        """Trigger scale up for a capability."""
        self.stats['auto_scaling_events'] += 1

        cprint(f"📈 Scaling up {capability.value}: load {current_load:.2f}", "green")

        # Broadcast scale up event
        await self.communication_bus.broadcast_message(
            sender_id="agent_registry",
            topic="scale_up_triggered",
            payload={
                'capability': capability.value,
                'current_load': current_load,
                'timestamp': time.time()
            }
        )

    async def _trigger_scale_down(self, capability: AgentCapability, current_load: float) -> None:
        """Trigger scale down for a capability."""
        self.stats['auto_scaling_events'] += 1

        cprint(f"📉 Scaling down {capability.value}: load {current_load:.2f}", "yellow")

        # Broadcast scale down event
        await self.communication_bus.broadcast_message(
            sender_id="agent_registry",
            topic="scale_down_triggered",
            payload={
                'capability': capability.value,
                'current_load': current_load,
                'timestamp': time.time()
            }
        )

    def _get_capabilities_distribution(self) -> Dict[str, int]:
        """Get distribution of agents by capability."""
        distribution = {}
        for agent in self.agents.values():
            for capability in agent.capabilities:
                cap_name = capability.value
                distribution[cap_name] = distribution.get(cap_name, 0) + 1
        return distribution

    def _get_health_distribution(self) -> Dict[str, int]:
        """Get distribution of agents by health status."""
        distribution = {'healthy': 0, 'degraded': 0, 'unhealthy': 0}
        for agent in self.agents.values():
            if agent.health_score >= 0.8:
                distribution['healthy'] += 1
            elif agent.health_score >= 0.5:
                distribution['degraded'] += 1
            else:
                distribution['unhealthy'] += 1
        return distribution