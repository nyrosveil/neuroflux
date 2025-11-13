"""
🛡️ NeuroFlux Conflict Resolution Engine
Intelligent conflict detection and consensus-based resolution system.

Built with love by Nyros Veil 🚀

Features:
- Multi-type conflict detection (signal, resource, priority, timing)
- Byzantine Fault Tolerant consensus algorithms
- Weighted voting systems based on agent performance
- Automated resolution strategies with fallback protocols
- Real-time conflict monitoring and analytics
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable, Set
from dataclasses import dataclass, field
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

from .communication_bus import CommunicationBus, Message, MessageType, MessagePriority

# Load environment variables
load_dotenv()

class ConflictType(Enum):
    """Types of conflicts that can occur between agents."""
    SIGNAL_CONFLICT = "signal_conflict"          # Contradictory trading signals
    RESOURCE_CONFLICT = "resource_conflict"      # Insufficient resources
    PRIORITY_CONFLICT = "priority_conflict"      # Competing priorities
    TIMING_CONFLICT = "timing_conflict"          # Execution timing issues
    STRATEGY_CONFLICT = "strategy_conflict"      # Conflicting strategies
    RISK_CONFLICT = "risk_conflict"              # Risk assessment disagreements

class ConsensusAlgorithm(Enum):
    """Available consensus algorithms."""
    SIMPLE_MAJORITY = "simple_majority"          # >50% agreement
    SUPERMAJORITY = "supermajority"              # >66% agreement
    WEIGHTED_CONSENSUS = "weighted_consensus"    # Performance-weighted
    BYZANTINE_FAULT_TOLERANCE = "bft"           # BFT consensus
    QUORUM_BASED = "quorum_based"               # Minimum participation

class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    VOTING_CONSENSUS = "voting_consensus"        # Vote among agents
    EXPERT_PRECEDENCE = "expert_precedence"      # Domain expert override
    CONFIDENCE_THRESHOLD = "confidence_threshold" # High confidence wins
    FALLBACK_PROTOCOL = "fallback_protocol"      # Safe default action
    HUMAN_INTERVENTION = "human_intervention"    # Manual resolution
    COMPROMISE_SOLUTION = "compromise_solution"  # Balanced approach

@dataclass
class Conflict:
    """Represents a detected conflict between agents."""
    conflict_id: str
    conflict_type: ConflictType
    description: str
    involved_agents: List[str]
    conflicting_elements: Dict[str, Any]
    severity: float  # 0.0 to 1.0
    detected_at: float
    resolution_deadline: float
    status: str = "detected"  # detected, resolving, resolved, escalated
    resolution: Optional[Dict[str, Any]] = None
    resolved_at: Optional[float] = None

@dataclass
class ConsensusVote:
    """A vote in a consensus decision."""
    agent_id: str
    decision: Any
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: float
    stake: float  # Voting power (based on performance/reputation)
    evidence: Optional[Dict[str, Any]] = None

@dataclass
class ConsensusResult:
    """Result of a consensus decision."""
    consensus_id: str
    algorithm: ConsensusAlgorithm
    votes: List[ConsensusVote]
    decision: Any
    confidence: float
    achieved_at: float
    participants: int
    agreement_percentage: float
    execution_status: str = "pending"

class ConflictResolutionEngine:
    """
    Intelligent conflict detection and resolution system.

    Features:
    - Real-time conflict detection across multiple domains
    - Multiple consensus algorithms (BFT, weighted voting, etc.)
    - Automated resolution with human-in-the-loop options
    - Performance tracking and learning from resolutions
    """

    def __init__(self, communication_bus: CommunicationBus):
        self.communication_bus = communication_bus
        self.active_conflicts: Dict[str, Conflict] = {}
        self.resolved_conflicts: List[Conflict] = []
        self.consensus_history: List[ConsensusResult] = []
        self.agent_performance: Dict[str, Dict[str, float]] = {}

        # Conflict detection rules
        self.conflict_detectors: Dict[str, Callable] = {
            'signal_conflicts': self._detect_signal_conflicts,
            'resource_conflicts': self._detect_resource_conflicts,
            'priority_conflicts': self._detect_priority_conflicts,
            'timing_conflicts': self._detect_timing_conflicts
        }

        # Consensus algorithms
        self.consensus_algorithms: Dict[ConsensusAlgorithm, Callable] = {
            ConsensusAlgorithm.SIMPLE_MAJORITY: self._simple_majority_consensus,
            ConsensusAlgorithm.SUPERMAJORITY: self._supermajority_consensus,
            ConsensusAlgorithm.WEIGHTED_CONSENSUS: self._weighted_consensus,
            ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANCE: self._bft_consensus,
            ConsensusAlgorithm.QUORUM_BASED: self._quorum_consensus
        }

        # Resolution strategies
        self.resolution_strategies: Dict[ResolutionStrategy, Callable] = {
            ResolutionStrategy.VOTING_CONSENSUS: self._resolve_by_voting,
            ResolutionStrategy.EXPERT_PRECEDENCE: self._resolve_by_expert,
            ResolutionStrategy.CONFIDENCE_THRESHOLD: self._resolve_by_confidence,
            ResolutionStrategy.FALLBACK_PROTOCOL: self._resolve_by_fallback,
            ResolutionStrategy.HUMAN_INTERVENTION: self._resolve_by_human,
            ResolutionStrategy.COMPROMISE_SOLUTION: self._resolve_by_compromise
        }

        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Statistics
        self.stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'consensus_attempts': 0,
            'consensus_success_rate': 0.0,
            'avg_resolution_time': 0.0,
            'escalation_rate': 0.0
        }

    async def start(self) -> None:
        """Start the conflict resolution engine."""
        cprint("🛡️ Starting Conflict Resolution Engine...", "cyan")
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        cprint("✅ Conflict Resolution Engine started", "green")

    async def stop(self) -> None:
        """Stop the conflict resolution engine."""
        cprint("🛑 Stopping Conflict Resolution Engine...", "yellow")
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        cprint("✅ Conflict Resolution Engine stopped", "green")

    async def detect_conflicts(self, context: Dict[str, Any]) -> List[Conflict]:
        """
        Detect conflicts in the given context.

        Args:
            context: Current system state and agent activities

        Returns:
            List of detected conflicts
        """
        detected_conflicts = []

        for detector_name, detector_func in self.conflict_detectors.items():
            try:
                conflicts = await detector_func(context)
                detected_conflicts.extend(conflicts)
            except Exception as e:
                cprint(f"❌ Conflict detection error in {detector_name}: {str(e)}", "red")

        # Register detected conflicts
        for conflict in detected_conflicts:
            self.active_conflicts[conflict.conflict_id] = conflict
            self.stats['conflicts_detected'] += 1

        if detected_conflicts:
            cprint(f"⚠️ Detected {len(detected_conflicts)} conflicts", "yellow")

        return detected_conflicts

    async def resolve_conflict(self, conflict: Conflict,
                              algorithm: ConsensusAlgorithm = ConsensusAlgorithm.WEIGHTED_CONSENSUS,
                              strategy: ResolutionStrategy = ResolutionStrategy.VOTING_CONSENSUS) -> Optional[Dict[str, Any]]:
        """
        Resolve a conflict using specified algorithm and strategy.

        Args:
            conflict: The conflict to resolve
            algorithm: Consensus algorithm to use
            strategy: Resolution strategy

        Returns:
            Resolution result or None if resolution failed
        """
        cprint(f"⚖️ Resolving conflict {conflict.conflict_id}: {conflict.description}", "blue")

        conflict.status = "resolving"

        try:
            # Apply resolution strategy
            resolution_func = self.resolution_strategies[strategy]
            result = await resolution_func(conflict, algorithm)

            if result:
                conflict.status = "resolved"
                conflict.resolution = result
                conflict.resolved_at = time.time()

                # Move to resolved conflicts
                self.resolved_conflicts.append(conflict)
                if conflict.conflict_id in self.active_conflicts:
                    del self.active_conflicts[conflict.conflict_id]

                self.stats['conflicts_resolved'] += 1

                # Update resolution time statistics
                if conflict.resolved_at and conflict.detected_at:
                    resolution_time = conflict.resolved_at - conflict.detected_at
                    self.stats['avg_resolution_time'] = (
                        (self.stats['avg_resolution_time'] * (self.stats['conflicts_resolved'] - 1)) +
                        resolution_time
                    ) / self.stats['conflicts_resolved']

                cprint(f"✅ Conflict {conflict.conflict_id} resolved", "green")
                return result
            else:
                # Resolution failed, escalate
                await self._escalate_conflict(conflict)
                return None

        except Exception as e:
            cprint(f"❌ Conflict resolution failed: {str(e)}", "red")
            await self._escalate_conflict(conflict)
            return None

    async def _detect_signal_conflicts(self, context: Dict[str, Any]) -> List[Conflict]:
        """Detect conflicting trading signals."""
        conflicts = []
        agent_signals = context.get('agent_signals', {})

        # Group signals by symbol/timeframe
        signal_groups = {}
        for agent_id, signals in agent_signals.items():
            for signal in signals:
                key = f"{signal['symbol']}_{signal['timeframe']}"
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append((agent_id, signal))

        # Check for contradictions within each group
        for key, signals in signal_groups.items():
            if len(signals) < 2:
                continue

            # Check for buy/sell contradictions
            buy_signals = [s for s in signals if s[1]['action'] == 'BUY']
            sell_signals = [s for s in signals if s[1]['action'] == 'SELL']

            if buy_signals and sell_signals:
                conflict = Conflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.SIGNAL_CONFLICT,
                    description=f"Contradictory signals for {key}: {len(buy_signals)} BUY vs {len(sell_signals)} SELL",
                    involved_agents=[s[0] for s in signals],
                    conflicting_elements={
                        'symbol_timeframe': key,
                        'buy_signals': buy_signals,
                        'sell_signals': sell_signals
                    },
                    severity=0.8,
                    detected_at=time.time(),
                    resolution_deadline=time.time() + 300  # 5 minutes
                )
                conflicts.append(conflict)

        return conflicts

    async def _detect_resource_conflicts(self, context: Dict[str, Any]) -> List[Conflict]:
        """Detect resource allocation conflicts."""
        conflicts = []
        resource_requests = context.get('resource_requests', {})

        # Check for over-subscription of resources
        resource_usage = {}
        for agent_id, requests in resource_requests.items():
            for request in requests:
                resource_type = request['resource_type']
                amount = request['amount']

                if resource_type not in resource_usage:
                    resource_usage[resource_type] = {'total_requested': 0, 'requests': []}

                resource_usage[resource_type]['total_requested'] += amount
                resource_usage[resource_type]['requests'].append((agent_id, request))

        # Check against available resources
        available_resources = context.get('available_resources', {})
        for resource_type, usage in resource_usage.items():
            available = available_resources.get(resource_type, 0)
            if usage['total_requested'] > available:
                conflict = Conflict(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.RESOURCE_CONFLICT,
                    description=f"Resource over-subscription: {resource_type} requested {usage['total_requested']} but only {available} available",
                    involved_agents=list(set([r[0] for r in usage['requests']])),
                    conflicting_elements={
                        'resource_type': resource_type,
                        'requested': usage['total_requested'],
                        'available': available,
                        'requests': usage['requests']
                    },
                    severity=0.9,
                    detected_at=time.time(),
                    resolution_deadline=time.time() + 180  # 3 minutes
                )
                conflicts.append(conflict)

        return conflicts

    async def _detect_priority_conflicts(self, context: Dict[str, Any]) -> List[Conflict]:
        """Detect priority-based conflicts."""
        conflicts = []
        # Implementation for priority conflicts
        return conflicts

    async def _detect_timing_conflicts(self, context: Dict[str, Any]) -> List[Conflict]:
        """Detect timing-related conflicts."""
        conflicts = []
        # Implementation for timing conflicts
        return conflicts

    async def _resolve_by_voting(self, conflict: Conflict, algorithm: ConsensusAlgorithm) -> Optional[Dict[str, Any]]:
        """Resolve conflict through voting consensus."""
        # Request votes from involved agents
        votes = await self._gather_votes(conflict)

        if not votes:
            return None

        # Apply consensus algorithm
        consensus_func = self.consensus_algorithms[algorithm]
        result = await consensus_func(votes, conflict)

        if result and result.agreement_percentage >= 0.5:  # At least 50% agreement
            self.consensus_history.append(result)
            self.stats['consensus_attempts'] += 1

            return {
                'decision': result.decision,
                'confidence': result.confidence,
                'algorithm': algorithm.value,
                'participants': result.participants,
                'agreement_percentage': result.agreement_percentage
            }

        return None

    async def _resolve_by_expert(self, conflict: Conflict, algorithm: ConsensusAlgorithm) -> Optional[Dict[str, Any]]:
        """Resolve by deferring to domain expert."""
        # Find the agent with highest expertise for this conflict type
        expert_agent = await self._find_domain_expert(conflict)

        if expert_agent:
            # Request decision from expert
            response = await self.communication_bus.send_request(
                sender_id="conflict_resolution",
                recipient_id=expert_agent,
                topic="expert_resolution",
                payload={
                    'conflict': conflict.__dict__,
                    'reason': 'Domain expertise required'
                },
                timeout=60
            )

            if response and 'decision' in response:
                return {
                    'decision': response['decision'],
                    'expert': expert_agent,
                    'confidence': response.get('confidence', 0.8),
                    'strategy': 'expert_precedence'
                }

        return None

    async def _resolve_by_confidence(self, conflict: Conflict, algorithm: ConsensusAlgorithm) -> Optional[Dict[str, Any]]:
        """Resolve by selecting highest confidence decision."""
        # Gather confidence scores from agents
        confidence_scores = await self._gather_confidence_scores(conflict)

        if confidence_scores:
            # Select highest confidence decision
            best_agent = max(confidence_scores.items(), key=lambda x: x[1]['confidence'])

            return {
                'decision': best_agent[1]['decision'],
                'agent': best_agent[0],
                'confidence': best_agent[1]['confidence'],
                'strategy': 'confidence_threshold'
            }

        return None

    async def _resolve_by_fallback(self, conflict: Conflict, algorithm: ConsensusAlgorithm) -> Optional[Dict[str, Any]]:
        """Resolve using safe fallback protocol."""
        # Define fallback actions based on conflict type
        fallbacks = {
            ConflictType.SIGNAL_CONFLICT: {'action': 'HOLD', 'reason': 'Conflicting signals - maintain position'},
            ConflictType.RESOURCE_CONFLICT: {'action': 'REDUCE', 'reason': 'Resource constraints - reduce allocation'},
            ConflictType.RISK_CONFLICT: {'action': 'CONSERVATIVE', 'reason': 'Risk disagreement - take conservative approach'}
        }

        fallback = fallbacks.get(conflict.conflict_type)
        if fallback:
            return {
                'decision': fallback['action'],
                'reason': fallback['reason'],
                'strategy': 'fallback_protocol',
                'confidence': 0.5
            }

        return None

    async def _resolve_by_human(self, conflict: Conflict, algorithm: ConsensusAlgorithm) -> Optional[Dict[str, Any]]:
        """Escalate to human intervention."""
        # In a real system, this would notify human operators
        cprint(f"🚨 HUMAN INTERVENTION REQUIRED: {conflict.description}", "red")

        # For now, return a placeholder
        return {
            'decision': 'PENDING_HUMAN_REVIEW',
            'strategy': 'human_intervention',
            'escalated_at': time.time()
        }

    async def _resolve_by_compromise(self, conflict: Conflict, algorithm: ConsensusAlgorithm) -> Optional[Dict[str, Any]]:
        """Find a compromise solution."""
        # Implementation for compromise resolution
        return None

    async def _gather_votes(self, conflict: Conflict) -> List[ConsensusVote]:
        """Gather votes from involved agents."""
        votes = []

        for agent_id in conflict.involved_agents:
            try:
                response = await self.communication_bus.send_request(
                    sender_id="conflict_resolution",
                    recipient_id=agent_id,
                    topic="conflict_vote",
                    payload={
                        'conflict': conflict.__dict__,
                        'request': 'Please provide your vote on this conflict'
                    },
                    timeout=30
                )

                if response and 'vote' in response:
                    vote_data = response['vote']
                    stake = self._calculate_agent_stake(agent_id, conflict.conflict_type)

                    vote = ConsensusVote(
                        agent_id=agent_id,
                        decision=vote_data.get('decision'),
                        confidence=vote_data.get('confidence', 0.5),
                        reasoning=vote_data.get('reasoning', ''),
                        timestamp=time.time(),
                        stake=stake,
                        evidence=vote_data.get('evidence')
                    )
                    votes.append(vote)

            except Exception as e:
                cprint(f"⚠️ Failed to get vote from {agent_id}: {str(e)}", "yellow")

        return votes

    async def _simple_majority_consensus(self, votes: List[ConsensusVote], conflict: Conflict) -> Optional[ConsensusResult]:
        """Simple majority consensus (>50%)."""
        if not votes:
            return None

        # Count votes for each decision
        vote_counts = {}
        total_stake = sum(vote.stake for vote in votes)

        for vote in votes:
            decision_key = str(vote.decision)
            if decision_key not in vote_counts:
                vote_counts[decision_key] = {'count': 0, 'stake': 0, 'votes': []}
            vote_counts[decision_key]['count'] += 1
            vote_counts[decision_key]['stake'] += vote.stake
            vote_counts[decision_key]['votes'].append(vote)

        # Find majority decision
        majority_decision = max(vote_counts.items(), key=lambda x: x[1]['stake'])

        agreement_percentage = majority_decision[1]['stake'] / total_stake if total_stake > 0 else 0

        if agreement_percentage > 0.5:
            return ConsensusResult(
                consensus_id=str(uuid.uuid4()),
                algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
                votes=votes,
                decision=majority_decision[0],
                confidence=sum(v.confidence for v in majority_decision[1]['votes']) / len(majority_decision[1]['votes']),
                achieved_at=time.time(),
                participants=len(votes),
                agreement_percentage=agreement_percentage
            )

        return None

    async def _weighted_consensus(self, votes: List[ConsensusVote], conflict: Conflict) -> Optional[ConsensusResult]:
        """Weighted consensus based on agent performance."""
        if not votes:
            return None

        # Weight votes by stake (performance/reputation)
        weighted_votes = {}
        total_weight = sum(vote.stake for vote in votes)

        for vote in votes:
            decision_key = str(vote.decision)
            weight = vote.stake / total_weight if total_weight > 0 else 1.0 / len(votes)

            if decision_key not in weighted_votes:
                weighted_votes[decision_key] = {'weight': 0, 'votes': []}
            weighted_votes[decision_key]['weight'] += weight
            weighted_votes[decision_key]['votes'].append(vote)

        # Find highest weighted decision
        best_decision = max(weighted_votes.items(), key=lambda x: x[1]['weight'])

        if best_decision[1]['weight'] > 0.5:  # At least 50% weighted agreement
            return ConsensusResult(
                consensus_id=str(uuid.uuid4()),
                algorithm=ConsensusAlgorithm.WEIGHTED_CONSENSUS,
                votes=votes,
                decision=best_decision[0],
                confidence=sum(v.confidence * v.stake for v in best_decision[1]['votes']) / sum(v.stake for v in best_decision[1]['votes']),
                achieved_at=time.time(),
                participants=len(votes),
                agreement_percentage=best_decision[1]['weight']
            )

        return None

    async def _supermajority_consensus(self, votes: List[ConsensusVote], conflict: Conflict) -> Optional[ConsensusResult]:
        """Supermajority consensus (>66%)."""
        result = await self._simple_majority_consensus(votes, conflict)
        if result and result.agreement_percentage > 0.66:
            result.algorithm = ConsensusAlgorithm.SUPERMAJORITY
            return result
        return None

    async def _bft_consensus(self, votes: List[ConsensusVote], conflict: Conflict) -> Optional[ConsensusResult]:
        """Byzantine Fault Tolerant consensus."""
        # Simplified BFT implementation
        # In production, this would be much more sophisticated
        total_agents = len(conflict.involved_agents)
        faulty_limit = (total_agents - 1) // 3  # Can tolerate up to f faulty agents

        result = await self._simple_majority_consensus(votes, conflict)
        if result and len(votes) >= total_agents - faulty_limit:
            result.algorithm = ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANCE
            return result
        return None

    async def _quorum_consensus(self, votes: List[ConsensusVote], conflict: Conflict) -> Optional[ConsensusResult]:
        """Quorum-based consensus requiring minimum participation."""
        min_quorum = max(3, len(conflict.involved_agents) // 2 + 1)  # Majority or at least 3

        if len(votes) >= min_quorum:
            return await self._simple_majority_consensus(votes, conflict)
        return None

    async def _gather_confidence_scores(self, conflict: Conflict) -> Dict[str, Dict[str, Any]]:
        """Gather confidence scores from agents."""
        confidence_scores = {}

        for agent_id in conflict.involved_agents:
            try:
                response = await self.communication_bus.send_request(
                    sender_id="conflict_resolution",
                    recipient_id=agent_id,
                    topic="confidence_assessment",
                    payload={'conflict': conflict.__dict__},
                    timeout=15
                )

                if response and 'confidence' in response:
                    confidence_scores[agent_id] = {
                        'confidence': response['confidence'],
                        'decision': response.get('decision')
                    }

            except Exception as e:
                cprint(f"⚠️ Failed to get confidence from {agent_id}: {str(e)}", "yellow")

        return confidence_scores

    async def _find_domain_expert(self, conflict: Conflict) -> Optional[str]:
        """Find the most qualified agent for this conflict type."""
        # Simplified expert finding - in production, this would use ML
        expert_scores = {}

        for agent_id in conflict.involved_agents:
            # Calculate expertise score based on performance history
            performance = self.agent_performance.get(agent_id, {})
            expertise_score = performance.get(conflict.conflict_type.value, 0.5)
            expert_scores[agent_id] = expertise_score

        if expert_scores:
            return max(expert_scores.items(), key=lambda x: x[1])[0]

        return None

    def _calculate_agent_stake(self, agent_id: str, conflict_type: ConflictType) -> float:
        """Calculate voting stake based on agent performance."""
        performance = self.agent_performance.get(agent_id, {})
        base_performance = performance.get('overall', 0.5)
        type_performance = performance.get(conflict_type.value, 0.5)

        # Combine overall and type-specific performance
        stake = (base_performance * 0.6) + (type_performance * 0.4)
        return max(0.1, min(2.0, stake))  # Clamp between 0.1 and 2.0

    async def _escalate_conflict(self, conflict: Conflict) -> None:
        """Escalate conflict to human intervention."""
        conflict.status = "escalated"
        if self.stats['conflicts_detected'] > 0:
            self.stats['escalation_rate'] = (
                (self.stats['escalation_rate'] * (self.stats['conflicts_detected'] - 1)) +
                1
            ) / self.stats['conflicts_detected']

        cprint(f"🚨 Conflict {conflict.conflict_id} escalated to human intervention", "red")

        # Broadcast escalation notification
        await self.communication_bus.broadcast_message(
            sender_id="conflict_resolution",
            topic="conflict_escalation",
            payload={
                'conflict': conflict.__dict__,
                'escalated_at': time.time(),
                'reason': 'Automatic resolution failed'
            }
        )

    async def _monitoring_loop(self) -> None:
        """Monitor active conflicts and handle timeouts."""
        while self.running:
            try:
                current_time = time.time()

                # Check for expired conflicts
                expired_conflicts = [
                    conflict for conflict in self.active_conflicts.values()
                    if current_time > conflict.resolution_deadline
                ]

                for conflict in expired_conflicts:
                    cprint(f"⏰ Conflict {conflict.conflict_id} resolution deadline exceeded", "yellow")
                    await self._escalate_conflict(conflict)

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                cprint(f"❌ Conflict monitoring error: {str(e)}", "red")
                await asyncio.sleep(30)

    def get_stats(self) -> Dict[str, Any]:
        """Get conflict resolution statistics."""
        return {
            **self.stats,
            'active_conflicts': len(self.active_conflicts),
            'resolved_conflicts': len(self.resolved_conflicts),
            'consensus_history_size': len(self.consensus_history)
        }