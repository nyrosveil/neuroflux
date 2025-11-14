"""
🧠 Collective Decision Engine
Advanced consensus algorithms and decision fusion for swarm intelligence.

Built with love by Nyros Veil 🚀

Features:
- Weighted voting consensus
- Bayesian consensus algorithms
- Decision fusion techniques
- Confidence scoring and risk assessment
- Real-time consensus monitoring
"""

import asyncio
import time
import math
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConsensusAlgorithm(Enum):
    """Available consensus algorithms."""
    WEIGHTED_VOTING = "weighted_voting"
    BAYESIAN_CONSENSUS = "bayesian_consensus"
    MAJORITY_VOTING = "majority_voting"
    QUORUM_CONSENSUS = "quorum_consensus"
    PROBABILISTIC_CONSENSUS = "probabilistic_consensus"


class DecisionType(Enum):
    """Types of decisions that can be made."""
    BINARY = "binary"  # Yes/No decisions
    MULTI_CHOICE = "multi_choice"  # Multiple options
    CONTINUOUS = "continuous"  # Numerical values
    RANKING = "ranking"  # Ordered preferences


@dataclass
class AgentOpinion:
    """Represents an agent's opinion on a decision."""
    agent_id: str
    decision_id: str
    value: Any
    confidence: float = 1.0  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """Result of a consensus decision."""
    decision_id: str
    consensus_value: Any
    confidence_score: float  # 0.0 to 1.0
    agreement_level: float  # 0.0 to 1.0 (percentage of agents agreeing)
    participant_count: int
    algorithm_used: ConsensusAlgorithm
    timestamp: float = field(default_factory=time.time)
    supporting_opinions: List[AgentOpinion] = field(default_factory=list)
    dissenting_opinions: List[AgentOpinion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Context information for a decision."""
    decision_id: str
    decision_type: DecisionType
    description: str
    options: List[Any] = field(default_factory=list)  # For multi-choice decisions
    deadline: Optional[float] = None
    min_participants: int = 1
    required_confidence: float = 0.7
    algorithm: ConsensusAlgorithm = ConsensusAlgorithm.WEIGHTED_VOTING
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionFusion:
    """
    Advanced decision fusion techniques for combining multiple agent opinions.
    """

    @staticmethod
    def weighted_average(opinions: List[AgentOpinion], weights: Optional[Dict[str, float]] = None) -> Tuple[Any, float]:
        """
        Calculate weighted average of opinions.

        Args:
            opinions: List of agent opinions
            weights: Optional weights for each agent (defaults to confidence-based)

        Returns:
            Tuple of (weighted_average, confidence_score)
        """
        if not opinions:
            return None, 0.0

        # Use confidence as default weights if not provided
        if weights is None:
            weights = {op.agent_id: op.confidence for op in opinions}

        total_weight = 0.0
        weighted_sum = 0.0

        for opinion in opinions:
            weight = weights.get(opinion.agent_id, opinion.confidence)
            if isinstance(opinion.value, (int, float)):
                weighted_sum += opinion.value * weight
                total_weight += weight

        if total_weight == 0:
            return None, 0.0

        average = weighted_sum / total_weight

        # Calculate confidence based on weight distribution
        confidence = min(1.0, total_weight / len(opinions))

        return average, confidence

    @staticmethod
    def majority_vote(opinions: List[AgentOpinion]) -> Tuple[Any, float]:
        """
        Determine majority vote from opinions.

        Args:
            opinions: List of agent opinions

        Returns:
            Tuple of (majority_value, confidence_score)
        """
        if not opinions:
            return None, 0.0

        # Count votes for each value
        vote_counts = {}
        total_confidence = 0.0

        for opinion in opinions:
            value = opinion.value
            if value not in vote_counts:
                vote_counts[value] = 0.0
            vote_counts[value] += opinion.confidence
            total_confidence += opinion.confidence

        if not vote_counts:
            return None, 0.0

        # Find majority
        majority_value = max(vote_counts.keys(), key=lambda x: vote_counts[x])
        majority_votes = vote_counts[majority_value]

        # Calculate confidence as percentage of total confidence
        confidence = majority_votes / total_confidence if total_confidence > 0 else 0.0

        return majority_value, confidence

    @staticmethod
    def probabilistic_fusion(opinions: List[AgentOpinion]) -> Tuple[Any, float]:
        """
        Use probabilistic methods to fuse opinions.

        Args:
            opinions: List of agent opinions

        Returns:
            Tuple of (fused_value, confidence_score)
        """
        if not opinions:
            return None, 0.0

        # For numerical values, use statistical methods
        numerical_values = [op.value for op in opinions if isinstance(op.value, (int, float))]

        if len(numerical_values) >= 2:
            # Use mean with confidence interval
            mean_value = statistics.mean(numerical_values)
            stdev = statistics.stdev(numerical_values) if len(numerical_values) > 1 else 0

            # Confidence based on coefficient of variation
            cv = stdev / abs(mean_value) if mean_value != 0 else float('inf')
            confidence = max(0.0, min(1.0, 1.0 - cv))

            return mean_value, confidence

        # For categorical values, use majority vote
        return DecisionFusion.majority_vote(opinions)

    @staticmethod
    def bayesian_fusion(opinions: List[AgentOpinion], prior_belief: Optional[float] = None) -> Tuple[Any, float]:
        """
        Bayesian approach to decision fusion.

        Args:
            opinions: List of agent opinions
            prior_belief: Prior belief probability

        Returns:
            Tuple of (fused_value, confidence_score)
        """
        if not opinions:
            return None, 0.0

        # Simplified Bayesian update for binary decisions
        if all(isinstance(op.value, bool) for op in opinions):
            # Binary decision fusion
            positive_votes = sum(op.confidence for op in opinions if op.value)
            negative_votes = sum(op.confidence for op in opinions if not op.value)

            total_evidence = positive_votes + negative_votes

            if total_evidence == 0:
                return None, 0.0

            # Bayesian update
            prior = prior_belief if prior_belief is not None else 0.5
            likelihood_ratio = positive_votes / negative_votes if negative_votes > 0 else float('inf')

            posterior = (prior * likelihood_ratio) / (prior * likelihood_ratio + (1 - prior))

            return posterior >= 0.5, min(1.0, posterior)

        # For other types, fall back to weighted average
        return DecisionFusion.weighted_average(opinions)


class CollectiveDecisionEngine:
    """
    Engine for collective decision-making in swarm intelligence systems.

    Features:
    - Multiple consensus algorithms
    - Real-time decision tracking
    - Confidence scoring and risk assessment
    - Decision fusion techniques
    - Adaptive algorithm selection
    """

    def __init__(self):
        self.active_decisions: Dict[str, DecisionContext] = {}
        self.opinions: Dict[str, List[AgentOpinion]] = {}  # decision_id -> opinions
        self.consensus_results: Dict[str, ConsensusResult] = {}
        self.agent_weights: Dict[str, float] = {}  # agent_id -> weight
        self.decision_history: List[ConsensusResult] = []

        # Performance tracking
        self.stats = {
            'decisions_made': 0,
            'avg_confidence': 0.0,
            'avg_participants': 0.0,
            'consensus_rate': 0.0,  # percentage of decisions reaching consensus
            'avg_decision_time': 0.0
        }

    def register_decision(self, context: DecisionContext) -> None:
        """
        Register a new decision for collective consideration.

        Args:
            context: Decision context information
        """
        self.active_decisions[context.decision_id] = context
        self.opinions[context.decision_id] = []

        cprint(f"📋 Registered decision: {context.decision_id} - {context.description}", "blue")

    def submit_opinion(self, opinion: AgentOpinion) -> None:
        """
        Submit an agent's opinion on a decision.

        Args:
            opinion: Agent's opinion
        """
        if opinion.decision_id not in self.active_decisions:
            cprint(f"⚠️ Decision {opinion.decision_id} not found", "yellow")
            return

        if opinion.decision_id not in self.opinions:
            self.opinions[opinion.decision_id] = []

        self.opinions[opinion.decision_id].append(opinion)

        cprint(f"💭 Agent {opinion.agent_id} submitted opinion on {opinion.decision_id}: {opinion.value}", "cyan")

        # Check if we can reach consensus
        self._check_consensus(opinion.decision_id)

    def _check_consensus(self, decision_id: str) -> None:
        """
        Check if consensus can be reached for a decision.

        Args:
            decision_id: ID of the decision to check
        """
        if decision_id not in self.active_decisions:
            return

        context = self.active_decisions[decision_id]
        opinions = self.opinions.get(decision_id, [])

        # Check minimum participants
        if len(opinions) < context.min_participants:
            return

        # Check deadline
        if context.deadline and time.time() > context.deadline:
            # Force decision even with insufficient participants
            pass
        elif len(opinions) < context.min_participants:
            return

        # Attempt to reach consensus
        result = self._calculate_consensus(decision_id, context, opinions)

        if result and result.confidence_score >= context.required_confidence:
            self._finalize_decision(result)
        elif context.deadline and time.time() > context.deadline:
            # Deadline reached, finalize with best available result
            if result:
                self._finalize_decision(result)

    def _calculate_consensus(self, decision_id: str, context: DecisionContext,
                           opinions: List[AgentOpinion]) -> Optional[ConsensusResult]:
        """
        Calculate consensus using the specified algorithm.

        Args:
            decision_id: Decision identifier
            context: Decision context
            opinions: List of opinions

        Returns:
            ConsensusResult if consensus reached, None otherwise
        """
        algorithm = context.algorithm

        try:
            if algorithm == ConsensusAlgorithm.WEIGHTED_VOTING:
                consensus_value, confidence = self._weighted_voting_consensus(opinions)
            elif algorithm == ConsensusAlgorithm.BAYESIAN_CONSENSUS:
                consensus_value, confidence = self._bayesian_consensus(opinions)
            elif algorithm == ConsensusAlgorithm.MAJORITY_VOTING:
                consensus_value, confidence = DecisionFusion.majority_vote(opinions)
            elif algorithm == ConsensusAlgorithm.QUORUM_CONSENSUS:
                consensus_value, confidence = self._quorum_consensus(opinions, context.min_participants)
            elif algorithm == ConsensusAlgorithm.PROBABILISTIC_CONSENSUS:
                consensus_value, confidence = DecisionFusion.probabilistic_fusion(opinions)
            else:
                cprint(f"⚠️ Unknown algorithm: {algorithm}", "yellow")
                return None

            if consensus_value is None:
                return None

            # Calculate agreement level
            agreement_level = self._calculate_agreement_level(opinions, consensus_value)

            # Separate supporting and dissenting opinions
            supporting, dissenting = self._categorize_opinions(opinions, consensus_value)

            result = ConsensusResult(
                decision_id=decision_id,
                consensus_value=consensus_value,
                confidence_score=confidence,
                agreement_level=agreement_level,
                participant_count=len(opinions),
                algorithm_used=algorithm,
                supporting_opinions=supporting,
                dissenting_opinions=dissenting
            )

            return result

        except Exception as e:
            cprint(f"❌ Consensus calculation error: {e}", "red")
            return None

    def _weighted_voting_consensus(self, opinions: List[AgentOpinion]) -> Tuple[Any, float]:
        """Calculate consensus using weighted voting."""
        return DecisionFusion.weighted_average(opinions, self.agent_weights)

    def _bayesian_consensus(self, opinions: List[AgentOpinion]) -> Tuple[Any, float]:
        """Calculate consensus using Bayesian methods."""
        return DecisionFusion.bayesian_fusion(opinions)

    def _quorum_consensus(self, opinions: List[AgentOpinion], quorum_size: int) -> Tuple[Any, float]:
        """
        Calculate consensus requiring minimum quorum.

        Args:
            opinions: List of opinions
            quorum_size: Minimum number of participants required

        Returns:
            Tuple of (consensus_value, confidence_score)
        """
        if len(opinions) < quorum_size:
            return None, 0.0

        # Use majority vote for quorum consensus
        return DecisionFusion.majority_vote(opinions)

    def _calculate_agreement_level(self, opinions: List[AgentOpinion], consensus_value: Any) -> float:
        """
        Calculate the level of agreement among opinions.

        Args:
            opinions: List of opinions
            consensus_value: The consensus value

        Returns:
            Agreement level (0.0 to 1.0)
        """
        if not opinions:
            return 0.0

        agreeing = 0
        total_confidence = 0.0

        for opinion in opinions:
            total_confidence += opinion.confidence
            if self._opinions_agree(opinion.value, consensus_value):
                agreeing += opinion.confidence

        return agreeing / total_confidence if total_confidence > 0 else 0.0

    def _opinions_agree(self, value1: Any, value2: Any, tolerance: float = 0.1) -> bool:
        """
        Check if two opinion values agree within tolerance.

        Args:
            value1: First value
            value2: Second value
            tolerance: Tolerance for numerical comparison

        Returns:
            True if values agree
        """
        if value1 == value2:
            return True

        # For numerical values, check within tolerance
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return abs(value1 - value2) <= tolerance

        # For boolean values
        if isinstance(value1, bool) and isinstance(value2, bool):
            return value1 == value2

        return False

    def _categorize_opinions(self, opinions: List[AgentOpinion], consensus_value: Any) -> Tuple[List[AgentOpinion], List[AgentOpinion]]:
        """
        Categorize opinions into supporting and dissenting.

        Args:
            opinions: List of all opinions
            consensus_value: The consensus value

        Returns:
            Tuple of (supporting_opinions, dissenting_opinions)
        """
        supporting = []
        dissenting = []

        for opinion in opinions:
            if self._opinions_agree(opinion.value, consensus_value):
                supporting.append(opinion)
            else:
                dissenting.append(opinion)

        return supporting, dissenting

    def _finalize_decision(self, result: ConsensusResult) -> None:
        """
        Finalize a decision and clean up resources.

        Args:
            result: The consensus result
        """
        decision_id = result.decision_id

        # Store result
        self.consensus_results[decision_id] = result
        self.decision_history.append(result)

        # Update statistics
        self.stats['decisions_made'] += 1
        self._update_performance_stats(result)

        # Clean up
        if decision_id in self.active_decisions:
            del self.active_decisions[decision_id]
        if decision_id in self.opinions:
            del self.opinions[decision_id]

        cprint(f"✅ Consensus reached for {decision_id}: {result.consensus_value} "
               f"(confidence: {result.confidence_score:.2f}, agreement: {result.agreement_level:.2f})", "green")

    def _update_performance_stats(self, result: ConsensusResult) -> None:
        """Update performance statistics."""
        # Running averages
        n = self.stats['decisions_made']

        self.stats['avg_confidence'] = ((self.stats['avg_confidence'] * (n - 1)) + result.confidence_score) / n
        self.stats['avg_participants'] = ((self.stats['avg_participants'] * (n - 1)) + result.participant_count) / n

        # Consensus rate (decisions with confidence >= 0.7)
        consensus_decisions = sum(1 for r in self.decision_history if r.confidence_score >= 0.7)
        self.stats['consensus_rate'] = consensus_decisions / len(self.decision_history)

    def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a decision.

        Args:
            decision_id: Decision identifier

        Returns:
            Status dictionary or None if decision not found
        """
        if decision_id in self.consensus_results:
            result = self.consensus_results[decision_id]
            return {
                'status': 'completed',
                'result': result.consensus_value,
                'confidence': result.confidence_score,
                'participants': result.participant_count,
                'timestamp': result.timestamp
            }
        elif decision_id in self.active_decisions:
            opinions = self.opinions.get(decision_id, [])
            return {
                'status': 'active',
                'participants': len(opinions),
                'required_participants': self.active_decisions[decision_id].min_participants,
                'deadline': self.active_decisions[decision_id].deadline
            }

        return None

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return {
            **self.stats,
            'active_decisions': len(self.active_decisions),
            'total_opinions': sum(len(opinions) for opinions in self.opinions.values()),
            'completed_decisions': len(self.consensus_results)
        }

    def set_agent_weight(self, agent_id: str, weight: float) -> None:
        """
        Set the weight/trust score for an agent.

        Args:
            agent_id: Agent identifier
            weight: Weight value (0.0 to 1.0)
        """
        self.agent_weights[agent_id] = max(0.0, min(1.0, weight))

    def get_agent_weights(self) -> Dict[str, float]:
        """Get all agent weights."""
        return self.agent_weights.copy()