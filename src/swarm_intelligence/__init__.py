"""
🧠 NeuroFlux Swarm Intelligence Package
Collective decision-making, neural swarm networks, and emergent behavior patterns.

Built with love by Nyros Veil 🚀

Features:
- Collective Decision Engine with consensus algorithms
- Neural Swarm Networks for inter-agent learning
- Emergent Behavior Patterns and self-organization
- Inter-Agent Learning with federated learning
- Swarm Consensus Algorithms
- Adaptive Swarm Coordination
"""

from .collective_decision_engine import CollectiveDecisionEngine, ConsensusAlgorithm, DecisionFusion
from .neural_swarm_network import NeuralSwarmNetwork, SynapticConnection, SwarmNeuron
from .emergent_behavior import EmergentBehaviorEngine, BehaviorPattern, SwarmDynamics
from .inter_agent_learning import InterAgentLearning, FederatedLearner, KnowledgeTransfer
# from .swarm_consensus import SwarmConsensus, WeightedVoting, BayesianConsensus
# from .adaptive_coordination import AdaptiveSwarmCoordinator, DynamicGrouping, TaskAllocation

__all__ = [
    'CollectiveDecisionEngine',
    'ConsensusAlgorithm',
    'DecisionFusion',
    'NeuralSwarmNetwork',
    'SynapticConnection',
    'SwarmNeuron',
    'EmergentBehaviorEngine',
    'BehaviorPattern',
    'SwarmDynamics',
    'InterAgentLearning',
    'FederatedLearner',
    'KnowledgeTransfer',
    # 'SwarmConsensus',
    # 'WeightedVoting',
    # 'BayesianConsensus',
    # 'AdaptiveSwarmCoordinator',
    # 'DynamicGrouping',
    # 'TaskAllocation'
]