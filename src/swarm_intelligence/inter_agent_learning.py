"""
🧠 Inter-Agent Learning
Federated learning and knowledge transfer between agents.

Built with love by Nyros Veil 🚀

Features:
- Federated learning across agents
- Knowledge transfer mechanisms
- Collaborative model training
- Experience sharing
- Learning from peer agents
"""

import asyncio
import time
import math
import random
import copy
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class KnowledgeTransfer:
    """Represents a knowledge transfer between agents."""

    source_agent_id: str
    target_agent_id: str
    knowledge_type: str
    knowledge_data: Any
    transfer_timestamp: float = field(default_factory=time.time)
    transfer_success: bool = False
    performance_impact: float = 0.0
    retention_rate: float = 1.0

    def apply_decay(self, decay_rate: float = 0.01) -> None:
        """Apply time-based decay to knowledge retention."""
        age = time.time() - self.transfer_timestamp
        self.retention_rate = max(0.0, math.exp(-decay_rate * age))

    def get_transfer_info(self) -> Dict[str, Any]:
        """Get transfer metadata."""
        return {
            'source': self.source_agent_id,
            'target': self.target_agent_id,
            'type': self.knowledge_type,
            'timestamp': self.transfer_timestamp,
            'success': self.transfer_success,
            'impact': self.performance_impact,
            'retention': self.retention_rate
        }


@dataclass
class FederatedLearner:
    """Manages federated learning across multiple agents."""

    learner_id: str
    participating_agents: Set[str] = field(default_factory=set)
    global_model: Dict[str, Any] = field(default_factory=dict)
    local_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aggregation_weights: Dict[str, float] = field(default_factory=dict)
    learning_round: int = 0
    convergence_threshold: float = 0.01
    max_rounds: int = 100

    def add_participant(self, agent_id: str, weight: float = 1.0) -> None:
        """Add an agent to the federated learning process."""
        self.participating_agents.add(agent_id)
        self.aggregation_weights[agent_id] = weight
        cprint(f"➕ Added agent {agent_id} to federated learner {self.learner_id}", "green")

    def remove_participant(self, agent_id: str) -> None:
        """Remove an agent from the federated learning process."""
        self.participating_agents.discard(agent_id)
        self.aggregation_weights.pop(agent_id, None)
        self.local_models.pop(agent_id, None)
        cprint(f"➖ Removed agent {agent_id} from federated learner {self.learner_id}", "yellow")

    def submit_local_model(self, agent_id: str, local_model: Dict[str, Any]) -> bool:
        """Submit a local model from an agent."""
        if agent_id not in self.participating_agents:
            cprint(f"❌ Agent {agent_id} not participating in federated learning", "red")
            return False

        self.local_models[agent_id] = copy.deepcopy(local_model)
        cprint(f"📤 Agent {agent_id} submitted local model", "blue")
        return True

    def aggregate_models(self) -> Dict[str, Any]:
        """Aggregate local models into a global model using weighted averaging."""
        if not self.local_models:
            cprint("⚠️ No local models to aggregate", "yellow")
            return self.global_model

        # Initialize aggregated model structure
        aggregated_model = {}
        total_weight = sum(self.aggregation_weights.get(agent_id, 1.0)
                          for agent_id in self.local_models.keys())

        if total_weight == 0:
            return self.global_model

        # Aggregate each parameter
        all_keys = set()
        for local_model in self.local_models.values():
            all_keys.update(local_model.keys())

        for key in all_keys:
            weighted_sum = 0.0
            total_key_weight = 0.0

            for agent_id, local_model in self.local_models.items():
                if key in local_model:
                    weight = self.aggregation_weights.get(agent_id, 1.0)
                    value = local_model[key]

                    # Handle different value types
                    if isinstance(value, (int, float)):
                        weighted_sum += value * weight
                        total_key_weight += weight
                    elif isinstance(value, dict):
                        # For nested dictionaries, aggregate recursively
                        if key not in aggregated_model:
                            aggregated_model[key] = {}
                        # This is simplified - real implementation would need proper nested aggregation
                        pass
                    elif isinstance(value, list):
                        # For lists, we might need different aggregation strategies
                        pass
                    else:
                        # For other types, use majority voting or keep first
                        if key not in aggregated_model:
                            aggregated_model[key] = value

            if total_key_weight > 0:
                aggregated_model[key] = weighted_sum / total_key_weight

        self.global_model = aggregated_model
        self.learning_round += 1

        cprint(f"🔄 Aggregated {len(self.local_models)} local models into global model (round {self.learning_round})", "cyan")
        return self.global_model

    def check_convergence(self) -> bool:
        """Check if the federated learning has converged."""
        if self.learning_round < 2:
            return False

        # Simple convergence check based on model stability
        # In a real implementation, this would compare model differences
        return self.learning_round >= self.max_rounds

    def get_global_model(self) -> Dict[str, Any]:
        """Get the current global model."""
        return copy.deepcopy(self.global_model)

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the federated learning process."""
        return {
            'learner_id': self.learner_id,
            'participants': len(self.participating_agents),
            'round': self.learning_round,
            'local_models': len(self.local_models),
            'converged': self.check_convergence(),
            'global_model_size': len(self.global_model)
        }


class InterAgentLearning:
    """Manages inter-agent learning and knowledge transfer."""

    def __init__(self, learning_network_id: str = "default_learning_network"):
        self.network_id = learning_network_id
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.knowledge_transfers: List[KnowledgeTransfer] = []
        self.federated_learners: Dict[str, FederatedLearner] = {}
        self.experience_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.learning_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.collaboration_graph: Dict[str, Set[str]] = defaultdict(set)

    def register_agent(self, agent_id: str, initial_knowledge: Optional[Dict[str, Any]] = None) -> None:
        """Register an agent in the learning network."""
        if initial_knowledge is None:
            initial_knowledge = {}

        self.agents[agent_id] = {
            'knowledge_base': copy.deepcopy(initial_knowledge),
            'performance_history': [],
            'learning_stats': {
                'transfers_received': 0,
                'transfers_sent': 0,
                'federated_rounds': 0,
                'improvement_rate': 0.0
            },
            'last_active': time.time()
        }

        cprint(f"📚 Registered agent {agent_id} in learning network", "green")

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the learning network."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.experience_pool.pop(agent_id, None)
            self.learning_metrics.pop(agent_id, None)

            # Remove from collaboration graph
            for collaborators in self.collaboration_graph.values():
                collaborators.discard(agent_id)
            self.collaboration_graph.pop(agent_id, None)

            # Remove from federated learners
            for learner in self.federated_learners.values():
                learner.remove_participant(agent_id)

            cprint(f"📚 Unregistered agent {agent_id} from learning network", "yellow")

    def transfer_knowledge(self, source_id: str, target_id: str,
                          knowledge_type: str, knowledge_data: Any) -> bool:
        """Transfer knowledge from one agent to another."""
        if source_id not in self.agents or target_id not in self.agents:
            cprint(f"❌ Cannot transfer knowledge: agents not registered", "red")
            return False

        # Create transfer record
        transfer = KnowledgeTransfer(
            source_agent_id=source_id,
            target_agent_id=target_id,
            knowledge_type=knowledge_type,
            knowledge_data=copy.deepcopy(knowledge_data)
        )

        # Apply knowledge to target agent
        success = self._apply_knowledge_to_agent(target_id, knowledge_type, knowledge_data)

        transfer.transfer_success = success
        self.knowledge_transfers.append(transfer)

        # Update collaboration graph
        self.collaboration_graph[source_id].add(target_id)

        # Update agent stats
        self.agents[source_id]['learning_stats']['transfers_sent'] += 1
        if success:
            self.agents[target_id]['learning_stats']['transfers_received'] += 1

        status = "✅" if success else "❌"
        cprint(f"{status} Knowledge transfer: {source_id} -> {target_id} ({knowledge_type})", "blue" if success else "red")
        return success

    def _apply_knowledge_to_agent(self, agent_id: str, knowledge_type: str, knowledge_data: Any) -> bool:
        """Apply transferred knowledge to an agent's knowledge base."""
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]
        knowledge_base = agent['knowledge_base']

        try:
            if knowledge_type == 'model_weights':
                # Update model weights
                if 'model_weights' not in knowledge_base:
                    knowledge_base['model_weights'] = {}
                knowledge_base['model_weights'].update(knowledge_data)

            elif knowledge_type == 'experience':
                # Add experience data
                if 'experiences' not in knowledge_base:
                    knowledge_base['experiences'] = []
                knowledge_base['experiences'].extend(knowledge_data)

            elif knowledge_type == 'strategies':
                # Update strategy knowledge
                if 'strategies' not in knowledge_base:
                    knowledge_base['strategies'] = {}
                knowledge_base['strategies'].update(knowledge_data)

            elif knowledge_type == 'patterns':
                # Update pattern recognition
                if 'patterns' not in knowledge_base:
                    knowledge_base['patterns'] = {}
                knowledge_base['patterns'].update(knowledge_data)

            else:
                # Generic knowledge storage
                knowledge_base[knowledge_type] = knowledge_data

            return True

        except Exception as e:
            cprint(f"❌ Error applying knowledge to agent {agent_id}: {e}", "red")
            return False

    def share_experience(self, agent_id: str, experience_data: Dict[str, Any]) -> None:
        """Share an experience with the learning network."""
        if agent_id not in self.agents:
            return

        # Add to agent's experience pool
        self.experience_pool[agent_id].append(experience_data)

        # Update agent's performance history
        if 'performance' in experience_data:
            self.agents[agent_id]['performance_history'].append(experience_data['performance'])

        # Keep only recent experiences (last 100)
        if len(self.experience_pool[agent_id]) > 100:
            self.experience_pool[agent_id].pop(0)

        cprint(f"📖 Agent {agent_id} shared experience", "cyan")

    def learn_from_experiences(self, agent_id: str, num_experiences: int = 10) -> Dict[str, Any]:
        """Learn from experiences shared by other agents."""
        if agent_id not in self.agents:
            return {}

        # Collect experiences from collaborative agents
        collaborative_agents = self.collaboration_graph.get(agent_id, set())
        all_experiences = []

        for collab_id in collaborative_agents:
            if collab_id in self.experience_pool:
                all_experiences.extend(self.experience_pool[collab_id])

        # Also include own experiences
        if agent_id in self.experience_pool:
            all_experiences.extend(self.experience_pool[agent_id])

        if not all_experiences:
            return {}

        # Select most relevant experiences (simplified selection)
        selected_experiences = all_experiences[-num_experiences:] if len(all_experiences) > num_experiences else all_experiences

        # Extract learning insights (simplified)
        insights = {
            'successful_actions': [],
            'failed_actions': [],
            'performance_trends': [],
            'learned_patterns': {}
        }

        for exp in selected_experiences:
            if 'outcome' in exp:
                if exp['outcome'] == 'success':
                    insights['successful_actions'].append(exp.get('action', 'unknown'))
                elif exp['outcome'] == 'failure':
                    insights['failed_actions'].append(exp.get('action', 'unknown'))

        # Update agent's knowledge base
        self.agents[agent_id]['knowledge_base']['learned_insights'] = insights

        cprint(f"🎓 Agent {agent_id} learned from {len(selected_experiences)} experiences", "magenta")
        return insights

    def create_federated_learner(self, learner_id: str, participant_ids: List[str]) -> FederatedLearner:
        """Create a new federated learning session."""
        learner = FederatedLearner(learner_id)
        self.federated_learners[learner_id] = learner

        for agent_id in participant_ids:
            if agent_id in self.agents:
                learner.add_participant(agent_id)

        cprint(f"🤝 Created federated learner {learner_id} with {len(participant_ids)} participants", "green")
        return learner

    def run_federated_round(self, learner_id: str) -> Optional[Dict[str, Any]]:
        """Run one round of federated learning."""
        if learner_id not in self.federated_learners:
            cprint(f"❌ Federated learner {learner_id} not found", "red")
            return None

        learner = self.federated_learners[learner_id]

        # Collect local models from participants
        for agent_id in learner.participating_agents:
            if agent_id in self.agents:
                local_model = self.agents[agent_id]['knowledge_base'].get('model_weights', {})
                learner.submit_local_model(agent_id, local_model)

        # Aggregate models
        global_model = learner.aggregate_models()

        # Distribute global model back to participants
        for agent_id in learner.participating_agents:
            if agent_id in self.agents:
                self.transfer_knowledge(learner_id, agent_id, 'global_model', global_model)

        # Update learning stats
        for agent_id in learner.participating_agents:
            if agent_id in self.agents:
                self.agents[agent_id]['learning_stats']['federated_rounds'] += 1

        return global_model

    def evaluate_learning_impact(self, agent_id: str) -> Dict[str, float]:
        """Evaluate the impact of learning on an agent's performance."""
        if agent_id not in self.agents:
            return {}

        agent = self.agents[agent_id]
        performance_history = agent['performance_history']
        learning_stats = agent['learning_stats']

        if len(performance_history) < 2:
            return {'improvement_rate': 0.0}

        # Calculate performance improvement
        recent_performance = performance_history[-10:]  # Last 10 performances
        if len(recent_performance) >= 2:
            improvement = (recent_performance[-1] - recent_performance[0]) / max(abs(recent_performance[0]), 0.001)
            learning_stats['improvement_rate'] = improvement
        else:
            improvement = 0.0

        return {
            'improvement_rate': improvement,
            'transfers_received': learning_stats['transfers_received'],
            'transfers_sent': learning_stats['transfers_sent'],
            'federated_rounds': learning_stats['federated_rounds'],
            'total_experiences': len(self.experience_pool.get(agent_id, []))
        }

    def get_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning network."""
        total_transfers = len(self.knowledge_transfers)
        successful_transfers = sum(1 for t in self.knowledge_transfers if t.transfer_success)

        return {
            'network_id': self.network_id,
            'total_agents': len(self.agents),
            'total_transfers': total_transfers,
            'successful_transfers': successful_transfers,
            'transfer_success_rate': successful_transfers / max(total_transfers, 1),
            'federated_learners': len(self.federated_learners),
            'total_experiences': sum(len(exp) for exp in self.experience_pool.values()),
            'collaboration_links': sum(len(links) for links in self.collaboration_graph.values())
        }

    async def run_learning_cycle(self) -> None:
        """Run one complete learning cycle including knowledge decay and network updates."""
        # Apply decay to old knowledge transfers
        for transfer in self.knowledge_transfers:
            transfer.apply_decay()

        # Clean up old transfers (keep last 1000)
        if len(self.knowledge_transfers) > 1000:
            self.knowledge_transfers = self.knowledge_transfers[-1000:]

        # Update agent activity timestamps
        current_time = time.time()
        inactive_agents = []
        for agent_id, agent_data in self.agents.items():
            if current_time - agent_data['last_active'] > 3600:  # 1 hour
                inactive_agents.append(agent_id)
            else:
                agent_data['last_active'] = current_time

        # Remove inactive agents (optional)
        # for agent_id in inactive_agents:
        #     self.unregister_agent(agent_id)

        # Small delay for cycle timing
        await asyncio.sleep(0.1)