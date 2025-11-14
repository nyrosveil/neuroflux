"""
🧠 Neural Swarm Networks
Inter-agent communication and learning through neural network-inspired connections.

Built with love by Nyros Veil 🚀

Features:
- Synaptic connections between agents
- Neural propagation of information
- Adaptive connection strengths
- Swarm neuron activation patterns
- Real-time network topology updates
"""

import asyncio
import time
import math
import random
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class SynapticConnection:
    """Represents a connection between two agents in the swarm network."""

    source_agent_id: str
    target_agent_id: str
    strength: float = 1.0  # Connection strength (0.0 to 1.0)
    last_activation: float = 0.0
    activation_count: int = 0
    learning_rate: float = 0.1
    created_at: float = field(default_factory=time.time)

    def activate(self, signal_strength: float) -> float:
        """Activate the connection with a signal."""
        self.last_activation = signal_strength * self.strength
        self.activation_count += 1

        # Adaptive learning: strengthen connections that are frequently used
        self.strength = min(1.0, self.strength + self.learning_rate * signal_strength)

        return self.last_activation

    def decay(self, decay_rate: float = 0.01) -> None:
        """Gradually decay connection strength over time."""
        time_since_creation = time.time() - self.created_at
        decay_factor = math.exp(-decay_rate * time_since_creation)
        self.strength *= decay_factor

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection metadata."""
        return {
            'source': self.source_agent_id,
            'target': self.target_agent_id,
            'strength': self.strength,
            'last_activation': self.last_activation,
            'activation_count': self.activation_count,
            'age': time.time() - self.created_at
        }


@dataclass
class SwarmNeuron:
    """Represents an agent as a neuron in the swarm network."""

    agent_id: str
    activation_threshold: float = 0.5
    current_activation: float = 0.0
    bias: float = 0.0
    refractory_period: float = 1.0  # seconds
    last_fired: float = 0.0
    input_connections: Dict[str, SynapticConnection] = field(default_factory=dict)
    output_connections: Dict[str, SynapticConnection] = field(default_factory=dict)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.5

    def add_input_connection(self, connection: SynapticConnection) -> None:
        """Add an incoming connection."""
        self.input_connections[connection.source_agent_id] = connection

    def add_output_connection(self, connection: SynapticConnection) -> None:
        """Add an outgoing connection."""
        self.output_connections[connection.target_agent_id] = connection

    def receive_signal(self, source_agent_id: str, signal_strength: float) -> float:
        """Receive a signal from another agent."""
        if source_agent_id in self.input_connections:
            connection = self.input_connections[source_agent_id]
            activation = connection.activate(signal_strength)
            self.current_activation += activation
            return activation
        return 0.0

    def should_fire(self) -> bool:
        """Check if the neuron should fire based on activation threshold."""
        time_since_last_fire = time.time() - self.last_fired
        return (self.current_activation >= self.activation_threshold and
                time_since_last_fire >= self.refractory_period)

    def fire(self) -> Dict[str, Any]:
        """Fire the neuron and propagate signal to connected agents."""
        if not self.should_fire():
            return {}

        self.last_fired = time.time()
        output_signal = self.current_activation + self.bias

        # Reset activation after firing
        self.current_activation = 0.0

        # Propagate to output connections
        propagated_signals = {}
        for target_id, connection in self.output_connections.items():
            propagated_signal = connection.activate(output_signal)
            propagated_signals[target_id] = propagated_signal

        return {
            'neuron_id': self.agent_id,
            'output_signal': output_signal,
            'propagated_signals': propagated_signals,
            'timestamp': self.last_fired
        }

    def update_performance(self, new_score: float) -> None:
        """Update the neuron's performance score."""
        # Exponential moving average for performance
        alpha = 0.1
        self.performance_score = alpha * new_score + (1 - alpha) * self.performance_score

    def get_neuron_state(self) -> Dict[str, Any]:
        """Get the current state of the neuron."""
        return {
            'agent_id': self.agent_id,
            'activation': self.current_activation,
            'threshold': self.activation_threshold,
            'performance': self.performance_score,
            'input_connections': len(self.input_connections),
            'output_connections': len(self.output_connections),
            'last_fired': self.last_fired,
            'knowledge_items': len(self.knowledge_base)
        }


class NeuralSwarmNetwork:
    """Neural network-inspired swarm communication system."""

    def __init__(self, network_id: str = "default_swarm"):
        self.network_id = network_id
        self.neurons: Dict[str, SwarmNeuron] = {}
        self.connections: Dict[Tuple[str, str], SynapticConnection] = {}
        self.network_topology: Dict[str, Set[str]] = defaultdict(set)
        self.global_knowledge_base: Dict[str, Any] = {}
        self.network_stats = {
            'total_signals': 0,
            'active_connections': 0,
            'average_activation': 0.0,
            'network_density': 0.0
        }
        self.learning_enabled = True
        self.adaptive_topology = True

    def add_agent(self, agent_id: str, activation_threshold: float = 0.5) -> SwarmNeuron:
        """Add an agent as a neuron to the network."""
        if agent_id not in self.neurons:
            neuron = SwarmNeuron(agent_id=agent_id, activation_threshold=activation_threshold)
            self.neurons[agent_id] = neuron
            self.network_topology[agent_id] = set()
            cprint(f"🧠 Added neuron {agent_id} to swarm network", "green")
        return self.neurons[agent_id]

    def create_connection(self, source_id: str, target_id: str,
                         initial_strength: float = 0.5) -> Optional[SynapticConnection]:
        """Create a synaptic connection between two agents."""
        if source_id not in self.neurons or target_id not in self.neurons:
            cprint(f"❌ Cannot create connection: agents {source_id} or {target_id} not in network", "red")
            return None

        connection_key = (source_id, target_id)
        if connection_key in self.connections:
            return self.connections[connection_key]

        connection = SynapticConnection(
            source_agent_id=source_id,
            target_agent_id=target_id,
            strength=initial_strength
        )

        self.connections[connection_key] = connection
        self.neurons[source_id].add_output_connection(connection)
        self.neurons[target_id].add_input_connection(connection)
        self.network_topology[source_id].add(target_id)

        cprint(f"🔗 Created connection {source_id} -> {target_id} (strength: {initial_strength})", "blue")
        return connection

    def remove_connection(self, source_id: str, target_id: str) -> bool:
        """Remove a connection between agents."""
        connection_key = (source_id, target_id)
        if connection_key not in self.connections:
            return False

        connection = self.connections[connection_key]
        self.neurons[source_id].output_connections.pop(target_id, None)
        self.neurons[target_id].input_connections.pop(source_id, None)
        self.network_topology[source_id].discard(target_id)
        del self.connections[connection_key]

        cprint(f"❌ Removed connection {source_id} -> {target_id}", "yellow")
        return True

    async def propagate_signal(self, source_agent_id: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate a signal through the network starting from a source agent."""
        if source_agent_id not in self.neurons:
            return {'error': f'Agent {source_agent_id} not in network'}

        source_neuron = self.neurons[source_agent_id]
        signal_strength = signal_data.get('strength', 1.0)
        signal_type = signal_data.get('type', 'information')
        signal_content = signal_data.get('content', {})

        # Initial activation of source neuron
        source_neuron.receive_signal(source_agent_id, signal_strength)

        propagation_results = []
        visited_neurons = set([source_agent_id])
        current_layer = [source_agent_id]

        # Breadth-first propagation through network layers
        for layer in range(3):  # Limit to 3 layers to prevent infinite loops
            next_layer = []

            for neuron_id in current_layer:
                neuron = self.neurons[neuron_id]

                # Check if neuron should fire
                if neuron.should_fire():
                    fire_result = neuron.fire()
                    propagation_results.append(fire_result)

                    # Propagate to connected neurons
                    for target_id, propagated_signal in fire_result['propagated_signals'].items():
                        if target_id not in visited_neurons:
                            target_neuron = self.neurons[target_id]
                            target_neuron.receive_signal(neuron_id, propagated_signal)
                            next_layer.append(target_id)
                            visited_neurons.add(target_id)

            current_layer = next_layer
            if not current_layer:
                break

            # Small delay between layers for realistic propagation
            await asyncio.sleep(0.01)

        self.network_stats['total_signals'] += 1
        self._update_network_stats()

        return {
            'source': source_agent_id,
            'signal_type': signal_type,
            'propagation_results': propagation_results,
            'total_activated': len(visited_neurons),
            'network_stats': self.network_stats.copy()
        }

    def share_knowledge(self, source_agent_id: str, knowledge_key: str,
                       knowledge_value: Any, target_agents: Optional[List[str]] = None) -> int:
        """Share knowledge from one agent to others through the network."""
        if source_agent_id not in self.neurons:
            return 0

        source_neuron = self.neurons[source_agent_id]
        source_neuron.knowledge_base[knowledge_key] = knowledge_value

        # If no specific targets, share with all connected agents
        if target_agents is None:
            target_agents = list(source_neuron.output_connections.keys())

        shared_count = 0
        for target_id in target_agents:
            if target_id in self.neurons:
                target_neuron = self.neurons[target_id]
                target_neuron.knowledge_base[knowledge_key] = knowledge_value

                # Strengthen connection due to knowledge sharing
                connection_key = (source_agent_id, target_id)
                if connection_key in self.connections:
                    self.connections[connection_key].strength = min(1.0,
                        self.connections[connection_key].strength + 0.1)

                shared_count += 1

        # Update global knowledge base
        self.global_knowledge_base[knowledge_key] = knowledge_value

        cprint(f"📚 Shared knowledge '{knowledge_key}' from {source_agent_id} to {shared_count} agents", "cyan")
        return shared_count

    def adapt_topology(self) -> None:
        """Adapt network topology based on agent performance and connection usage."""
        if not self.adaptive_topology:
            return

        # Remove weak connections
        weak_connections = []
        for (source_id, target_id), connection in self.connections.items():
            if connection.strength < 0.1:
                weak_connections.append((source_id, target_id))

        for source_id, target_id in weak_connections:
            self.remove_connection(source_id, target_id)

        # Create new connections between high-performing agents
        high_performers = [nid for nid, neuron in self.neurons.items()
                          if neuron.performance_score > 0.7]

        for i, agent1 in enumerate(high_performers):
            for agent2 in high_performers[i+1:]:
                if (agent1, agent2) not in self.connections and random.random() < 0.1:
                    self.create_connection(agent1, agent2, initial_strength=0.3)

    def _update_network_stats(self) -> None:
        """Update network statistics."""
        total_neurons = len(self.neurons)
        total_connections = len(self.connections)

        if total_neurons > 1:
            self.network_stats['network_density'] = (2 * total_connections) / (total_neurons * (total_neurons - 1))
        else:
            self.network_stats['network_density'] = 0.0

        self.network_stats['active_connections'] = total_connections
        self.network_stats['average_activation'] = sum(
            neuron.current_activation for neuron in self.neurons.values()
        ) / max(1, total_neurons)

    def get_network_state(self) -> Dict[str, Any]:
        """Get the current state of the entire network."""
        return {
            'network_id': self.network_id,
            'neurons': {nid: neuron.get_neuron_state() for nid, neuron in self.neurons.items()},
            'connections': {f"{src}->{tgt}": conn.get_connection_info()
                          for (src, tgt), conn in self.connections.items()},
            'topology': dict(self.network_topology),
            'stats': self.network_stats.copy(),
            'global_knowledge': list(self.global_knowledge_base.keys()),
            'learning_enabled': self.learning_enabled,
            'adaptive_topology': self.adaptive_topology
        }

    async def run_network_cycle(self) -> None:
        """Run one complete network cycle including adaptation."""
        # Decay old connections
        for connection in self.connections.values():
            connection.decay()

        # Adapt topology if enabled
        if self.adaptive_topology:
            self.adapt_topology()

        # Update network stats
        self._update_network_stats()

        # Small delay for cycle timing
        await asyncio.sleep(0.1)
