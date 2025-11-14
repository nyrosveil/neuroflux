"""
🧠 Emergent Behavior Engine
Self-organization and emergent behavior patterns in swarm intelligence.

Built with love by Nyros Veil 🚀

Features:
- Self-organization algorithms
- Emergent behavior pattern detection
- Swarm dynamics modeling
- Collective behavior emergence
- Pattern recognition and classification
"""

import asyncio
import time
import math
import random
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class BehaviorPattern(Enum):
    """Types of emergent behavior patterns."""
    FLOCKING = "flocking"  # Agents moving together
    SWARMING = "swarming"  # Dense clustering
    SCHOOLING = "schooling"  # Coordinated movement
    HERDING = "herding"  # Following leader
    DISPERSING = "dispersing"  # Spreading out
    CONVERGING = "converging"  # Coming together
    OSCILLATING = "oscillating"  # Periodic behavior
    CHAOTIC = "chaotic"  # Unpredictable behavior
    SYNCHRONIZED = "synchronized"  # Coordinated timing
    EMERGENT_LEADERSHIP = "emergent_leadership"  # Spontaneous leaders


@dataclass
class SwarmDynamics:
    """Represents the dynamic state of a swarm."""

    swarm_id: str
    timestamp: float = field(default_factory=time.time)
    center_of_mass: Tuple[float, float] = (0.0, 0.0)
    average_velocity: Tuple[float, float] = (0.0, 0.0)
    cohesion: float = 0.0  # How closely grouped agents are
    alignment: float = 0.0  # How aligned movement directions are
    separation: float = 0.0  # How agents avoid crowding
    polarization: float = 0.0  # Degree of directional agreement
    density: float = 0.0  # Agent density
    entropy: float = 0.0  # Disorder measure
    pattern_strength: Dict[BehaviorPattern, float] = field(default_factory=dict)

    def update_dynamics(self, agent_positions: Dict[str, Tuple[float, float]],
                       agent_velocities: Dict[str, Tuple[float, float]]) -> None:
        """Update swarm dynamics from agent states."""
        if not agent_positions:
            return

        self.timestamp = time.time()

        # Calculate center of mass
        positions = list(agent_positions.values())
        self.center_of_mass = (
            sum(p[0] for p in positions) / len(positions),
            sum(p[1] for p in positions) / len(positions)
        )

        # Calculate average velocity
        if agent_velocities:
            velocities = list(agent_velocities.values())
            self.average_velocity = (
                sum(v[0] for v in velocities) / len(velocities),
                sum(v[1] for v in velocities) / len(velocities)
            )

        # Calculate cohesion (inverse of average distance from center)
        distances = [math.dist(pos, self.center_of_mass) for pos in positions]
        avg_distance = statistics.mean(distances) if distances else 0
        self.cohesion = 1.0 / (1.0 + avg_distance) if avg_distance > 0 else 1.0

        # Calculate alignment (average dot product of velocity vectors)
        if agent_velocities and len(agent_velocities) > 1:
            vel_list = list(agent_velocities.values())
            avg_vel = self.average_velocity
            avg_vel_magnitude = math.dist((0, 0), avg_vel)
            if avg_vel_magnitude > 0:
                normalized_avg = (avg_vel[0] / avg_vel_magnitude, avg_vel[1] / avg_vel_magnitude)
                alignments = []
                for vel in vel_list:
                    vel_magnitude = math.dist((0, 0), vel)
                    if vel_magnitude > 0:
                        normalized_vel = (vel[0] / vel_magnitude, vel[1] / vel_magnitude)
                        dot_product = (normalized_vel[0] * normalized_avg[0] +
                                     normalized_vel[1] * normalized_avg[1])
                        alignments.append(max(0, dot_product))  # Only positive alignment
                self.alignment = statistics.mean(alignments) if alignments else 0.0

        # Calculate separation (minimum distance between agents)
        min_distances = []
        pos_list = list(agent_positions.values())
        for i, pos1 in enumerate(pos_list):
            for j, pos2 in enumerate(pos_list[i+1:], i+1):
                distance = math.dist(pos1, pos2)
                min_distances.append(distance)
        self.separation = statistics.mean(min_distances) if min_distances else 0.0

        # Calculate polarization (strength of directional agreement)
        if agent_velocities:
            vel_magnitudes = [math.dist((0, 0), v) for v in agent_velocities.values()]
            avg_magnitude = statistics.mean(vel_magnitudes) if vel_magnitudes else 0
            if avg_magnitude > 0:
                self.polarization = math.dist((0, 0), self.average_velocity) / avg_magnitude
            else:
                self.polarization = 0.0

        # Calculate density (agents per unit area)
        if positions:
            # Simple bounding box area calculation
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            width = max(x_coords) - min(x_coords) if x_coords else 1
            height = max(y_coords) - min(y_coords) if y_coords else 1
            area = max(width * height, 1)  # Avoid division by zero
            self.density = len(positions) / area

        # Calculate entropy (diversity of positions)
        if positions:
            # Discretize positions into grid cells
            grid_size = 10
            grid = defaultdict(int)
            for pos in positions:
                grid_x = int(pos[0] / grid_size)
                grid_y = int(pos[1] / grid_size)
                grid[(grid_x, grid_y)] += 1

            total_cells = len(grid)
            if total_cells > 0:
                probabilities = [count / len(positions) for count in grid.values()]
                self.entropy = -sum(p * math.log(p) for p in probabilities if p > 0)

    def detect_patterns(self) -> Dict[BehaviorPattern, float]:
        """Detect emergent behavior patterns based on dynamics."""
        patterns = {}

        # Flocking: High cohesion, high alignment, moderate separation
        flocking_score = (self.cohesion * 0.4 + self.alignment * 0.4 +
                         min(1.0, self.separation / 10.0) * 0.2)
        patterns[BehaviorPattern.FLOCKING] = flocking_score

        # Swarming: High density, high cohesion, low separation
        swarming_score = (self.density * 0.4 + self.cohesion * 0.4 +
                         (1.0 - min(1.0, self.separation / 5.0)) * 0.2)
        patterns[BehaviorPattern.SWARMING] = swarming_score

        # Schooling: High alignment, moderate cohesion, good separation
        schooling_score = (self.alignment * 0.5 + self.cohesion * 0.3 +
                          min(1.0, self.separation / 15.0) * 0.2)
        patterns[BehaviorPattern.SCHOOLING] = schooling_score

        # Herding: High polarization, moderate cohesion
        herding_score = (self.polarization * 0.6 + self.cohesion * 0.4)
        patterns[BehaviorPattern.HERDING] = herding_score

        # Dispersing: Low cohesion, high separation
        dispersing_score = ((1.0 - self.cohesion) * 0.6 + min(1.0, self.separation / 20.0) * 0.4)
        patterns[BehaviorPattern.DISPERSING] = dispersing_score

        # Converging: Increasing cohesion over time (would need historical data)
        patterns[BehaviorPattern.CONVERGING] = self.cohesion

        # Oscillating: Would need time series analysis
        patterns[BehaviorPattern.OSCILLATING] = 0.0  # Placeholder

        # Chaotic: High entropy, low alignment
        chaotic_score = (self.entropy * 0.5 + (1.0 - self.alignment) * 0.5)
        patterns[BehaviorPattern.CHAOTIC] = chaotic_score

        # Synchronized: High alignment, low entropy
        synchronized_score = (self.alignment * 0.6 + (1.0 - self.entropy) * 0.4)
        patterns[BehaviorPattern.SYNCHRONIZED] = synchronized_score

        # Emergent leadership: High polarization with some agents leading
        emergent_leadership_score = self.polarization * 0.7 + random.random() * 0.3  # Simplified
        patterns[BehaviorPattern.EMERGENT_LEADERSHIP] = emergent_leadership_score

        self.pattern_strength = patterns
        return patterns

    def get_dynamics_summary(self) -> Dict[str, Any]:
        """Get a summary of current swarm dynamics."""
        return {
            'timestamp': self.timestamp,
            'center_of_mass': self.center_of_mass,
            'average_velocity': self.average_velocity,
            'cohesion': self.cohesion,
            'alignment': self.alignment,
            'separation': self.separation,
            'polarization': self.polarization,
            'density': self.density,
            'entropy': self.entropy,
            'dominant_patterns': sorted(self.pattern_strength.items(),
                                      key=lambda x: x[1], reverse=True)[:3]
        }


class EmergentBehaviorEngine:
    """Engine for detecting and managing emergent behaviors in swarms."""

    def __init__(self, swarm_id: str = "default_swarm"):
        self.swarm_id = swarm_id
        self.dynamics_history: List[SwarmDynamics] = []
        self.current_dynamics = SwarmDynamics(swarm_id)
        self.pattern_history: Dict[BehaviorPattern, List[float]] = defaultdict(list)
        self.behavior_transitions: List[Tuple[BehaviorPattern, BehaviorPattern, float]] = []
        self.self_organization_active = True
        self.pattern_detection_threshold = 0.6
        self.adaptation_rate = 0.1

    def update_swarm_state(self, agent_positions: Dict[str, Tuple[float, float]],
                          agent_velocities: Dict[str, Tuple[float, float]]) -> None:
        """Update the swarm state and detect emergent behaviors."""
        # Update current dynamics
        self.current_dynamics.update_dynamics(agent_positions, agent_velocities)

        # Detect patterns
        patterns = self.current_dynamics.detect_patterns()

        # Store in history
        self.dynamics_history.append(self.current_dynamics)

        # Update pattern history
        for pattern, strength in patterns.items():
            self.pattern_history[pattern].append(strength)

            # Keep only recent history (last 100 entries)
            if len(self.pattern_history[pattern]) > 100:
                self.pattern_history[pattern].pop(0)

        # Detect behavior transitions
        if len(self.dynamics_history) >= 2:
            prev_dynamics = self.dynamics_history[-2]
            current_patterns = self.current_dynamics.pattern_strength
            prev_patterns = prev_dynamics.pattern_strength

            # Find dominant patterns
            current_dominant = max(current_patterns.items(), key=lambda x: x[1])
            prev_dominant = max(prev_patterns.items(), key=lambda x: x[1])

            if (current_dominant[0] != prev_dominant[0] and
                current_dominant[1] > self.pattern_detection_threshold):
                transition = (prev_dominant[0], current_dominant[0], time.time())
                self.behavior_transitions.append(transition)

                cprint(f"🔄 Behavior transition detected: {prev_dominant[0].value} -> {current_dominant[0].value}",
                      "yellow")

        # Limit history size
        if len(self.dynamics_history) > 1000:
            self.dynamics_history.pop(0)
        if len(self.behavior_transitions) > 100:
            self.behavior_transitions.pop(0)

    def get_dominant_patterns(self, time_window: int = 10) -> List[Tuple[BehaviorPattern, float]]:
        """Get the dominant behavior patterns in recent history."""
        if not self.pattern_history:
            return []

        # Get recent pattern strengths
        recent_patterns = {}
        for pattern, history in self.pattern_history.items():
            recent = history[-time_window:] if len(history) >= time_window else history
            if recent:
                recent_patterns[pattern] = statistics.mean(recent)

        # Sort by strength
        return sorted(recent_patterns.items(), key=lambda x: x[1], reverse=True)

    def predict_behavior_transition(self) -> Optional[Tuple[BehaviorPattern, BehaviorPattern, float]]:
        """Predict likely behavior transitions based on historical data."""
        if len(self.behavior_transitions) < 5:
            return None

        # Analyze transition frequencies
        transitions = Counter((t[0], t[1]) for t in self.behavior_transitions)
        if not transitions:
            return None

        # Find most common transition from current dominant pattern
        dominant_patterns = self.get_dominant_patterns(5)
        if not dominant_patterns:
            return None

        current_pattern = dominant_patterns[0][0]

        # Find transitions from current pattern
        possible_transitions = [(to_pattern, count) for (from_pattern, to_pattern), count
                              in transitions.items() if from_pattern == current_pattern]

        if not possible_transitions:
            return None

        # Return most likely transition
        most_likely = max(possible_transitions, key=lambda x: x[1])
        confidence = most_likely[1] / sum(t[1] for t in possible_transitions)

        return (current_pattern, most_likely[0], confidence)

    def apply_self_organization(self, agent_positions: Dict[str, Tuple[float, float]],
                              agent_velocities: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply self-organization rules to adjust agent behaviors."""
        if not self.self_organization_active:
            return {}

        adjustments = {}
        dominant_patterns = self.get_dominant_patterns(5)

        if not dominant_patterns:
            return adjustments

        primary_pattern = dominant_patterns[0][0]

        # Apply pattern-specific self-organization rules
        if primary_pattern == BehaviorPattern.FLOCKING:
            adjustments = self._apply_flocking_rules(agent_positions, agent_velocities)
        elif primary_pattern == BehaviorPattern.SWARMING:
            adjustments = self._apply_swarming_rules(agent_positions, agent_velocities)
        elif primary_pattern == BehaviorPattern.DISPERSING:
            adjustments = self._apply_dispersing_rules(agent_positions, agent_velocities)
        elif primary_pattern == BehaviorPattern.SYNCHRONIZED:
            adjustments = self._apply_synchronization_rules(agent_positions, agent_velocities)

        return adjustments

    def _apply_flocking_rules(self, positions: Dict[str, Tuple[float, float]],
                            velocities: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply flocking rules: cohesion, alignment, separation."""
        adjustments = {}

        for agent_id, pos in positions.items():
            cohesion_force = self._calculate_cohesion_force(agent_id, positions)
            alignment_force = self._calculate_alignment_force(agent_id, velocities)
            separation_force = self._calculate_separation_force(agent_id, positions)

            # Combine forces with weights
            total_force = (
                cohesion_force[0] * 0.4 + alignment_force[0] * 0.4 + separation_force[0] * 0.2,
                cohesion_force[1] * 0.4 + alignment_force[1] * 0.4 + separation_force[1] * 0.2
            )

            adjustments[agent_id] = total_force

        return adjustments

    def _apply_swarming_rules(self, positions: Dict[str, Tuple[float, float]],
                            velocities: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply swarming rules: strong cohesion, weak separation."""
        adjustments = {}

        for agent_id, pos in positions.items():
            cohesion_force = self._calculate_cohesion_force(agent_id, positions)
            separation_force = self._calculate_separation_force(agent_id, positions)

            # Stronger cohesion for swarming
            total_force = (
                cohesion_force[0] * 0.7 + separation_force[0] * 0.3,
                cohesion_force[1] * 0.7 + separation_force[1] * 0.3
            )

            adjustments[agent_id] = total_force

        return adjustments

    def _apply_dispersing_rules(self, positions: Dict[str, Tuple[float, float]],
                              velocities: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply dispersing rules: move away from neighbors."""
        adjustments = {}

        for agent_id, pos in positions.items():
            separation_force = self._calculate_separation_force(agent_id, positions, strength_multiplier=2.0)

            # Only apply separation for dispersing
            adjustments[agent_id] = separation_force

        return adjustments

    def _apply_synchronization_rules(self, positions: Dict[str, Tuple[float, float]],
                                   velocities: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Apply synchronization rules: align velocities."""
        adjustments = {}

        for agent_id in positions.keys():
            alignment_force = self._calculate_alignment_force(agent_id, velocities, strength_multiplier=1.5)
            adjustments[agent_id] = alignment_force

        return adjustments

    def _calculate_cohesion_force(self, agent_id: str, positions: Dict[str, Tuple[float, float]],
                                max_distance: float = 50.0) -> Tuple[float, float]:
        """Calculate cohesion force towards group center."""
        if agent_id not in positions:
            return (0.0, 0.0)

        agent_pos = positions[agent_id]
        neighbors = []

        for other_id, other_pos in positions.items():
            if other_id != agent_id:
                distance = math.dist(agent_pos, other_pos)
                if distance <= max_distance:
                    neighbors.append(other_pos)

        if not neighbors:
            return (0.0, 0.0)

        # Calculate center of neighbors
        center_x = sum(p[0] for p in neighbors) / len(neighbors)
        center_y = sum(p[1] for p in neighbors) / len(neighbors)

        # Force towards center
        force_x = (center_x - agent_pos[0]) * self.adaptation_rate
        force_y = (center_y - agent_pos[1]) * self.adaptation_rate

        return (force_x, force_y)

    def _calculate_alignment_force(self, agent_id: str, velocities: Dict[str, Tuple[float, float]],
                                 max_distance: float = 50.0, strength_multiplier: float = 1.0) -> Tuple[float, float]:
        """Calculate alignment force to match neighbor velocities."""
        if agent_id not in velocities:
            return (0.0, 0.0)

        agent_vel = velocities[agent_id]
        agent_pos = None  # Would need positions to check distance

        neighbor_velocities = []
        for other_id, other_vel in velocities.items():
            if other_id != agent_id:
                # In a real implementation, check distance using positions
                neighbor_velocities.append(other_vel)

        if not neighbor_velocities:
            return (0.0, 0.0)

        # Calculate average velocity of neighbors
        avg_vel_x = sum(v[0] for v in neighbor_velocities) / len(neighbor_velocities)
        avg_vel_y = sum(v[1] for v in neighbor_velocities) / len(neighbor_velocities)

        # Force towards average velocity
        force_x = (avg_vel_x - agent_vel[0]) * self.adaptation_rate * strength_multiplier
        force_y = (avg_vel_y - agent_vel[1]) * self.adaptation_rate * strength_multiplier

        return (force_x, force_y)

    def _calculate_separation_force(self, agent_id: str, positions: Dict[str, Tuple[float, float]],
                                  min_distance: float = 10.0, strength_multiplier: float = 1.0) -> Tuple[float, float]:
        """Calculate separation force to avoid crowding."""
        if agent_id not in positions:
            return (0.0, 0.0)

        agent_pos = positions[agent_id]
        force_x, force_y = 0.0, 0.0

        for other_id, other_pos in positions.items():
            if other_id != agent_id:
                distance = math.dist(agent_pos, other_pos)
                if distance < min_distance and distance > 0:
                    # Repulsive force inversely proportional to distance
                    repulsion = (min_distance - distance) / distance
                    force_x += (agent_pos[0] - other_pos[0]) * repulsion
                    force_y += (agent_pos[1] - other_pos[1]) * repulsion

        # Apply adaptation rate and multiplier
        force_x *= self.adaptation_rate * strength_multiplier
        force_y *= self.adaptation_rate * strength_multiplier

        return (force_x, force_y)

    def get_behavior_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of emergent behaviors."""
        dominant_patterns = self.get_dominant_patterns()
        predicted_transition = self.predict_behavior_transition()

        return {
            'swarm_id': self.swarm_id,
            'current_dynamics': self.current_dynamics.get_dynamics_summary(),
            'dominant_patterns': [(p.value, s) for p, s in dominant_patterns],
            'pattern_history_length': {p.value: len(h) for p, h in self.pattern_history.items()},
            'behavior_transitions': len(self.behavior_transitions),
            'predicted_transition': predicted_transition,
            'self_organization_active': self.self_organization_active,
            'pattern_detection_threshold': self.pattern_detection_threshold
        }

    async def run_behavior_cycle(self) -> None:
        """Run one complete behavior analysis and adaptation cycle."""
        # This would be called periodically to update behavior analysis
        # In a real implementation, this would integrate with the swarm network

        # Decay old pattern strengths over time
        for pattern in self.pattern_history:
            if self.pattern_history[pattern]:
                # Apply small decay to recent patterns
                recent_strength = self.pattern_history[pattern][-1]
                decayed_strength = recent_strength * 0.99
                self.pattern_history[pattern][-1] = decayed_strength

        # Small delay for cycle timing
        await asyncio.sleep(0.1)