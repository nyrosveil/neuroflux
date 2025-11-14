# Conflict Resolution API Reference

The Conflict Resolution Engine provides intelligent conflict detection and consensus-based resolution for NeuroFlux's multi-agent system, ensuring coordinated decision-making across agents with conflicting objectives.

## Overview

The Conflict Resolution Engine handles:
- Real-time conflict detection across multiple domains
- Byzantine Fault Tolerant consensus algorithms
- Weighted voting systems based on agent performance
- Automated resolution with human-in-the-loop options
- Performance tracking and learning from resolutions

## Core Classes

### ConflictType Enum

```python
class ConflictType(Enum):
    SIGNAL_CONFLICT = "signal_conflict"          # Contradictory trading signals
    RESOURCE_CONFLICT = "resource_conflict"      # Insufficient resources
    PRIORITY_CONFLICT = "priority_conflict"      # Competing priorities
    TIMING_CONFLICT = "timing_conflict"          # Execution timing issues
    STRATEGY_CONFLICT = "strategy_conflict"      # Conflicting strategies
    RISK_CONFLICT = "risk_conflict"              # Risk assessment disagreements
```

### ConsensusAlgorithm Enum

```python
class ConsensusAlgorithm(Enum):
    SIMPLE_MAJORITY = "simple_majority"          # >50% agreement
    SUPERMAJORITY = "supermajority"              # >66% agreement
    WEIGHTED_CONSENSUS = "weighted_consensus"    # Performance-weighted
    BYZANTINE_FAULT_TOLERANCE = "bft"           # BFT consensus
    QUORUM_BASED = "quorum_based"               # Minimum participation
```

### ResolutionStrategy Enum

```python
class ResolutionStrategy(Enum):
    VOTING_CONSENSUS = "voting_consensus"        # Vote among agents
    EXPERT_PRECEDENCE = "expert_precedence"      # Domain expert override
    CONFIDENCE_THRESHOLD = "confidence_threshold" # High confidence wins
    FALLBACK_PROTOCOL = "fallback_protocol"      # Safe default action
    HUMAN_INTERVENTION = "human_intervention"    # Manual resolution
    COMPROMISE_SOLUTION = "compromise_solution"  # Balanced approach
```

### Conflict

```python
@dataclass
class Conflict:
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
```

### ConsensusVote

```python
@dataclass
class ConsensusVote:
    agent_id: str
    decision: Any
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timestamp: float
    stake: float  # Voting power (based on performance/reputation)
    evidence: Optional[Dict[str, Any]] = None
```

### ConsensusResult

```python
@dataclass
class ConsensusResult:
    consensus_id: str
    algorithm: ConsensusAlgorithm
    votes: List[ConsensusVote]
    decision: Any
    confidence: float
    achieved_at: float
    participants: int
    agreement_percentage: float
    execution_status: str = "pending"
```

## ConflictResolutionEngine Class

### Initialization

```python
engine = ConflictResolutionEngine(communication_bus: CommunicationBus)
```

### Lifecycle Methods

#### `async def start() -> None`
Start the conflict resolution engine and monitoring loop.

#### `async def stop() -> None`
Stop the conflict resolution engine and clean up resources.

### Conflict Detection

#### `async def detect_conflicts(context: Dict[str, Any]) -> List[Conflict]`
Detect conflicts in the given system context.

**Parameters:**
- `context` (Dict[str, Any]): Current system state containing agent activities, signals, and resource usage

**Returns:**
- List of detected conflicts

**Example:**
```python
context = {
    'agent_signals': {
        'agent_1': [{'symbol': 'BTC/USD', 'action': 'BUY', 'timeframe': '1h'}],
        'agent_2': [{'symbol': 'BTC/USD', 'action': 'SELL', 'timeframe': '1h'}]
    },
    'resource_requests': {
        'agent_1': [{'resource_type': 'api_calls', 'amount': 100}],
        'agent_2': [{'resource_type': 'api_calls', 'amount': 80}]
    },
    'available_resources': {'api_calls': 150}
}

conflicts = await engine.detect_conflicts(context)
for conflict in conflicts:
    print(f"Detected: {conflict.description} (severity: {conflict.severity})")
```

### Conflict Resolution

#### `async def resolve_conflict(conflict: Conflict, algorithm: ConsensusAlgorithm = ConsensusAlgorithm.WEIGHTED_CONSENSUS, strategy: ResolutionStrategy = ResolutionStrategy.VOTING_CONSENSUS) -> Optional[Dict[str, Any]]`
Resolve a conflict using specified algorithm and strategy.

**Parameters:**
- `conflict` (Conflict): The conflict to resolve
- `algorithm` (ConsensusAlgorithm): Consensus algorithm to use
- `strategy` (ResolutionStrategy): Resolution strategy

**Returns:**
- Resolution result or None if resolution failed

**Example:**
```python
from neuroflux.orchestration.conflict_resolution import ConsensusAlgorithm, ResolutionStrategy

result = await engine.resolve_conflict(
    conflict=my_conflict,
    algorithm=ConsensusAlgorithm.WEIGHTED_CONSENSUS,
    strategy=ResolutionStrategy.VOTING_CONSENSUS
)

if result:
    print(f"Resolved: {result['decision']} with confidence {result['confidence']}")
else:
    print("Resolution failed, conflict escalated")
```

### Statistics and Monitoring

#### `def get_stats() -> Dict[str, Any]`
Get conflict resolution statistics.

**Returns:**
```python
{
    'conflicts_detected': int,
    'conflicts_resolved': int,
    'consensus_attempts': int,
    'consensus_success_rate': float,
    'avg_resolution_time': float,
    'escalation_rate': float,
    'active_conflicts': int,
    'resolved_conflicts': int,
    'consensus_history_size': int
}
```

## Usage Examples

### Basic Conflict Detection

```python
from neuroflux.orchestration import ConflictResolutionEngine

# Initialize engine
engine = ConflictResolutionEngine(bus)
await engine.start()

# Monitor for conflicts
async def monitor_conflicts():
    while True:
        # Gather current system context
        context = await gather_system_context()

        # Detect conflicts
        conflicts = await engine.detect_conflicts(context)

        # Resolve detected conflicts
        for conflict in conflicts:
            if conflict.severity > 0.7:  # High priority conflicts
                await engine.resolve_conflict(conflict)
            else:
                # Queue for later resolution
                await queue_conflict(conflict)

        await asyncio.sleep(5)  # Check every 5 seconds
```

### Signal Conflict Resolution

```python
# Example: Resolving contradictory trading signals
signal_conflict = Conflict(
    conflict_id=str(uuid.uuid4()),
    conflict_type=ConflictType.SIGNAL_CONFLICT,
    description="Contradictory signals for BTC/USD",
    involved_agents=["momentum_agent", "mean_reversion_agent"],
    conflicting_elements={
        'symbol': 'BTC/USD',
        'signals': [
            {'agent': 'momentum_agent', 'action': 'BUY', 'confidence': 0.8},
            {'agent': 'mean_reversion_agent', 'action': 'SELL', 'confidence': 0.7}
        ]
    },
    severity=0.8,
    detected_at=time.time(),
    resolution_deadline=time.time() + 300
)

# Resolve using weighted consensus
resolution = await engine.resolve_conflict(
    signal_conflict,
    algorithm=ConsensusAlgorithm.WEIGHTED_CONSENSUS,
    strategy=ResolutionStrategy.VOTING_CONSENSUS
)

if resolution:
    final_action = resolution['decision']
    print(f"Consensus: {final_action}")
```

### Resource Conflict Resolution

```python
# Example: Resolving resource allocation conflicts
resource_conflict = Conflict(
    conflict_id=str(uuid.uuid4()),
    conflict_type=ConflictType.RESOURCE_CONFLICT,
    description="API rate limit exceeded",
    involved_agents=["data_agent_1", "data_agent_2", "data_agent_3"],
    conflicting_elements={
        'resource_type': 'api_calls',
        'requested': 250,
        'available': 200,
        'requests': [
            {'agent': 'data_agent_1', 'amount': 100},
            {'agent': 'data_agent_2', 'amount': 80},
            {'agent': 'data_agent_3', 'amount': 70}
        ]
    },
    severity=0.9,
    detected_at=time.time(),
    resolution_deadline=time.time() + 180
)

# Use fallback protocol for resource conflicts
resolution = await engine.resolve_conflict(
    resource_conflict,
    algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
    strategy=ResolutionStrategy.FALLBACK_PROTOCOL
)
```

### Expert Resolution

```python
# Example: Using domain expert for complex conflicts
complex_conflict = Conflict(
    conflict_id=str(uuid.uuid4()),
    conflict_type=ConflictType.STRATEGY_CONFLICT,
    description="Conflicting portfolio rebalancing strategies",
    involved_agents=["risk_agent", "alpha_agent", "beta_agent"],
    conflicting_elements={
        'strategies': [
            {'agent': 'risk_agent', 'strategy': 'risk_parity', 'allocation': {'stocks': 0.4, 'bonds': 0.6}},
            {'agent': 'alpha_agent', 'strategy': 'momentum', 'allocation': {'stocks': 0.8, 'bonds': 0.2}},
            {'agent': 'beta_agent', 'strategy': 'market_neutral', 'allocation': {'stocks': 0.5, 'bonds': 0.5}}
        ]
    },
    severity=0.6,
    detected_at=time.time(),
    resolution_deadline=time.time() + 600
)

# Defer to portfolio management expert
resolution = await engine.resolve_conflict(
    complex_conflict,
    algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
    strategy=ResolutionStrategy.EXPERT_PRECEDENCE
)
```

## Consensus Algorithms

### Simple Majority
- Requires >50% agreement among participants
- Each agent has equal voting power
- Fast resolution for straightforward conflicts

### Weighted Consensus
- Performance-weighted voting based on agent track record
- Higher-performing agents have more influence
- Better for complex decision-making

### Byzantine Fault Tolerance (BFT)
- Tolerates up to (n-1)/3 faulty agents
- Complex consensus for critical decisions
- High computational overhead

### Supermajority
- Requires >66% agreement
- Higher confidence in decisions
- Slower resolution process

### Quorum-based
- Minimum participation threshold
- Balances speed and confidence
- Configurable quorum size

## Resolution Strategies

### Voting Consensus
- Agents vote on conflict resolution
- Uses specified consensus algorithm
- Most democratic approach

### Expert Precedence
- Defers to domain expert agent
- Fast resolution for specialized conflicts
- Requires expert agent identification

### Confidence Threshold
- Selects highest confidence decision
- Individual agent responsibility
- Fast but potentially biased

### Fallback Protocol
- Uses predefined safe defaults
- Prevents system deadlock
- Conservative approach

### Human Intervention
- Escalates to human operators
- For complex or critical conflicts
- Manual resolution process

### Compromise Solution
- Finds balanced middle ground
- Negotiated settlements
- Complex implementation

## Error Handling

The Conflict Resolution Engine includes comprehensive error handling:

- **Detection Failures**: Graceful handling of conflict detection errors
- **Resolution Timeouts**: Automatic escalation of unresolved conflicts
- **Consensus Failures**: Fallback to alternative algorithms or strategies
- **Communication Errors**: Retry logic for agent communication
- **Invalid Conflicts**: Validation of conflict data structures

## Performance Considerations

- **Real-time Detection**: Efficient conflict pattern matching
- **Scalable Consensus**: Algorithms scale with agent count
- **Memory Management**: Automatic cleanup of resolved conflicts
- **Async Processing**: Non-blocking conflict resolution
- **Resource Limits**: Configurable timeouts and limits

## Integration with Other Components

The Conflict Resolution Engine integrates with:

- **Communication Bus**: For agent coordination and voting
- **Agent Registry**: For agent capability and performance data
- **Task Orchestrator**: For conflict-aware task assignment
- **Neural Swarm Network**: For distributed conflict resolution

## Cross-References

- See [Communication Bus API](communication_bus.md) for agent messaging
- See [Agent Registry API](agent_registry.md) for agent performance data
- See [Task Orchestrator API](task_orchestrator.md) for conflict-aware scheduling
- See [Base Agent Framework](../base_agent.md) for agent conflict participation</content>
<parameter name="filePath">neuroflux/docs/api/conflict_resolution.md