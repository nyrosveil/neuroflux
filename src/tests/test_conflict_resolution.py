"""
🧪 Test Suite for Conflict Resolution Engine
Comprehensive testing of NeuroFlux's intelligent conflict detection and resolution system.

Built with love by Nyros Veil 🚀

Tests Cover:
- Conflict detection algorithms (signal, resource, priority, timing)
- Consensus algorithms (simple majority, weighted, BFT, quorum)
- Resolution strategies (voting, expert, confidence, fallback, human)
- Monitoring loop and timeout handling
- Statistics tracking and analytics
- Integration with CommunicationBus
"""

import asyncio
import time
import uuid
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import the modules to test
from orchestration.conflict_resolution import (
    ConflictResolutionEngine,
    Conflict,
    ConflictType,
    ConsensusAlgorithm,
    ResolutionStrategy,
    ConsensusVote,
    ConsensusResult
)
from orchestration.communication_bus import CommunicationBus, Message, MessageType, MessagePriority


class TestConflictResolutionEngine:
    """Test suite for the ConflictResolutionEngine class."""

    @pytest.fixture
    def mock_communication_bus(self):
        """Create a mock CommunicationBus for testing."""
        bus = Mock(spec=CommunicationBus)
        bus.send_request = AsyncMock()
        bus.broadcast_message = AsyncMock()
        return bus

    @pytest.fixture
    def conflict_resolution_engine(self, mock_communication_bus):
        """Create a ConflictResolutionEngine instance for testing."""
        return ConflictResolutionEngine(mock_communication_bus)

    @pytest.fixture
    def sample_conflict(self):
        """Create a sample conflict for testing."""
        return Conflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type=ConflictType.SIGNAL_CONFLICT,
            description="Test signal conflict",
            involved_agents=["agent1", "agent2"],
            conflicting_elements={"test": "data"},
            severity=0.8,
            detected_at=time.time(),
            resolution_deadline=time.time() + 300
        )

    @pytest.fixture
    def sample_votes(self):
        """Create sample consensus votes for testing."""
        return [
            ConsensusVote(
                agent_id="agent1",
                decision="BUY",
                confidence=0.8,
                reasoning="Strong bullish signal",
                timestamp=time.time(),
                stake=1.0
            ),
            ConsensusVote(
                agent_id="agent2",
                decision="SELL",
                confidence=0.7,
                reasoning="Bearish divergence",
                timestamp=time.time(),
                stake=0.8
            ),
            ConsensusVote(
                agent_id="agent3",
                decision="BUY",
                confidence=0.9,
                reasoning="Confirmed uptrend",
                timestamp=time.time(),
                stake=1.2
            )
        ]

    def test_initialization(self, conflict_resolution_engine, mock_communication_bus):
        """Test ConflictResolutionEngine initialization."""
        assert conflict_resolution_engine.communication_bus == mock_communication_bus
        assert conflict_resolution_engine.active_conflicts == {}
        assert conflict_resolution_engine.resolved_conflicts == []
        assert conflict_resolution_engine.consensus_history == []
        assert not conflict_resolution_engine.running
        assert conflict_resolution_engine.monitoring_task is None

    @pytest.mark.asyncio
    async def test_start_stop(self, conflict_resolution_engine):
        """Test starting and stopping the conflict resolution engine."""
        # Test start
        await conflict_resolution_engine.start()
        assert conflict_resolution_engine.running
        assert conflict_resolution_engine.monitoring_task is not None

        # Test stop
        await conflict_resolution_engine.stop()
        assert not conflict_resolution_engine.running
        assert conflict_resolution_engine.monitoring_task is None

    def test_conflict_creation(self):
        """Test Conflict dataclass creation."""
        conflict_id = str(uuid.uuid4())
        detected_at = time.time()

        conflict = Conflict(
            conflict_id=conflict_id,
            conflict_type=ConflictType.SIGNAL_CONFLICT,
            description="Test conflict",
            involved_agents=["agent1", "agent2"],
            conflicting_elements={"signal": "contradiction"},
            severity=0.8,
            detected_at=detected_at,
            resolution_deadline=detected_at + 300
        )

        assert conflict.conflict_id == conflict_id
        assert conflict.conflict_type == ConflictType.SIGNAL_CONFLICT
        assert conflict.description == "Test conflict"
        assert conflict.involved_agents == ["agent1", "agent2"]
        assert conflict.severity == 0.8
        assert conflict.status == "detected"
        assert conflict.resolution is None
        assert conflict.resolved_at is None

    def test_consensus_vote_creation(self):
        """Test ConsensusVote dataclass creation."""
        vote = ConsensusVote(
            agent_id="agent1",
            decision="BUY",
            confidence=0.8,
            reasoning="Strong signal",
            timestamp=time.time(),
            stake=1.0
        )

        assert vote.agent_id == "agent1"
        assert vote.decision == "BUY"
        assert vote.confidence == 0.8
        assert vote.reasoning == "Strong signal"
        assert vote.stake == 1.0
        assert vote.evidence is None

    def test_consensus_result_creation(self):
        """Test ConsensusResult dataclass creation."""
        votes = [
            ConsensusVote("agent1", "BUY", 0.8, "reason1", time.time(), 1.0),
            ConsensusVote("agent2", "SELL", 0.7, "reason2", time.time(), 0.8)
        ]

        result = ConsensusResult(
            consensus_id=str(uuid.uuid4()),
            algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
            votes=votes,
            decision="BUY",
            confidence=0.8,
            achieved_at=time.time(),
            participants=2,
            agreement_percentage=0.6
        )

        assert result.algorithm == ConsensusAlgorithm.SIMPLE_MAJORITY
        assert result.decision == "BUY"
        assert result.confidence == 0.8
        assert result.participants == 2
        assert result.agreement_percentage == 0.6
        assert result.execution_status == "pending"

    @pytest.mark.asyncio
    async def test_detect_signal_conflicts(self, conflict_resolution_engine):
        """Test signal conflict detection."""
        # Test context with conflicting signals
        context = {
            'agent_signals': {
                'agent1': [
                    {'symbol': 'BTC', 'timeframe': '1H', 'action': 'BUY', 'confidence': 0.8}
                ],
                'agent2': [
                    {'symbol': 'BTC', 'timeframe': '1H', 'action': 'SELL', 'confidence': 0.7}
                ],
                'agent3': [
                    {'symbol': 'BTC', 'timeframe': '1H', 'action': 'BUY', 'confidence': 0.9}
                ]
            }
        }

        conflicts = await conflict_resolution_engine._detect_signal_conflicts(context)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.SIGNAL_CONFLICT
        assert 'BTC_1H' in conflict.description
        assert conflict.involved_agents == ['agent1', 'agent2', 'agent3']
        assert conflict.severity == 0.8

    @pytest.mark.asyncio
    async def test_detect_resource_conflicts(self, conflict_resolution_engine):
        """Test resource conflict detection."""
        context = {
            'resource_requests': {
                'agent1': [{'resource_type': 'cpu', 'amount': 60}],
                'agent2': [{'resource_type': 'cpu', 'amount': 50}],
                'agent3': [{'resource_type': 'cpu', 'amount': 40}]
            },
            'available_resources': {'cpu': 100}
        }

        conflicts = await conflict_resolution_engine._detect_resource_conflicts(context)

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.conflict_type == ConflictType.RESOURCE_CONFLICT
        assert 'cpu' in conflict.description
        assert conflict.severity == 0.9

    @pytest.mark.asyncio
    async def test_simple_majority_consensus(self, conflict_resolution_engine, sample_conflict, sample_votes):
        """Test simple majority consensus algorithm."""
        result = await conflict_resolution_engine._simple_majority_consensus(sample_votes, sample_conflict)

        assert result is not None
        assert result.algorithm == ConsensusAlgorithm.SIMPLE_MAJORITY
        assert result.decision == "BUY"  # 2 BUY vs 1 SELL
        assert result.participants == 3
        assert result.agreement_percentage == 2/3  # 2 out of 3 votes for BUY

    @pytest.mark.asyncio
    async def test_weighted_consensus(self, conflict_resolution_engine, sample_conflict, sample_votes):
        """Test weighted consensus algorithm."""
        result = await conflict_resolution_engine._weighted_consensus(sample_votes, sample_conflict)

        assert result is not None
        assert result.algorithm == ConsensusAlgorithm.WEIGHTED_CONSENSUS
        assert result.decision == "BUY"  # Higher weighted stake for BUY
        assert result.participants == 3

    @pytest.mark.asyncio
    async def test_supermajority_consensus(self, conflict_resolution_engine, sample_conflict, sample_votes):
        """Test supermajority consensus algorithm."""
        # Create votes with supermajority for BUY
        supermajority_votes = [
            ConsensusVote("agent1", "BUY", 0.8, "reason1", time.time(), 1.0),
            ConsensusVote("agent2", "BUY", 0.7, "reason2", time.time(), 0.8),
            ConsensusVote("agent3", "BUY", 0.9, "reason3", time.time(), 1.2)
        ]

        result = await conflict_resolution_engine._supermajority_consensus(supermajority_votes, sample_conflict)

        assert result is not None
        assert result.algorithm == ConsensusAlgorithm.SUPERMAJORITY
        assert result.agreement_percentage > 0.66

    @pytest.mark.asyncio
    async def test_bft_consensus(self, conflict_resolution_engine, sample_conflict, sample_votes):
        """Test Byzantine Fault Tolerant consensus."""
        # Create a larger set of agents for BFT
        bft_conflict = Conflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type=ConflictType.SIGNAL_CONFLICT,
            description="BFT test conflict",
            involved_agents=["agent1", "agent2", "agent3", "agent4"],
            conflicting_elements={},
            severity=0.8,
            detected_at=time.time(),
            resolution_deadline=time.time() + 300
        )

        bft_votes = sample_votes + [
            ConsensusVote("agent4", "BUY", 0.6, "reason4", time.time(), 0.9)
        ]

        result = await conflict_resolution_engine._bft_consensus(bft_votes, bft_conflict)

        assert result is not None
        assert result.algorithm == ConsensusAlgorithm.BYZANTINE_FAULT_TOLERANCE

    @pytest.mark.asyncio
    async def test_quorum_consensus(self, conflict_resolution_engine, sample_conflict, sample_votes):
        """Test quorum-based consensus."""
        result = await conflict_resolution_engine._quorum_consensus(sample_votes, sample_conflict)

        assert result is not None
        assert result.algorithm == ConsensusAlgorithm.SIMPLE_MAJORITY  # Falls back to simple majority

    @pytest.mark.asyncio
    async def test_resolve_by_voting(self, conflict_resolution_engine, sample_conflict, mock_communication_bus):
        """Test voting-based conflict resolution."""
        # Mock the consensus algorithm and vote gathering
        with patch.object(conflict_resolution_engine, '_gather_votes', return_value=[
            ConsensusVote("agent1", "BUY", 0.8, "reason1", time.time(), 1.0),
            ConsensusVote("agent2", "BUY", 0.7, "reason2", time.time(), 0.8)
        ]):
            with patch.object(conflict_resolution_engine, '_simple_majority_consensus') as mock_consensus:
                mock_consensus.return_value = ConsensusResult(
                    consensus_id=str(uuid.uuid4()),
                    algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
                    votes=[],
                    decision="BUY",
                    confidence=0.75,
                    achieved_at=time.time(),
                    participants=2,
                    agreement_percentage=1.0
                )

                result = await conflict_resolution_engine._resolve_by_voting(
                    sample_conflict, ConsensusAlgorithm.SIMPLE_MAJORITY
                )

                assert result is not None
                assert result['decision'] == "BUY"
                assert result['confidence'] == 0.75
                assert result['participants'] == 2

    @pytest.mark.asyncio
    async def test_resolve_by_expert(self, conflict_resolution_engine, sample_conflict, mock_communication_bus):
        """Test expert-based conflict resolution."""
        # Mock communication bus response
        mock_response = {'decision': 'HOLD', 'confidence': 0.9}
        mock_communication_bus.send_request.return_value = mock_response

        # Mock expert finding
        with patch.object(conflict_resolution_engine, '_find_domain_expert', return_value="expert_agent"):
            result = await conflict_resolution_engine._resolve_by_expert(
                sample_conflict, ConsensusAlgorithm.SIMPLE_MAJORITY
            )

            assert result is not None
            assert result['decision'] == 'HOLD'
            assert result['expert'] == 'expert_agent'
            assert result['confidence'] == 0.9

    @pytest.mark.asyncio
    async def test_resolve_by_confidence(self, conflict_resolution_engine, sample_conflict, mock_communication_bus):
        """Test confidence-based conflict resolution."""
        # Mock confidence gathering
        with patch.object(conflict_resolution_engine, '_gather_confidence_scores', return_value={
            'agent1': {'confidence': 0.6, 'decision': 'BUY'},
            'agent2': {'confidence': 0.9, 'decision': 'SELL'}
        }):
            result = await conflict_resolution_engine._resolve_by_confidence(
                sample_conflict, ConsensusAlgorithm.SIMPLE_MAJORITY
            )

            assert result is not None
            assert result['decision'] == 'SELL'  # Higher confidence wins
            assert result['agent'] == 'agent2'
            assert result['confidence'] == 0.9

    @pytest.mark.asyncio
    async def test_resolve_by_fallback(self, conflict_resolution_engine, sample_conflict):
        """Test fallback protocol resolution."""
        result = await conflict_resolution_engine._resolve_by_fallback(
            sample_conflict, ConsensusAlgorithm.SIMPLE_MAJORITY
        )

        assert result is not None
        assert result['decision'] == 'HOLD'  # Fallback for signal conflict
        assert result['strategy'] == 'fallback_protocol'
        assert result['confidence'] == 0.5

    @pytest.mark.asyncio
    async def test_resolve_by_human(self, conflict_resolution_engine, sample_conflict):
        """Test human intervention resolution."""
        result = await conflict_resolution_engine._resolve_by_human(
            sample_conflict, ConsensusAlgorithm.SIMPLE_MAJORITY
        )

        assert result is not None
        assert result['decision'] == 'PENDING_HUMAN_REVIEW'
        assert result['strategy'] == 'human_intervention'

    @pytest.mark.asyncio
    async def test_full_conflict_resolution_workflow(self, conflict_resolution_engine, mock_communication_bus):
        """Test complete conflict resolution workflow."""
        # Create a conflict
        conflict = Conflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type=ConflictType.SIGNAL_CONFLICT,
            description="Workflow test conflict",
            involved_agents=["agent1", "agent2"],
            conflicting_elements={},
            severity=0.8,
            detected_at=time.time(),
            resolution_deadline=time.time() + 300
        )

        # Mock vote gathering
        with patch.object(conflict_resolution_engine, '_gather_votes', return_value=[
            ConsensusVote("agent1", "BUY", 0.8, "reason1", time.time(), 1.0),
            ConsensusVote("agent2", "BUY", 0.7, "reason2", time.time(), 0.8)
        ]):
            with patch.object(conflict_resolution_engine, '_simple_majority_consensus') as mock_consensus:
                mock_consensus.return_value = ConsensusResult(
                    consensus_id=str(uuid.uuid4()),
                    algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
                    votes=[],
                    decision="BUY",
                    confidence=0.75,
                    achieved_at=time.time(),
                    participants=2,
                    agreement_percentage=1.0
                )

                # Resolve the conflict
                result = await conflict_resolution_engine.resolve_conflict(
                    conflict, ConsensusAlgorithm.SIMPLE_MAJORITY, ResolutionStrategy.VOTING_CONSENSUS
                )

                assert result is not None
                assert conflict.status == "resolved"
                assert conflict.resolution == result
                assert conflict.resolved_at is not None

                # Check statistics
                stats = conflict_resolution_engine.get_stats()
                assert stats['conflicts_resolved'] == 1
                assert len(conflict_resolution_engine.resolved_conflicts) == 1

    @pytest.mark.asyncio
    async def test_conflict_timeout_escalation(self, conflict_resolution_engine, mock_communication_bus):
        """Test conflict timeout and escalation."""
        # Create an expired conflict
        past_time = time.time() - 400  # 400 seconds ago
        conflict = Conflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type=ConflictType.SIGNAL_CONFLICT,
            description="Timeout test conflict",
            involved_agents=["agent1"],
            conflicting_elements={},
            severity=0.8,
            detected_at=past_time,
            resolution_deadline=past_time + 300  # Already expired
        )

        conflict_resolution_engine.active_conflicts[conflict.conflict_id] = conflict

        # Start monitoring to trigger escalation
        await conflict_resolution_engine.start()

        # Wait for monitoring loop to check
        await asyncio.sleep(11)  # Monitoring checks every 10 seconds

        # Stop monitoring
        await conflict_resolution_engine.stop()

        # Check that conflict was escalated
        assert conflict.status == "escalated"
        mock_communication_bus.broadcast_message.assert_called()

    def test_statistics_tracking(self, conflict_resolution_engine):
        """Test statistics tracking functionality."""
        # Initially all stats should be 0
        stats = conflict_resolution_engine.get_stats()
        assert stats['conflicts_detected'] == 0
        assert stats['conflicts_resolved'] == 0
        assert stats['consensus_attempts'] == 0
        assert stats['active_conflicts'] == 0
        assert stats['resolved_conflicts'] == 0

        # Add some mock data
        conflict_resolution_engine.stats['conflicts_detected'] = 5
        conflict_resolution_engine.stats['conflicts_resolved'] = 3
        conflict_resolution_engine.stats['consensus_attempts'] = 4

        # Add active and resolved conflicts
        conflict_resolution_engine.active_conflicts['test1'] = Mock()
        conflict_resolution_engine.active_conflicts['test2'] = Mock()
        conflict_resolution_engine.resolved_conflicts.append(Mock())
        conflict_resolution_engine.consensus_history.append(Mock())

        stats = conflict_resolution_engine.get_stats()
        assert stats['conflicts_detected'] == 5
        assert stats['conflicts_resolved'] == 3
        assert stats['consensus_attempts'] == 4
        assert stats['active_conflicts'] == 2
        assert stats['resolved_conflicts'] == 1
        assert stats['consensus_history_size'] == 1

    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, conflict_resolution_engine):
        """Test agent performance and stake calculation."""
        # Set up agent performance data
        conflict_resolution_engine.agent_performance = {
            'agent1': {'overall': 0.8, 'signal_conflict': 0.9},
            'agent2': {'overall': 0.6, 'signal_conflict': 0.7},
            'agent3': {'overall': 0.9, 'signal_conflict': 0.5}
        }

        # Test stake calculation
        stake1 = conflict_resolution_engine._calculate_agent_stake('agent1', ConflictType.SIGNAL_CONFLICT)
        stake2 = conflict_resolution_engine._calculate_agent_stake('agent2', ConflictType.SIGNAL_CONFLICT)
        stake3 = conflict_resolution_engine._calculate_agent_stake('agent3', ConflictType.SIGNAL_CONFLICT)

        # Agent1 should have highest stake (good overall + good type performance)
        assert stake1 > stake2
        assert stake1 > stake3

        # All stakes should be within bounds
        assert 0.1 <= stake1 <= 2.0
        assert 0.1 <= stake2 <= 2.0
        assert 0.1 <= stake3 <= 2.0

    @pytest.mark.asyncio
    async def test_domain_expert_finding(self, conflict_resolution_engine):
        """Test domain expert identification."""
        # Set up agent performance data
        conflict_resolution_engine.agent_performance = {
            'agent1': {'signal_conflict': 0.6},
            'agent2': {'signal_conflict': 0.9},  # Best expert
            'agent3': {'signal_conflict': 0.7}
        }

        conflict = Conflict(
            conflict_id=str(uuid.uuid4()),
            conflict_type=ConflictType.SIGNAL_CONFLICT,
            description="Expert test",
            involved_agents=["agent1", "agent2", "agent3"],
            conflicting_elements={},
            severity=0.8,
            detected_at=time.time(),
            resolution_deadline=time.time() + 300
        )

        expert = await conflict_resolution_engine._find_domain_expert(conflict)
        assert expert == "agent2"  # Should pick the highest scoring agent


# Integration tests
@pytest.mark.integration
class TestConflictResolutionIntegration:
    """Integration tests for conflict resolution system."""

    @pytest.fixture
    async def running_engine(self):
        """Create and start a conflict resolution engine for integration testing."""
        bus = CommunicationBus()
        engine = ConflictResolutionEngine(bus)

        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_conflict_resolution(self, running_engine):
        """Test complete end-to-end conflict resolution process."""
        # This would be a full integration test with real communication
        # For now, just verify the engine can start and stop
        assert running_engine.running
        assert running_engine.monitoring_task is not None

        stats = running_engine.get_stats()
        assert isinstance(stats, dict)
        assert 'conflicts_detected' in stats


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])