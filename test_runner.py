#!/usr/bin/env python3
"""
🧪 Simple Test Runner for Conflict Resolution Engine
Manual test execution to verify functionality.

Built with love by Nyros Veil 🚀
"""

import sys
import os
import asyncio
import time
import uuid
from unittest.mock import Mock, AsyncMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestration.conflict_resolution import (
    ConflictResolutionEngine,
    Conflict,
    ConflictType,
    ConsensusAlgorithm,
    ResolutionStrategy,
    ConsensusVote,
    ConsensusResult
)
from orchestration.communication_bus import CommunicationBus


async def test_basic_functionality():
    """Test basic conflict resolution functionality."""
    print("🧪 Testing Conflict Resolution Engine...")

    # Create mock communication bus
    bus = Mock(spec=CommunicationBus)
    bus.send_request = AsyncMock()
    bus.broadcast_message = AsyncMock()

    # Create engine
    engine = ConflictResolutionEngine(bus)

    # Test 1: Engine initialization
    print("✅ Engine initialized successfully")

    # Test 2: Start/stop engine
    await engine.start()
    assert engine.running
    print("✅ Engine started successfully")

    await engine.stop()
    assert not engine.running
    print("✅ Engine stopped successfully")

    # Test 3: Create and resolve a simple conflict
    conflict = Conflict(
        conflict_id=str(uuid.uuid4()),
        conflict_type=ConflictType.SIGNAL_CONFLICT,
        description="Test conflict",
        involved_agents=["agent1", "agent2"],
        conflicting_elements={"test": "data"},
        severity=0.8,
        detected_at=time.time(),
        resolution_deadline=time.time() + 300
    )

    # Mock vote gathering
    async def mock_gather_votes(conflict):
        return [
            ConsensusVote("agent1", "BUY", 0.8, "reason1", time.time(), 1.0),
            ConsensusVote("agent2", "BUY", 0.7, "reason2", time.time(), 0.8)
        ]

    engine._gather_votes = mock_gather_votes

    # Mock consensus algorithm
    async def mock_consensus(votes, conflict):
        return ConsensusResult(
            consensus_id=str(uuid.uuid4()),
            algorithm=ConsensusAlgorithm.SIMPLE_MAJORITY,
            votes=votes,
            decision="BUY",
            confidence=0.75,
            achieved_at=time.time(),
            participants=2,
            agreement_percentage=1.0
        )

    engine._simple_majority_consensus = mock_consensus

    # Resolve conflict
    result = await engine.resolve_conflict(
        conflict, ConsensusAlgorithm.SIMPLE_MAJORITY, ResolutionStrategy.VOTING_CONSENSUS
    )

    assert result is not None
    assert result['decision'] == "BUY"
    assert conflict.status == "resolved"
    print("✅ Conflict resolved successfully")

    # Test 4: Statistics tracking
    stats = engine.get_stats()
    assert stats['conflicts_resolved'] == 1
    assert len(engine.resolved_conflicts) == 1
    print("✅ Statistics tracking works")

    print("🎉 All basic tests passed!")


async def test_conflict_detection():
    """Test conflict detection algorithms."""
    print("🧪 Testing Conflict Detection...")

    bus = Mock(spec=CommunicationBus)
    engine = ConflictResolutionEngine(bus)

    # Test signal conflict detection
    context = {
        'agent_signals': {
            'agent1': [{'symbol': 'BTC', 'timeframe': '1H', 'action': 'BUY', 'confidence': 0.8}],
            'agent2': [{'symbol': 'BTC', 'timeframe': '1H', 'action': 'SELL', 'confidence': 0.7}],
            'agent3': [{'symbol': 'BTC', 'timeframe': '1H', 'action': 'BUY', 'confidence': 0.9}]
        }
    }

    conflicts = await engine._detect_signal_conflicts(context)
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.SIGNAL_CONFLICT
    print("✅ Signal conflict detection works")

    # Test resource conflict detection
    context = {
        'resource_requests': {
            'agent1': [{'resource_type': 'cpu', 'amount': 60}],
            'agent2': [{'resource_type': 'cpu', 'amount': 50}]
        },
        'available_resources': {'cpu': 100}
    }

    conflicts = await engine._detect_resource_conflicts(context)
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == ConflictType.RESOURCE_CONFLICT
    print("✅ Resource conflict detection works")

    print("🎉 Conflict detection tests passed!")


async def test_consensus_algorithms():
    """Test consensus algorithms."""
    print("🧪 Testing Consensus Algorithms...")

    bus = Mock(spec=CommunicationBus)
    engine = ConflictResolutionEngine(bus)

    # Create test votes
    votes = [
        ConsensusVote("agent1", "BUY", 0.8, "reason1", time.time(), 1.0),
        ConsensusVote("agent2", "SELL", 0.7, "reason2", time.time(), 0.8),
        ConsensusVote("agent3", "BUY", 0.9, "reason3", time.time(), 1.2)
    ]

    conflict = Conflict(
        conflict_id=str(uuid.uuid4()),
        conflict_type=ConflictType.SIGNAL_CONFLICT,
        description="Test",
        involved_agents=["agent1", "agent2", "agent3"],
        conflicting_elements={},
        severity=0.8,
        detected_at=time.time(),
        resolution_deadline=time.time() + 300
    )

    # Test simple majority
    result = await engine._simple_majority_consensus(votes, conflict)
    assert result is not None
    assert result.decision == "BUY"
    assert result.agreement_percentage > 0.5  # Simple majority should have >50% agreement
    print("✅ Simple majority consensus works")

    # Test weighted consensus
    result = await engine._weighted_consensus(votes, conflict)
    assert result is not None
    assert result.decision == "BUY"
    print("✅ Weighted consensus works")

    print("🎉 Consensus algorithm tests passed!")


async def main():
    """Run all tests."""
    print("🚀 Starting Conflict Resolution Engine Tests\n")

    try:
        await test_basic_functionality()
        print()
        await test_conflict_detection()
        print()
        await test_consensus_algorithms()
        print()

        print("🎊 ALL TESTS PASSED! Conflict Resolution Engine is working correctly!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)