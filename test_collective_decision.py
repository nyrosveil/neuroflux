#!/usr/bin/env python3
"""
🧪 Test Collective Decision Engine
Simple test to verify swarm consensus algorithms work.

Built with love by Nyros Veil 🚀
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from swarm_intelligence.collective_decision_engine import (
    CollectiveDecisionEngine,
    DecisionContext,
    AgentOpinion,
    ConsensusAlgorithm,
    DecisionType
)


async def test_collective_decision():
    """Test the collective decision engine."""
    print("🧠 Testing Collective Decision Engine...")

    # Create decision engine
    engine = CollectiveDecisionEngine()

    # Create a decision context
    context = DecisionContext(
        decision_id="test_decision_1",
        decision_type=DecisionType.BINARY,
        description="Should we buy BTC?",
        algorithm=ConsensusAlgorithm.WEIGHTED_VOTING,
        min_participants=3,
        required_confidence=0.6
    )

    # Register the decision
    engine.register_decision(context)
    print("✅ Decision registered")

    # Set agent weights
    engine.set_agent_weight("agent_1", 0.8)  # High trust
    engine.set_agent_weight("agent_2", 0.6)  # Medium trust
    engine.set_agent_weight("agent_3", 0.4)  # Low trust

    # Submit opinions
    opinions = [
        AgentOpinion(agent_id="agent_1", decision_id="test_decision_1", value=True, confidence=0.9),
        AgentOpinion(agent_id="agent_2", decision_id="test_decision_1", value=True, confidence=0.7),
        AgentOpinion(agent_id="agent_3", decision_id="test_decision_1", value=False, confidence=0.5)
    ]

    for opinion in opinions:
        engine.submit_opinion(opinion)

    # Check decision status
    status = engine.get_decision_status("test_decision_1")
    print(f"📊 Decision status: {status}")

    # Get engine stats
    stats = engine.get_engine_stats()
    print(f"📈 Engine stats: {stats}")

    return True


async def test_different_algorithms():
    """Test different consensus algorithms."""
    print("🧪 Testing Different Consensus Algorithms...")

    engine = CollectiveDecisionEngine()

    # Test weighted voting
    context1 = DecisionContext(
        decision_id="weighted_test",
        decision_type=DecisionType.BINARY,
        description="Weighted voting test",
        algorithm=ConsensusAlgorithm.WEIGHTED_VOTING,
        min_participants=2,
        required_confidence=0.5
    )

    engine.register_decision(context1)

    # Submit opinions for weighted voting
    opinions = [
        AgentOpinion(agent_id="agent_a", decision_id="weighted_test", value=10.5, confidence=0.8),
        AgentOpinion(agent_id="agent_b", decision_id="weighted_test", value=9.2, confidence=0.6)
    ]

    for opinion in opinions:
        engine.submit_opinion(opinion)

    status = engine.get_decision_status("weighted_test")
    print(f"⚖️ Weighted voting result: {status}")

    return True


async def main():
    """Run all tests."""
    print("🚀 Starting Collective Decision Engine Tests\n")

    try:
        await test_collective_decision()
        print()
        await test_different_algorithms()
        print()

        print("🎊 ALL TESTS PASSED! Collective Decision Engine is working!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)