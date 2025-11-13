#!/usr/bin/env python3
"""
🧪 Simple Test Runner for Agent Registry & Discovery System
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

from orchestration.agent_registry import (
    AgentRegistry,
    AgentInfo,
    AgentStatus,
    AgentCapability,
    ServiceQuery,
    HealthCheck
)
from orchestration.communication_bus import CommunicationBus


async def test_basic_functionality():
    """Test basic agent registry functionality."""
    print("🗂️ Testing Agent Registry & Discovery System...")

    # Create mock communication bus
    bus = Mock(spec=CommunicationBus)
    bus.send_request = AsyncMock()
    bus.broadcast_message = AsyncMock()

    # Create registry
    registry = AgentRegistry(bus)

    # Test 1: Registry initialization
    print("✅ Registry initialized successfully")

    # Test 2: Start/stop registry
    await registry.start()
    assert registry.running
    print("✅ Registry started successfully")

    await registry.stop()
    assert not registry.running
    print("✅ Registry stopped successfully")

    # Test 3: Agent registration
    agent_info = {
        'agent_type': 'trading_agent',
        'capabilities': ['TRADING', 'EXECUTION'],
        'metadata': {'exchange': 'test_exchange'},
        'version': '1.0.0',
        'tags': ['test_agent']
    }

    agent_id = await registry.register_agent(agent_info)
    assert agent_id in registry.agents

    agent = registry.agents[agent_id]
    assert agent.agent_type == 'trading_agent'
    assert AgentCapability.TRADING in agent.capabilities
    assert AgentCapability.EXECUTION in agent.capabilities
    assert agent.status == AgentStatus.ACTIVE
    print("✅ Agent registration works")

    # Test 4: Service discovery
    query = ServiceQuery(capabilities={AgentCapability.TRADING})
    results = await registry.discover_agents(query)
    assert len(results) == 1
    assert results[0].agent_id == agent_id
    print("✅ Service discovery works")

    # Test 5: Health monitoring
    await registry.update_agent_health(agent_id, {
        'health_score': 0.9,
        'load_factor': 0.3,
        'response_time': 0.1
    })

    agent = registry.agents[agent_id]
    assert agent.health_score == 0.9
    assert agent.load_factor == 0.3
    print("✅ Health monitoring works")

    # Test 6: Performance tracking
    await registry.update_agent_performance(agent_id, {
        'requests_served': 100,
        'success_rate': 0.95,
        'avg_response_time': 0.2
    })

    stats = registry.performance_stats[agent_id]
    assert stats['requests_served'] == 100
    assert stats['success_rate'] == 0.95
    print("✅ Performance tracking works")

    # Test 7: Agent deregistration
    success = await registry.deregister_agent(agent_id)
    assert success
    assert agent_id not in registry.agents
    print("✅ Agent deregistration works")

    print("🎉 All basic tests passed!")


async def test_service_discovery():
    """Test advanced service discovery features."""
    print("🧪 Testing Service Discovery...")

    bus = Mock(spec=CommunicationBus)
    registry = AgentRegistry(bus)

    # Register multiple agents with different characteristics
    agents_data = [
        {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING'],
            'tags': ['btc', 'high_freq']
        },
        {
            'agent_type': 'research_agent',
            'capabilities': ['RESEARCH', 'NEWS_ANALYSIS'],
            'tags': ['btc', 'analysis']
        },
        {
            'agent_type': 'risk_agent',
            'capabilities': ['RISK_MANAGEMENT'],
            'tags': ['btc', 'safety']
        },
        {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING', 'EXECUTION'],
            'tags': ['eth', 'high_freq']
        }
    ]

    agent_ids = []
    for data in agents_data:
        agent_id = await registry.register_agent(data)
        agent_ids.append(agent_id)

    # Test capability-based discovery
    trading_query = ServiceQuery(capabilities={AgentCapability.TRADING})
    trading_results = await registry.discover_agents(trading_query)
    assert len(trading_results) == 2
    print("✅ Capability-based discovery works")

    # Test tag-based discovery
    btc_query = ServiceQuery(tags={'btc'})
    btc_results = await registry.discover_agents(btc_query)
    assert len(btc_results) == 3
    print("✅ Tag-based discovery works")

    # Test combined filters
    combined_query = ServiceQuery(
        capabilities={AgentCapability.TRADING},
        tags={'high_freq'}
    )
    combined_results = await registry.discover_agents(combined_query)
    assert len(combined_results) == 2
    print("✅ Combined filter discovery works")

    # Test agent type filtering
    research_query = ServiceQuery(agent_type='research_agent')
    research_results = await registry.discover_agents(research_query)
    assert len(research_results) == 1
    assert research_results[0].agent_type == 'research_agent'
    print("✅ Agent type filtering works")

    print("🎉 Service discovery tests passed!")


async def test_health_monitoring():
    """Test health monitoring and status transitions."""
    print("🧪 Testing Health Monitoring...")

    bus = Mock(spec=CommunicationBus)
    registry = AgentRegistry(bus)

    # Register an agent
    agent_info = {
        'agent_type': 'trading_agent',
        'capabilities': ['TRADING']
    }
    agent_id = await registry.register_agent(agent_info)

    # Test healthy agent
    await registry.update_agent_health(agent_id, {'health_score': 0.95})
    assert registry.agents[agent_id].status == AgentStatus.ACTIVE
    print("✅ Healthy agent status works")

    # Test degraded agent
    await registry.update_agent_health(agent_id, {'health_score': 0.7})
    assert registry.agents[agent_id].status == AgentStatus.DEGRADED
    print("✅ Degraded agent status works")

    # Test unhealthy agent
    await registry.update_agent_health(agent_id, {'health_score': 0.4})
    assert registry.agents[agent_id].status == AgentStatus.SUSPENDED
    print("✅ Unhealthy agent status works")

    # Test recovery
    await registry.update_agent_health(agent_id, {'health_score': 0.9})
    assert registry.agents[agent_id].status == AgentStatus.ACTIVE
    print("✅ Agent recovery works")

    print("🎉 Health monitoring tests passed!")


async def test_statistics():
    """Test registry statistics and analytics."""
    print("🧪 Testing Statistics & Analytics...")

    bus = Mock(spec=CommunicationBus)
    registry = AgentRegistry(bus)

    # Register multiple agents
    capabilities_list = [
        ['TRADING'],
        ['TRADING', 'EXECUTION'],
        ['RESEARCH'],
        ['RISK_MANAGEMENT'],
        ['TRADING', 'RISK_MANAGEMENT']
    ]

    for i, caps in enumerate(capabilities_list):
        agent_info = {
            'agent_type': f'agent_{i}',
            'capabilities': caps
        }
        await registry.register_agent(agent_info)

    # Get statistics
    stats = registry.get_registry_stats()

    # Check basic stats
    assert stats['total_registrations'] == 5
    assert stats['active_agents'] == 5
    print("✅ Basic statistics work")

    # Check capability distribution
    cap_dist = stats['capabilities_distribution']
    assert cap_dist['trading'] == 3  # TRADING appears in 3 agents
    assert cap_dist['execution'] == 1  # EXECUTION appears in 1 agent
    assert cap_dist['research'] == 1  # RESEARCH appears in 1 agent
    assert cap_dist['risk_management'] == 2  # RISK_MANAGEMENT appears in 2 agents
    print("✅ Capability distribution works")

    # Check health distribution (all should be healthy initially)
    health_dist = stats['health_distribution']
    assert health_dist['healthy'] == 5
    assert health_dist['degraded'] == 0
    assert health_dist['unhealthy'] == 0
    print("✅ Health distribution works")

    print("🎉 Statistics tests passed!")


async def main():
    """Run all tests."""
    print("🚀 Starting Agent Registry & Discovery Tests\n")

    try:
        await test_basic_functionality()
        print()
        await test_service_discovery()
        print()
        await test_health_monitoring()
        print()
        await test_statistics()
        print()

        print("🎊 ALL TESTS PASSED! Agent Registry & Discovery System is working correctly!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)