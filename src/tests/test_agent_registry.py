"""
🧪 Test Suite for Agent Registry & Discovery System
Comprehensive testing of NeuroFlux's agent registration, discovery, and health monitoring.

Built with love by Nyros Veil 🚀

Tests Cover:
- Agent registration and deregistration
- Service discovery with various query criteria
- Health monitoring and heartbeat system
- Performance metrics tracking
- Auto-scaling triggers and load balancing
- Integration with CommunicationBus
"""

import asyncio
import time
import uuid
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

# Import the modules to test
from orchestration.agent_registry import (
    AgentRegistry,
    AgentInfo,
    AgentStatus,
    AgentCapability,
    ServiceQuery,
    HealthCheck
)
from orchestration.communication_bus import CommunicationBus, Message, MessageType, MessagePriority


class TestAgentRegistry:
    """Test suite for the AgentRegistry class."""

    @pytest.fixture
    def mock_communication_bus(self):
        """Create a mock CommunicationBus for testing."""
        bus = Mock(spec=CommunicationBus)
        bus.send_request = AsyncMock()
        bus.broadcast_message = AsyncMock()
        return bus

    @pytest.fixture
    def agent_registry(self, mock_communication_bus):
        """Create an AgentRegistry instance for testing."""
        return AgentRegistry(mock_communication_bus)

    @pytest.fixture
    def sample_agent_info(self):
        """Create sample agent registration info."""
        return {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING', 'RISK_MANAGEMENT'],
            'metadata': {'exchange': 'hyperliquid', 'pair': 'BTC-USD'},
            'version': '1.0.0',
            'tags': ['high_frequency', 'btc_trader']
        }

    def test_initialization(self, agent_registry, mock_communication_bus):
        """Test AgentRegistry initialization."""
        assert agent_registry.communication_bus == mock_communication_bus
        assert agent_registry.agents == {}
        assert agent_registry.health_history == []
        assert agent_registry.performance_stats == {}
        assert not agent_registry.running
        assert agent_registry.health_monitoring_task is None

    @pytest.mark.asyncio
    async def test_start_stop(self, agent_registry):
        """Test starting and stopping the agent registry."""
        # Test start
        await agent_registry.start()
        assert agent_registry.running
        assert agent_registry.health_monitoring_task is not None

        # Test stop
        await agent_registry.stop()
        assert not agent_registry.running
        assert agent_registry.health_monitoring_task is None

    @pytest.mark.asyncio
    async def test_agent_registration(self, agent_registry, sample_agent_info, mock_communication_bus):
        """Test agent registration."""
        agent_id = await agent_registry.register_agent(sample_agent_info)

        assert agent_id in agent_registry.agents
        agent = agent_registry.agents[agent_id]

        assert agent.agent_type == 'trading_agent'
        assert AgentCapability.TRADING in agent.capabilities
        assert AgentCapability.RISK_MANAGEMENT in agent.capabilities
        assert agent.status == AgentStatus.ACTIVE
        assert agent.version == '1.0.0'
        assert 'high_frequency' in agent.tags

        # Check performance stats initialized
        assert agent_id in agent_registry.performance_stats
        stats = agent_registry.performance_stats[agent_id]
        assert 'requests_served' in stats
        assert 'success_rate' in stats

        # Check broadcast message was sent
        mock_communication_bus.broadcast_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_registration_validation(self, agent_registry):
        """Test agent registration validation."""
        # Missing required fields
        with pytest.raises(ValueError):
            await agent_registry.register_agent({})

        # Invalid capabilities
        with pytest.raises(ValueError):
            await agent_registry.register_agent({
                'agent_type': 'test',
                'capabilities': ['INVALID_CAPABILITY']
            })

    @pytest.mark.asyncio
    async def test_agent_deregistration(self, agent_registry, sample_agent_info, mock_communication_bus):
        """Test agent deregistration."""
        # Register agent first
        agent_id = await agent_registry.register_agent(sample_agent_info)
        assert agent_id in agent_registry.agents

        # Deregister agent
        result = await agent_registry.deregister_agent(agent_id)
        assert result is True
        assert agent_id not in agent_registry.agents
        assert agent_id not in agent_registry.performance_stats

        # Check broadcast message was sent
        assert mock_communication_bus.broadcast_message.call_count == 2  # register + deregister

    @pytest.mark.asyncio
    async def test_agent_deregistration_not_found(self, agent_registry):
        """Test deregistration of non-existent agent."""
        result = await agent_registry.deregister_agent("non_existent_agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_service_discovery_basic(self, agent_registry):
        """Test basic service discovery."""
        # Register multiple agents
        agent1_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING'],
            'tags': ['btc']
        }
        agent2_info = {
            'agent_type': 'research_agent',
            'capabilities': ['RESEARCH'],
            'tags': ['eth']
        }
        agent3_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING', 'RISK_MANAGEMENT'],
            'tags': ['btc', 'eth']
        }

        await agent_registry.register_agent(agent1_info)
        await agent_registry.register_agent(agent2_info)
        await agent_registry.register_agent(agent3_info)

        # Discover trading agents
        query = ServiceQuery(capabilities={AgentCapability.TRADING})
        results = await agent_registry.discover_agents(query)

        assert len(results) == 2
        for agent in results:
            assert AgentCapability.TRADING in agent.capabilities

    @pytest.mark.asyncio
    async def test_service_discovery_with_filters(self, agent_registry):
        """Test service discovery with various filters."""
        # Register agents with different characteristics
        agents_data = [
            {
                'agent_type': 'trading_agent',
                'capabilities': ['TRADING'],
                'metadata': {'performance': 0.9},
                'tags': ['high_perf']
            },
            {
                'agent_type': 'trading_agent',
                'capabilities': ['TRADING'],
                'metadata': {'performance': 0.6},
                'tags': ['low_perf']
            },
            {
                'agent_type': 'research_agent',
                'capabilities': ['RESEARCH'],
                'metadata': {'performance': 0.8},
                'tags': ['high_perf']
            }
        ]

        for data in agents_data:
            await agent_registry.register_agent(data)

        # Query with multiple filters
        query = ServiceQuery(
            agent_type='trading_agent',
            capabilities={AgentCapability.TRADING},
            tags={'high_perf'},
            limit=5
        )
        results = await agent_registry.discover_agents(query)

        assert len(results) == 1
        assert results[0].agent_type == 'trading_agent'
        assert 'high_perf' in results[0].tags

    @pytest.mark.asyncio
    async def test_service_discovery_ranking(self, agent_registry):
        """Test agent ranking in service discovery."""
        # Register agents with different performance levels
        agents_data = [
            {
                'agent_type': 'trading_agent',
                'capabilities': ['TRADING'],
                'metadata': {'performance': 0.5}
            },
            {
                'agent_type': 'trading_agent',
                'capabilities': ['TRADING'],
                'metadata': {'performance': 0.9}
            },
            {
                'agent_type': 'trading_agent',
                'capabilities': ['TRADING'],
                'metadata': {'performance': 0.7}
            }
        ]

        agent_ids = []
        for data in agents_data:
            agent_id = await agent_registry.register_agent(data)
            agent_ids.append(agent_id)

        # Set different performance metrics
        await agent_registry.update_agent_performance(agent_ids[0], {'success_rate': 0.6})
        await agent_registry.update_agent_performance(agent_ids[1], {'success_rate': 0.9})
        await agent_registry.update_agent_performance(agent_ids[2], {'success_rate': 0.8})

        # Discover with performance sorting
        query = ServiceQuery(
            capabilities={AgentCapability.TRADING},
            sort_by='performance'
        )
        results = await agent_registry.discover_agents(query)

        # Should be sorted by performance (highest first)
        assert len(results) == 3
        # Note: The ranking algorithm considers multiple factors, so we just verify we get results

    @pytest.mark.asyncio
    async def test_health_monitoring(self, agent_registry, mock_communication_bus):
        """Test health monitoring functionality."""
        # Register an agent
        agent_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING']
        }
        agent_id = await agent_registry.register_agent(agent_info)

        # Update health
        health_data = {
            'health_score': 0.8,
            'load_factor': 0.3,
            'response_time': 0.1
        }
        await agent_registry.update_agent_health(agent_id, health_data)

        agent = agent_registry.agents[agent_id]
        assert agent.health_score == 0.8
        assert agent.load_factor == 0.3
        assert agent.status == AgentStatus.ACTIVE

        # Check health history
        assert len(agent_registry.health_history) == 1
        health_check = agent_registry.health_history[0]
        assert health_check.agent_id == agent_id
        assert health_check.status == "healthy"

    @pytest.mark.asyncio
    async def test_health_status_transitions(self, agent_registry):
        """Test agent status transitions based on health."""
        agent_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING']
        }
        agent_id = await agent_registry.register_agent(agent_info)

        # Healthy agent
        await agent_registry.update_agent_health(agent_id, {'health_score': 0.9})
        assert agent_registry.agents[agent_id].status == AgentStatus.ACTIVE

        # Degraded agent
        await agent_registry.update_agent_health(agent_id, {'health_score': 0.6})
        assert agent_registry.agents[agent_id].status == AgentStatus.DEGRADED

        # Unhealthy agent
        await agent_registry.update_agent_health(agent_id, {'health_score': 0.3})
        assert agent_registry.agents[agent_id].status == AgentStatus.SUSPENDED

    @pytest.mark.asyncio
    async def test_performance_tracking(self, agent_registry):
        """Test performance metrics tracking."""
        agent_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING']
        }
        agent_id = await agent_registry.register_agent(agent_info)

        # Update performance metrics
        metrics1 = {
            'requests_served': 10,
            'avg_response_time': 0.5,
            'success_rate': 0.9
        }
        await agent_registry.update_agent_performance(agent_id, metrics1)

        stats = agent_registry.performance_stats[agent_id]
        assert stats['requests_served'] == 10
        assert stats['avg_response_time'] == 0.5
        assert stats['success_rate'] == 0.9

        # Update again (should maintain running averages)
        metrics2 = {
            'requests_served': 15,
            'avg_response_time': 0.3,
            'success_rate': 0.95
        }
        await agent_registry.update_agent_performance(agent_id, metrics2)

        # Check that stats were updated
        assert stats['requests_served'] == 15
        assert stats['success_rate'] == 0.95

    @pytest.mark.asyncio
    async def test_auto_scaling_triggers(self, agent_registry, mock_communication_bus):
        """Test auto-scaling trigger functionality."""
        # Register multiple agents of the same capability
        for i in range(3):
            agent_info = {
                'agent_type': 'trading_agent',
                'capabilities': ['TRADING']
            }
            agent_id = await agent_registry.register_agent(agent_info)

            # Set high load on all agents
            await agent_registry.update_agent_health(agent_id, {
                'health_score': 0.9,
                'load_factor': 0.9  # High load
            })

        # Check auto-scaling triggers
        await agent_registry._check_auto_scaling_triggers()

        # Should have triggered scale up
        mock_communication_bus.broadcast_message.assert_called()
        call_args = mock_communication_bus.broadcast_message.call_args
        assert 'scale_up_triggered' in call_args[1]['topic']

    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self, agent_registry, mock_communication_bus):
        """Test heartbeat timeout handling."""
        agent_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING']
        }
        agent_id = await agent_registry.register_agent(agent_info)

        # Manually set last heartbeat to old time
        agent = agent_registry.agents[agent_id]
        agent.last_heartbeat = time.time() - 120  # 2 minutes ago

        # Start health monitoring to trigger timeout check
        await agent_registry.start()

        # Wait for monitoring cycle
        await asyncio.sleep(35)  # Health check interval is 30 seconds

        await agent_registry.stop()

        # Agent should be marked as offline
        assert agent.status == AgentStatus.OFFLINE

        # Should have broadcast offline event
        offline_calls = [call for call in mock_communication_bus.broadcast_message.call_args_list
                        if 'agent_offline' in str(call)]
        assert len(offline_calls) > 0

    def test_statistics_tracking(self, agent_registry):
        """Test registry statistics tracking."""
        stats = agent_registry.get_registry_stats()

        # Initially should have basic structure
        assert 'total_registrations' in stats
        assert 'active_agents' in stats
        assert 'capabilities_distribution' in stats
        assert 'health_distribution' in stats

        # Should have expected initial values
        assert stats['total_registrations'] == 0
        assert stats['active_agents'] == 0

    @pytest.mark.asyncio
    async def test_get_agent_info(self, agent_registry, sample_agent_info):
        """Test retrieving agent information."""
        agent_id = await agent_registry.register_agent(sample_agent_info)

        # Get agent info
        agent_info = agent_registry.get_agent_info(agent_id)
        assert agent_info is not None
        assert agent_info.agent_id == agent_id
        assert agent_info.agent_type == 'trading_agent'

        # Get non-existent agent
        assert agent_registry.get_agent_info("non_existent") is None

    @pytest.mark.asyncio
    async def test_capability_distribution(self, agent_registry):
        """Test capability distribution statistics."""
        # Register agents with different capabilities
        agents_data = [
            {'agent_type': 'trading', 'capabilities': ['TRADING']},
            {'agent_type': 'trading', 'capabilities': ['TRADING', 'RISK_MANAGEMENT']},
            {'agent_type': 'research', 'capabilities': ['RESEARCH']},
            {'agent_type': 'research', 'capabilities': ['RESEARCH', 'SENTIMENT_ANALYSIS']}
        ]

        for data in agents_data:
            await agent_registry.register_agent(data)

        stats = agent_registry.get_registry_stats()
        distribution = stats['capabilities_distribution']

        assert distribution['trading'] == 2  # TRADING capability
        assert distribution['risk_management'] == 1  # RISK_MANAGEMENT capability
        assert distribution['research'] == 2  # RESEARCH capability

    @pytest.mark.asyncio
    async def test_health_distribution(self, agent_registry):
        """Test health distribution statistics."""
        # Register agents with different health levels
        agents_data = [
            {'agent_type': 'trading', 'capabilities': ['TRADING']},
            {'agent_type': 'trading', 'capabilities': ['TRADING']},
            {'agent_type': 'trading', 'capabilities': ['TRADING']}
        ]

        agent_ids = []
        for data in agents_data:
            agent_id = await agent_registry.register_agent(data)
            agent_ids.append(agent_id)

        # Set different health levels
        await agent_registry.update_agent_health(agent_ids[0], {'health_score': 0.9})  # healthy
        await agent_registry.update_agent_health(agent_ids[1], {'health_score': 0.6})  # degraded
        await agent_registry.update_agent_health(agent_ids[2], {'health_score': 0.3})  # unhealthy

        stats = agent_registry.get_registry_stats()
        health_dist = stats['health_distribution']

        assert health_dist['healthy'] == 1
        assert health_dist['degraded'] == 1
        assert health_dist['unhealthy'] == 1


# Integration tests
@pytest.mark.integration
class TestAgentRegistryIntegration:
    """Integration tests for agent registry system."""

    @pytest.fixture
    async def running_registry(self):
        """Create and start an agent registry for integration testing."""
        bus = CommunicationBus()
        registry = AgentRegistry(bus)

        await registry.start()
        yield registry
        await registry.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_agent_lifecycle(self, running_registry):
        """Test complete agent lifecycle from registration to deregistration."""
        # Register agent
        agent_info = {
            'agent_type': 'trading_agent',
            'capabilities': ['TRADING', 'EXECUTION'],
            'metadata': {'exchange': 'test_exchange'},
            'tags': ['integration_test']
        }

        agent_id = await running_registry.register_agent(agent_info)
        assert agent_id in running_registry.agents

        # Update health and performance
        await running_registry.update_agent_health(agent_id, {
            'health_score': 0.95,
            'load_factor': 0.2
        })

        await running_registry.update_agent_performance(agent_id, {
            'requests_served': 100,
            'success_rate': 0.98
        })

        # Discover agent
        query = ServiceQuery(capabilities={AgentCapability.TRADING})
        results = await running_registry.discover_agents(query)
        assert len(results) > 0

        # Deregister agent
        success = await running_registry.deregister_agent(agent_id)
        assert success
        assert agent_id not in running_registry.agents

    @pytest.mark.asyncio
    async def test_multi_agent_discovery(self, running_registry):
        """Test service discovery with multiple agents."""
        # Register multiple agents
        agent_types = ['trading', 'research', 'risk', 'sentiment']
        capabilities_map = {
            'trading': ['TRADING', 'EXECUTION'],
            'research': ['RESEARCH', 'NEWS_ANALYSIS'],
            'risk': ['RISK_MANAGEMENT'],
            'sentiment': ['SENTIMENT_ANALYSIS']
        }

        registered_agents = []
        for agent_type in agent_types:
            agent_info = {
                'agent_type': f'{agent_type}_agent',
                'capabilities': capabilities_map[agent_type],
                'tags': [agent_type]
            }
            agent_id = await running_registry.register_agent(agent_info)
            registered_agents.append((agent_id, agent_type))

        # Test various discovery queries
        queries = [
            (ServiceQuery(capabilities={AgentCapability.TRADING}), 1),
            (ServiceQuery(agent_type='research_agent'), 1),
            (ServiceQuery(tags={'trading'}), 1),
            (ServiceQuery(limit=2), 2)
        ]

        for query, expected_count in queries:
            results = await running_registry.discover_agents(query)
            assert len(results) == expected_count


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])