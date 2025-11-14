"""
🧠 NeuroFlux Orchestrator v3.2
Advanced multi-agent orchestration system with dynamic agent discovery and task management.

Built with love by Nyros Veil 🚀

Features:
- Dynamic agent registration and discovery
- Task orchestration with dependency management
- Conflict resolution and consensus building
- Real-time health monitoring and auto-scaling
- Communication bus for inter-agent coordination
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

from orchestration import (
    CommunicationBus,
    TaskOrchestrator,
    ConflictResolutionEngine,
    AgentRegistry,
    AgentCapability as OrchestrationCapability,
    ServiceQuery
)
from orchestration.task_orchestrator import Task, TaskPriority

# Load environment variables
load_dotenv()


class NeuroFluxOrchestratorV32:
    """
    Advanced NeuroFlux orchestrator using the new orchestration system.

    This orchestrator manages agents through:
    - Agent Registry for discovery and health monitoring
    - Task Orchestrator for dynamic task assignment
    - Conflict Resolution for consensus building
    - Communication Bus for inter-agent messaging
    """

    def __init__(self):
        # Core orchestration components
        self.communication_bus = CommunicationBus()
        self.agent_registry = AgentRegistry(self.communication_bus)
        self.conflict_resolution = ConflictResolutionEngine(self.communication_bus, self.agent_registry)
        self.task_orchestrator = TaskOrchestrator(self.communication_bus, self.agent_registry, self.conflict_resolution)

        # System state
        self.running = False
        self.agent_tasks = {}  # Maps agent_id to asyncio tasks
        self.system_tasks = []  # Background system tasks

        # Configuration
        self.cycle_interval = 300  # 5 minutes between cycles
        self.max_concurrent_tasks = 10
        self.health_check_interval = 60  # 1 minute

        # Statistics
        self.stats = {
            'cycles_completed': 0,
            'tasks_created': 0,
            'tasks_completed': 0,
            'conflicts_resolved': 0,
            'agents_registered': 0,
            'start_time': time.time()
        }

    async def initialize(self) -> None:
        """Initialize the orchestration system."""
        cprint("🧠 Initializing NeuroFlux Orchestrator v3.2...", "cyan", attrs=['bold'])

        # Start core components
        await self.communication_bus.start()
        await self.task_orchestrator.start()
        await self.conflict_resolution.start()
        await self.agent_registry.start()

        # Register system agents
        await self._register_system_agents()

        # Start background tasks
        self.system_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._cycle_scheduler()),
            asyncio.create_task(self._conflict_monitoring_loop())
        ]

        self.running = True
        cprint("✅ NeuroFlux Orchestrator v3.2 initialized", "green")

    async def shutdown(self) -> None:
        """Shutdown the orchestration system."""
        cprint("🛑 Shutting down NeuroFlux Orchestrator v3.2...", "yellow")

        self.running = False

        # Cancel system tasks
        for task in self.system_tasks:
            task.cancel()
        await asyncio.gather(*self.system_tasks, return_exceptions=True)

        # Stop core components
        await self.agent_registry.stop()
        await self.conflict_resolution.stop()
        await self.task_orchestrator.stop()
        await self.communication_bus.stop()

        cprint("✅ NeuroFlux Orchestrator v3.2 shutdown complete", "green")

    async def _register_system_agents(self) -> None:
        """Register the core NeuroFlux agents with the registry."""
        cprint("🤖 Registering NeuroFlux agents...", "blue")

        # Define agent configurations
        agent_configs = [
            {
                'agent_type': 'risk_management',
                'capabilities': ['RISK_MANAGEMENT'],
                'metadata': {'priority': 'critical', 'cycle_interval': 60},
                'tags': ['core', 'safety', 'monitoring']
            },
            {
                'agent_type': 'sentiment_analysis',
                'capabilities': ['SENTIMENT_ANALYSIS'],
                'metadata': {'priority': 'high', 'cycle_interval': 300},
                'tags': ['analysis', 'market_data']
            },
            {
                'agent_type': 'trading_execution',
                'capabilities': ['TRADING', 'EXECUTION'],
                'metadata': {'priority': 'high', 'cycle_interval': 60},
                'tags': ['core', 'execution']
            },
            {
                'agent_type': 'research_analysis',
                'capabilities': ['RESEARCH', 'NEWS_ANALYSIS'],
                'metadata': {'priority': 'medium', 'cycle_interval': 600},
                'tags': ['analysis', 'research']
            },
            {
                'agent_type': 'technical_analysis',
                'capabilities': ['CHART_ANALYSIS'],
                'metadata': {'priority': 'medium', 'cycle_interval': 300},
                'tags': ['analysis', 'technical']
            },
            {
                'agent_type': 'funding_monitor',
                'capabilities': ['FUNDING_ANALYSIS'],
                'metadata': {'priority': 'low', 'cycle_interval': 300},
                'tags': ['monitoring', 'funding']
            },
            {
                'agent_type': 'liquidation_monitor',
                'capabilities': ['LIQUIDATION_MONITORING'],
                'metadata': {'priority': 'medium', 'cycle_interval': 60},
                'tags': ['monitoring', 'risk']
            },
            {
                'agent_type': 'whale_tracker',
                'capabilities': ['WHALE_TRACKING'],
                'metadata': {'priority': 'low', 'cycle_interval': 300},
                'tags': ['monitoring', 'whales']
            },
            {
                'agent_type': 'market_data',
                'capabilities': ['RESEARCH'],
                'metadata': {'priority': 'medium', 'cycle_interval': 180},
                'tags': ['data', 'coingecko']
            },
            {
                'agent_type': 'strategy_optimizer',
                'capabilities': ['STRATEGY_OPTIMIZATION', 'BACKTESTING'],
                'metadata': {'priority': 'low', 'cycle_interval': 3600},
                'tags': ['optimization', 'backtesting']
            },
            {
                'agent_type': 'sniper_agent',
                'capabilities': ['TRADING'],
                'metadata': {'priority': 'low', 'cycle_interval': 30},
                'tags': ['trading', 'opportunities']
            },
             {
                 'agent_type': 'swarm_intelligence',
                 'capabilities': ['RESEARCH', 'SENTIMENT_ANALYSIS'],
                 'metadata': {'priority': 'high', 'cycle_interval': 300},
                 'tags': ['consensus', 'swarm']
             },
             {
                 'agent_type': 'ml_prediction',
                 'capabilities': ['ML_PREDICTION', 'TIME_SERIES_ANALYSIS'],
                 'metadata': {'priority': 'high', 'cycle_interval': 180},
                 'tags': ['ml', 'prediction', 'forecasting']
             }
        ]

        # Register each agent
        for config in agent_configs:
            try:
                agent_id = await self.agent_registry.register_agent(config)
                self.stats['agents_registered'] += 1
                cprint(f"✅ Registered {config['agent_type']} ({agent_id})", "green")
            except Exception as e:
                cprint(f"❌ Failed to register {config['agent_type']}: {e}", "red")

        cprint(f"🤖 Registered {self.stats['agents_registered']} agents", "blue")

    async def create_trading_cycle_tasks(self) -> List[Task]:
        """Create tasks for a complete trading cycle."""
        tasks = []

        # 1. Risk Assessment Task (Critical)
        risk_task = Task(
            task_id=f"risk_assessment_{int(time.time())}",
            name="Risk Assessment",
            description="Assess current portfolio risk and market conditions",
            task_type="risk_analysis",
            priority=TaskPriority.CRITICAL,
            payload={
                'analysis_type': 'comprehensive',
                'include_positions': True,
                'include_market_risk': True
            },
            required_capabilities=['RISK_MANAGEMENT'],
            estimated_duration=60,
            timeout=120
        )
        tasks.append(risk_task)

        # 2. Market Data Collection Task
        market_data_task = Task(
            task_id=f"market_data_{int(time.time())}",
            name="Market Data Collection",
            description="Collect current market data and metrics",
            task_type="data_collection",
            priority=TaskPriority.HIGH,
            payload={
                'sources': ['coingecko', 'exchange_api'],
                'tokens': ['BTC', 'ETH', 'SOL', 'ADA'],
                'include_funding_rates': True
            },
            required_capabilities=['RESEARCH'],
            estimated_duration=30,
            timeout=60
        )
        tasks.append(market_data_task)

        # 2.5. ML Price Prediction Task (depends on market data)
        ml_prediction_task = Task(
            task_id=f"ml_price_prediction_{int(time.time())}",
            name="ML Price Prediction",
            description="Generate price predictions using machine learning models",
            task_type="ml_prediction",
            priority=TaskPriority.HIGH,
            payload={
                'tokens': ['BTC', 'ETH', 'SOL'],
                'prediction_horizon': 60,  # 1 hour ahead
                'models': ['arima', 'exponential_smoothing', 'simple_moving_average'],
                'confidence_threshold': 0.7,
                'include_confidence_intervals': True
            },
            dependencies=[market_data_task.task_id],
            required_capabilities=['ML_PREDICTION', 'TIME_SERIES_ANALYSIS'],
            estimated_duration=120,
            timeout=300
        )
        tasks.append(ml_prediction_task)

        # 2.6. ML Volume Prediction Task (depends on market data)
        ml_volume_task = Task(
            task_id=f"ml_volume_prediction_{int(time.time())}",
            name="ML Volume Prediction",
            description="Predict trading volume using ML models",
            task_type="ml_prediction",
            priority=TaskPriority.MEDIUM,
            payload={
                'tokens': ['BTC', 'ETH', 'SOL'],
                'prediction_horizon': 30,  # 30 minutes ahead
                'models': ['arima', 'exponential_smoothing'],
                'volume_types': ['total_volume', 'buy_volume', 'sell_volume']
            },
            dependencies=[market_data_task.task_id],
            required_capabilities=['ML_PREDICTION', 'TIME_SERIES_ANALYSIS'],
            estimated_duration=90,
            timeout=240
        )
        tasks.append(ml_volume_task)

        # 4. Sentiment Analysis Task
        sentiment_task = Task(
            task_id=f"sentiment_analysis_{int(time.time())}",
            name="Sentiment Analysis",
            description="Analyze market sentiment from multiple sources",
            task_type="sentiment_analysis",
            priority=TaskPriority.HIGH,
            payload={
                'tokens': ['BTC', 'ETH', 'SOL'],
                'sources': ['news', 'social_media', 'technical_indicators'],
                'timeframe': '24h'
            },
            required_capabilities=['SENTIMENT_ANALYSIS'],
            estimated_duration=120,
            timeout=300
        )
        tasks.append(sentiment_task)

        # 5. Technical Analysis Task
        technical_task = Task(
            task_id=f"technical_analysis_{int(time.time())}",
            name="Technical Analysis",
            description="Perform technical analysis on key trading pairs",
            task_type="technical_analysis",
            priority=TaskPriority.MEDIUM,
            payload={
                'tokens': ['BTC', 'ETH', 'SOL'],
                'indicators': ['rsi', 'macd', 'bollinger_bands', 'volume'],
                'timeframes': ['1h', '4h', '1d']
            },
            required_capabilities=['CHART_ANALYSIS'],
            estimated_duration=90,
            timeout=180
        )
        tasks.append(technical_task)

        # 8. Research Analysis Task (depends on market data)
        research_task = Task(
            task_id=f"research_analysis_{int(time.time())}",
            name="Research Analysis",
            description="Perform fundamental research and news analysis",
            task_type="fundamental_research",
            priority=TaskPriority.MEDIUM,
            payload={
                'focus_areas': ['defi', 'infrastructure', 'adoption'],
                'include_news': True,
                'include_on_chain': True
            },
            dependencies=[market_data_task.task_id],
            required_capabilities=['RESEARCH', 'NEWS_ANALYSIS'],
            estimated_duration=180,
            timeout=600
        )
        tasks.append(research_task)

        # 9. Whale Tracking Task
        whale_task = Task(
            task_id=f"whale_tracking_{int(time.time())}",
            name="Whale Movement Tracking",
            description="Monitor large wallet movements and accumulation",
            task_type="whale_monitoring",
            priority=TaskPriority.LOW,
            payload={
                'threshold_usd': 1000000,
                'timeframe': '24h',
                'include_exchanges': True
            },
            required_capabilities=['WHALE_TRACKING'],
            estimated_duration=60,
            timeout=120
        )
        tasks.append(whale_task)

        # 11. Funding Rate Analysis Task
        funding_task = Task(
            task_id=f"funding_analysis_{int(time.time())}",
            name="Funding Rate Analysis",
            description="Analyze funding rates for arbitrage opportunities",
            task_type="funding_analysis",
            priority=TaskPriority.LOW,
            payload={
                'pairs': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
                'include_predictions': True
            },
            required_capabilities=['FUNDING_ANALYSIS'],
            estimated_duration=45,
            timeout=90
        )
        tasks.append(funding_task)

        # 12. Liquidation Monitoring Task
        liquidation_task = Task(
            task_id=f"liquidation_monitoring_{int(time.time())}",
            name="Liquidation Risk Monitoring",
            description="Monitor liquidation levels and potential cascades",
            task_type="liquidation_monitoring",
            priority=TaskPriority.MEDIUM,
            payload={
                'exchanges': ['hyperliquid', 'binance', 'bybit'],
                'alert_threshold': 10000000  # $10M
            },
            required_capabilities=['LIQUIDATION_MONITORING'],
            estimated_duration=30,
            timeout=60
        )
        tasks.append(liquidation_task)

        # 13. Swarm Intelligence Consensus Task (depends on sentiment and technical)
        swarm_task = Task(
            task_id=f"swarm_consensus_{int(time.time())}",
            name="Swarm Intelligence Consensus",
            description="Build consensus from multiple analysis sources",
            task_type="consensus_building",
            priority=TaskPriority.HIGH,
            payload={
                'input_sources': ['sentiment', 'technical', 'research'],
                'consensus_method': 'weighted_voting',
                'min_agreement_threshold': 0.7
            },
            dependencies=[sentiment_task.task_id, technical_task.task_id],
            required_capabilities=['RESEARCH', 'SENTIMENT_ANALYSIS'],
            estimated_duration=90,
            timeout=180
        )
        tasks.append(swarm_task)

        # 12. Trading Decision Task (depends on risk, swarm consensus, and ML predictions)
        trading_task = Task(
            task_id=f"trading_decision_{int(time.time())}",
            name="Trading Decision",
            description="Make trading decisions based on all analysis including ML predictions",
            task_type="trading_execution",
            priority=TaskPriority.HIGH,
            payload={
                'decision_criteria': ['risk_approved', 'consensus_positive', 'ml_confidence_threshold', 'opportunity_score'],
                'ml_confidence_threshold': 0.7,  # Minimum confidence for ML-based decisions
                'max_positions': 5,
                'risk_limits': {'max_position_size': 0.1, 'max_total_risk': 0.3},
                'ml_weight': 0.4,  # Weight given to ML predictions in decision making
                'use_predictions_for_timing': True  # Use ML predictions for entry/exit timing
            },
            dependencies=[risk_task.task_id, swarm_task.task_id, ml_prediction_task.task_id, ml_volume_task.task_id],
            required_capabilities=['TRADING', 'EXECUTION'],
            estimated_duration=60,
            timeout=120
        )
        tasks.append(trading_task)

        return tasks

    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Run a complete trading cycle using the orchestration system."""
        cprint(f"\n🧠 Starting Trading Cycle #{self.stats['cycles_completed'] + 1}", "cyan", attrs=['bold'])

        cycle_start = time.time()
        cycle_results = {
            'cycle_id': int(cycle_start),
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'conflicts_detected': 0,
            'start_time': cycle_start
        }

        try:
            # Create cycle tasks
            tasks = await self.create_trading_cycle_tasks()
            cycle_results['tasks_created'] = len(tasks)

            # Submit tasks to orchestrator
            for task in tasks:
                await self.task_orchestrator.submit_task(
                    name=task.name,
                    description=task.description,
                    task_type=task.task_type,
                    payload=task.payload,
                    priority=task.priority,
                    dependencies=task.dependencies,
                    required_capabilities=task.required_capabilities,
                    estimated_duration=task.estimated_duration,
                    timeout=task.timeout
                )
                self.stats['tasks_created'] += 1

            # Wait for tasks to complete (with timeout)
            await asyncio.sleep(10)  # Let orchestrator start assigning tasks

            # Monitor task completion
            completed_tasks = 0
            failed_tasks = 0
            max_wait_time = 900  # 15 minutes max per cycle
            wait_start = time.time()

            while time.time() - wait_start < max_wait_time:
                # Check task status
                active_tasks = [t for t in self.task_orchestrator.tasks.values()
                              if t.status in ['PENDING', 'ASSIGNED', 'RUNNING']]

                if not active_tasks:
                    break

                await asyncio.sleep(5)

            # Count final results
            for task in tasks:
                task_obj = self.task_orchestrator.tasks.get(task.task_id)
                if task_obj:
                    if task_obj.status == 'COMPLETED':
                        completed_tasks += 1
                    elif task_obj.status in ['FAILED', 'TIMEOUT', 'CANCELLED']:
                        failed_tasks += 1

            cycle_results['tasks_completed'] = completed_tasks
            cycle_results['tasks_failed'] = failed_tasks

            # Update statistics
            self.stats['cycles_completed'] += 1
            self.stats['tasks_completed'] += completed_tasks

            cycle_time = time.time() - cycle_start
            cprint(f"✅ Cycle completed in {cycle_time:.1f}s: {completed_tasks}/{len(tasks)} tasks successful", "green")

        except Exception as e:
            cprint(f"❌ Cycle failed: {e}", "red")
            cycle_results['error'] = str(e)

        return cycle_results

    async def _health_monitoring_loop(self) -> None:
        """Monitor agent health and update registry."""
        while self.running:
            try:
                # Get all registered agents
                registry_stats = self.agent_registry.get_registry_stats()

                # Log health summary
                healthy = registry_stats.get('healthy_agents', 0)
                total = registry_stats.get('registered_agents', 0)

                if total > 0:
                    health_rate = healthy / total
                    cprint(f"💚 Agent Health: {healthy}/{total} ({health_rate:.1%})", "green" if health_rate > 0.8 else "yellow")

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                cprint(f"⚠️ Health monitoring error: {e}", "yellow")
                await asyncio.sleep(30)

    async def _cycle_scheduler(self) -> None:
        """Schedule regular trading cycles."""
        while self.running:
            try:
                # Run trading cycle
                await self.run_trading_cycle()

                # Wait for next cycle
                await asyncio.sleep(self.cycle_interval)

            except Exception as e:
                cprint(f"❌ Cycle scheduler error: {e}", "red")
                await asyncio.sleep(60)

    async def _conflict_monitoring_loop(self) -> None:
        """Monitor and resolve conflicts between agents."""
        while self.running:
            try:
                # Check for conflicts in the conflict resolution engine
                active_conflicts = len(self.conflict_resolution.active_conflicts)

                if active_conflicts > 0:
                    cprint(f"⚖️ Active conflicts: {active_conflicts}", "yellow")

                    # Try to resolve conflicts
                    resolved = await self.conflict_resolution.resolve_pending_conflicts()
                    if resolved > 0:
                        self.stats['conflicts_resolved'] += resolved
                        cprint(f"✅ Resolved {resolved} conflicts", "green")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                cprint(f"⚠️ Conflict monitoring error: {e}", "yellow")
                await asyncio.sleep(30)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        registry_stats = self.agent_registry.get_registry_stats()

        return {
            'system': {
                'running': self.running,
                'version': '3.2',
                'uptime': time.time() - self.stats['start_time']
            },
            'orchestration': {
                'cycles_completed': self.stats['cycles_completed'],
                'tasks_created': self.stats['tasks_created'],
                'tasks_completed': self.stats['tasks_completed'],
                'conflicts_resolved': self.stats['conflicts_resolved']
            },
            'agents': {
                'registered': registry_stats.get('registered_agents', 0),
                'active': registry_stats.get('active_agents', 0),
                'healthy': registry_stats.get('healthy_agents', 0),
                'degraded': registry_stats.get('degraded_agents', 0),
                'capabilities': registry_stats.get('capabilities_distribution', {})
            },
            'communication': {
                'messages_processed': self.communication_bus.stats.get('messages_processed', 0),
                'active_connections': len(self.communication_bus.agent_queues)
            },
            'timestamp': datetime.now().isoformat()
        }

    async def run_continuous(self, max_cycles: Optional[int] = None) -> None:
        """Run the orchestrator continuously."""
        cprint("🚀 NeuroFlux Orchestrator v3.2 - Continuous Mode", "green", attrs=['bold'])

        try:
            await self.initialize()

            cycle_count = 0
            while self.running and (max_cycles is None or cycle_count < max_cycles):
                # The cycle scheduler runs automatically
                await asyncio.sleep(60)  # Check every minute
                cycle_count = self.stats['cycles_completed']

                # Print status update every 5 cycles
                if cycle_count > 0 and cycle_count % 5 == 0:
                    status = self.get_system_status()
                    cprint(f"📊 Status: {cycle_count} cycles, {status['agents']['active']} agents active", "blue")

        except KeyboardInterrupt:
            cprint("\n🛑 Orchestrator stopped by user", "red", attrs=['bold'])
        except Exception as e:
            cprint(f"\n❌ Orchestrator crashed: {e}", "red", attrs=['bold'])
        finally:
            await self.shutdown()