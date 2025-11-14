"""
🧠 NeuroFlux Main Orchestrator
Multi-Agent Trading System with Neuro-Flux Adaptation
Built with love by Nyros Veil 🚀

This is the main entry point for the NeuroFlux trading system.
It orchestrates multiple AI agents for comprehensive market analysis and trading.
"""

import os
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from termcolor import colored, cprint
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import NeuroFlux components
import config

# Import Analytics Engine for Phase 4.3.6 Integration
try:
    from analytics import AnalyticsEngine
    ANALYTICS_ENABLED = True
    cprint("📊 Analytics Engine imported successfully", "green")
except ImportError as e:
    cprint(f"⚠️  Analytics Engine not available: {e}", "yellow")
    ANALYTICS_ENABLED = False
    AnalyticsEngine = None

# ============================================================================
# ACTIVE AGENTS CONFIGURATION
# ============================================================================

# Master switch for all agents
AGENTS_ENABLED = True

# Individual agent enable/disable switches
ACTIVE_AGENTS = {
    # Core Trading Agents
    'trading_agent': True,        # Main trading execution
    'risk_agent': True,          # Risk management and position monitoring
    'strategy_agent': True,      # Strategy-based trading signals

    # Market Analysis Agents
    'sentiment_agent': True,     # Market sentiment analysis
    'research_agent': True,      # Fundamental research and news
    'chartanalysis_agent': True, # Technical analysis
    'coingecko_agent': True,     # CoinGecko data and metrics

    # Specialized Agents
    'funding_agent': True,       # Funding rate analysis
    'liquidation_agent': True,   # Liquidation monitoring
    'whale_agent': True,         # Large wallet tracking
    'websearch_agent': True,     # Web search and data gathering

    # Advanced Agents
    'copybot_agent': False,      # Copy trading (disabled by default)
    'sniper_agent': True,        # New launch sniping
    'swarm_agent': True,         # Swarm intelligence consensus
    'rbi_agent': True,           # Research-Backed Intelligence
    'ml_prediction_agent': True, # ML-based price and volume predictions

    # Backtesting
    'backtest_runner': False,    # Automated backtesting (run manually)
}

# Agent execution priorities (higher = runs first)
AGENT_PRIORITIES = {
    'risk_agent': 10,           # Always run risk checks first
    'sentiment_agent': 8,       # Market sentiment affects all decisions
    'research_agent': 7,        # Fundamental analysis
    'chartanalysis_agent': 6,   # Technical analysis
    'coingecko_agent': 5,       # Market data
    'funding_agent': 4,         # Funding rates
    'liquidation_agent': 3,     # Liquidation risks
    'whale_agent': 2,           # Whale movements
    'strategy_agent': 1,        # Trading strategies
    'swarm_agent': 1,           # Swarm consensus
    'rbi_agent': 1,             # RBI analysis
    'ml_prediction_agent': 1,   # ML predictions (high priority for trading decisions)
    'sniper_agent': 0,          # New launches (lower priority)
    'websearch_agent': 0,       # Background research
    'copybot_agent': 0,         # Copy trading
    'trading_agent': -1,        # Execute trades last
    'backtest_runner': -10,     # Manual only
}

# ============================================================================
# NEURO-FLUX ORCHESTRATION
# ============================================================================

class NeuroFluxOrchestrator:
    """Main orchestrator for NeuroFlux multi-agent system"""

    def __init__(self):
        self.agent_instances = {}
        self.last_run_times = {}
        self.agent_status = {}
        self.flux_state = {
            'level': config.FLUX_SENSITIVITY,
            'last_update': datetime.now(),
            'market_stability': 1.0
        }

        # Initialize Analytics Engine (Phase 4.3.6)
        self.analytics_engine = None
        if ANALYTICS_ENABLED and AnalyticsEngine:
            try:
                self.analytics_engine = AnalyticsEngine()
                cprint("📊 Analytics Engine initialized successfully", "green")
            except Exception as e:
                cprint(f"❌ Failed to initialize Analytics Engine: {e}", "red")
                self.analytics_engine = None

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all active agents"""
        cprint("🧠 Initializing NeuroFlux Agents...", "cyan")

        # Import agents individually to handle errors gracefully
        agent_imports = {
            'trading_agent': self._safe_import('agents.trading_agent'),
            'risk_agent': self._safe_import('agents.risk_agent'),
            'sentiment_agent': self._safe_import('agents.sentiment_agent'),
            'research_agent': self._safe_import('agents.research_agent'),
            'chartanalysis_agent': self._safe_import('agents.chartanalysis_agent'),
            'funding_agent': self._safe_import('agents.funding_agent'),
            'liquidation_agent': self._safe_import('agents.liquidation_agent'),
            'whale_agent': self._safe_import('agents.whale_agent'),
            'coingecko_agent': self._safe_import('agents.coingecko_agent'),
            'copybot_agent': self._safe_import('agents.copybot_agent'),
            'sniper_agent': self._safe_import('agents.sniper_agent'),
            'strategy_agent': self._safe_import('agents.strategy_agent'),
            'swarm_agent': self._safe_import('agents.swarm_agent'),
            'websearch_agent': self._safe_import('agents.websearch_agent'),
            'rbi_agent': self._safe_import('agents.rbi_agent'),
            'ml_prediction_agent': self._safe_import('agents.ml_prediction_agent'),
            'backtest_runner': self._safe_import('agents.backtest_runner'),
        }

        for agent_name, enabled in ACTIVE_AGENTS.items():
            if enabled:
                agent_module = agent_imports.get(agent_name)
                if agent_module:
                    self.agent_instances[agent_name] = agent_module
                    self.agent_status[agent_name] = 'initialized'
                    cprint(f"✅ {agent_name} initialized", "green")
                else:
                    cprint(f"❌ Failed to import {agent_name}", "red")
                    self.agent_status[agent_name] = 'failed'
            else:
                self.agent_status[agent_name] = 'disabled'

        cprint(f"🧠 NeuroFlux initialized with {len([s for s in self.agent_status.values() if s == 'initialized'])} active agents", "cyan")

    def _safe_import(self, module_name):
        """Safely import a module and return None if it fails"""
        try:
            import importlib
            return importlib.import_module(module_name)
        except Exception as e:
            cprint(f"⚠️  Could not import {module_name}: {e}", "yellow")
            return None

    def _update_flux_state(self):
        """Update neuro-flux state based on market conditions"""
        # This would integrate with real-time market data
        # For now, simulate flux adaptation
        current_time = datetime.now()
        time_since_update = (current_time - self.flux_state['last_update']).seconds

        if time_since_update > 300:  # Update every 5 minutes
            # Simulate flux changes based on "market conditions"
            flux_change = (time.time() % 100) / 1000  # Small random changes
            self.flux_state['level'] = max(0, min(1, self.flux_state['level'] + flux_change))
            self.flux_state['last_update'] = current_time
            self.flux_state['market_stability'] = 1 - self.flux_state['level']

            cprint(f"🌊 Flux updated: {self.flux_state['level']:.2f} | Stability: {self.flux_state['market_stability']:.2f}", "yellow")

    def _should_run_agent(self, agent_name: str) -> bool:
        """Determine if an agent should run based on flux conditions and timing"""
        if agent_name not in ACTIVE_AGENTS or not ACTIVE_AGENTS[agent_name]:
            return False

        # Risk agent always runs
        if agent_name == 'risk_agent':
            return True

        # In high flux, reduce frequency of some agents
        if self.flux_state['level'] > 0.7:
            # Reduce frequency of research and analysis agents in high flux
            if agent_name in ['research_agent', 'websearch_agent', 'coingecko_agent']:
                last_run = self.last_run_times.get(agent_name)
                if last_run and (datetime.now() - last_run).seconds < 1800:  # 30 min
                    return False

        # Trading agent only runs when conditions are favorable
        if agent_name == 'trading_agent' and self.flux_state['market_stability'] < 0.3:
            return False

        return True

    def _run_agent(self, agent_name: str):
        """Run a single agent iteration with error handling"""
        if agent_name not in self.agent_instances:
            return

        try:
            cprint(f"🤖 Running {agent_name}...", "blue")
            start_time = time.time()

            # Get the agent module
            agent_module = self.agent_instances[agent_name]

            # Call the appropriate function for each agent
            result = None
            if agent_name == 'swarm_agent' and hasattr(agent_module, 'run'):
                # Swarm agent has a special run method
                swarm_instance = agent_module.SwarmIntelligence()
                result = swarm_instance.run()
            elif hasattr(agent_module, 'main'):
                # For agents with infinite loops, run a single iteration
                result = self._run_agent_single_iteration(agent_module, agent_name)
            else:
                cprint(f"⚠️  {agent_name} has no main() function", "yellow")
                return None

            execution_time = time.time() - start_time
            self.last_run_times[agent_name] = datetime.now()

            cprint(f"✅ {agent_name} completed in {execution_time:.1f}s", "green")

            # Phase 4.3.6: Collect analytics data after agent execution
            if self.analytics_engine:
                try:
                    self._collect_agent_analytics(agent_name, result, execution_time)
                except Exception as e:
                    cprint(f"⚠️  Analytics collection failed for {agent_name}: {e}", "yellow")

            return result

        except Exception as e:
            cprint(f"❌ {agent_name} failed: {str(e)}", "red")
            traceback.print_exc()

            # Phase 4.3.6: Collect error analytics even on failure
            if self.analytics_engine:
                try:
                    self._collect_agent_error_analytics(agent_name, str(e))
                except Exception as analytics_error:
                    cprint(f"⚠️  Error analytics collection failed: {analytics_error}", "yellow")

            return None

    def _run_agent_single_iteration(self, agent_module, agent_name: str):
        """Run a single iteration of an agent that normally runs in a loop"""
        # Extract the core logic from each agent's main loop
        if agent_name == 'risk_agent':
            return self._run_risk_agent_iteration(agent_module)
        elif agent_name == 'sentiment_agent':
            return self._run_sentiment_agent_iteration(agent_module)
        elif agent_name == 'trading_agent':
            return self._run_trading_agent_iteration(agent_module)
        elif agent_name == 'research_agent':
            return self._run_research_agent_iteration(agent_module)
        elif agent_name == 'chartanalysis_agent':
            return self._run_chartanalysis_agent_iteration(agent_module)
        elif agent_name == 'funding_agent':
            return self._run_funding_agent_iteration(agent_module)
        elif agent_name == 'liquidation_agent':
            return self._run_liquidation_agent_iteration(agent_module)
        elif agent_name == 'whale_agent':
            return self._run_whale_agent_iteration(agent_module)
        elif agent_name == 'coingecko_agent':
            return self._run_coingecko_agent_iteration(agent_module)
        elif agent_name == 'sniper_agent':
            return self._run_sniper_agent_iteration(agent_module)
        elif agent_name == 'strategy_agent':
            return self._run_strategy_agent_iteration(agent_module)
        elif agent_name == 'swarm_agent':
            return self._run_swarm_agent_iteration(agent_module)
        elif agent_name == 'websearch_agent':
            return self._run_websearch_agent_iteration(agent_module)
        elif agent_name == 'rbi_agent':
            return self._run_rbi_agent_iteration(agent_module)
        else:
            # For unknown agents, try to run main() in a separate process with timeout
            import subprocess
            import sys
            try:
                # Run the agent in a subprocess with timeout
                result = subprocess.run([
                    sys.executable, '-c',
                    f'import sys; sys.path.insert(0, "src"); import agents.{agent_name} as agent; agent.main()'
                ], timeout=10, capture_output=True, text=True)
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                cprint(f"⏰ {agent_name} timed out", "yellow")
                return None

    def _run_risk_agent_iteration(self, agent_module):
        """Run a single iteration of the risk agent"""
        try:
            # Import required functions from the agent module
            calculate_flux_level = agent_module.calculate_flux_level
            get_portfolio_balance = agent_module.get_portfolio_balance
            get_positions = agent_module.get_positions
            check_risk_limits = agent_module.check_risk_limits
            save_risk_report = agent_module.save_risk_report

            # Calculate current market flux
            flux_level = calculate_flux_level()
            cprint(f"🌊 Market flux level: {flux_level:.3f}", "blue")

            # Get portfolio data
            balance = get_portfolio_balance()
            positions = get_positions()

            cprint(f"💰 Balance: ${balance['equity']:.2f} | Positions: {len(positions)}", "white")

            # Perform risk checks
            risk_results = check_risk_limits(balance, positions, flux_level)

            # Save report
            save_risk_report(risk_results, balance, positions, flux_level)

            # Handle violations (simplified for single run)
            if not risk_results['ok']:
                cprint("⚠️  Risk violations detected!", "red", attrs=['bold'])
                for violation in risk_results['violations']:
                    cprint(f"🚨 {violation['message']}", "red")

            if risk_results['ok']:
                cprint("✅ Risk checks passed - trading allowed", "green")
                return True
            else:
                cprint("❌ Risk checks failed - trading blocked", "red")
                return False

        except Exception as e:
            cprint(f"❌ Risk agent iteration error: {e}", "red")
            return None

    def _run_sentiment_agent_iteration(self, agent_module):
        """Run a single iteration of the sentiment agent"""
        try:
            # Import required functions
            get_active_tokens = agent_module.get_active_tokens
            aggregate_sentiment = agent_module.aggregate_sentiment
            display_sentiment = agent_module.display_sentiment
            save_sentiment_analysis = agent_module.save_sentiment_analysis

            # Get tokens to analyze
            tokens = get_active_tokens()
            if not tokens:
                tokens = ['BTC', 'ETH', 'SOL']  # Default tokens for testing

            cprint(f"📊 Analyzing sentiment for {len(tokens)} tokens", "blue")

            results = []
            for token in tokens:
                # Calculate current flux level
                flux_level = agent_module.FLUX_SENSITIVITY  # Placeholder

                # Perform sentiment analysis
                result = aggregate_sentiment(token, flux_level)
                results.append(result)

                # Display results
                display_sentiment(result)

                # Save results
                save_sentiment_analysis(result)

            cprint(f"✅ Sentiment analysis complete for {len(tokens)} tokens", "green")
            return results

        except Exception as e:
            cprint(f"❌ Sentiment agent iteration error: {e}", "red")
            return None

    def _run_trading_agent_iteration(self, agent_module):
        """Run a single iteration of the trading agent"""
        try:
            # Import required functions
            get_active_tokens = agent_module.get_active_tokens
            get_market_data = agent_module.get_market_data
            analyze_market_neuro = agent_module.analyze_market_neuro
            execute_trade = agent_module.execute_trade

            # Get tokens to monitor
            tokens = get_active_tokens()
            cprint(f"📊 Trading agent monitoring {len(tokens)} tokens", "blue")

            for token in tokens[:2]:  # Limit to 2 tokens for single run
                # Get market data
                data = get_market_data(token)
                cprint(f"📈 {token}: ${data['price']:.4f}", "white")

                # Neuro-enhanced analysis
                analysis = analyze_market_neuro(data)

                # Determine trade amount
                trade_amount = min(agent_module.usd_size, agent_module.max_usd_order_size)

                # Execute trade if signal (simulated)
                result = execute_trade(analysis['signal'], token, trade_amount, analysis)

            return True

        except Exception as e:
            cprint(f"❌ Trading agent iteration error: {e}", "red")
            return None

    def _run_research_agent_iteration(self, agent_module):
        """Run a single iteration of the research agent"""
        try:
            # Import required functions
            gather_market_data = agent_module.gather_market_data

            # Gather market data
            market_data = gather_market_data()
            cprint(f"🔬 Research agent gathered data for {len(market_data)} categories", "blue")

            return market_data

        except Exception as e:
            cprint(f"❌ Research agent iteration error: {e}", "red")
            return None

    def _run_chartanalysis_agent_iteration(self, agent_module):
        """Run a single iteration of the chart analysis agent"""
        try:
            cprint("📊 Chart analysis agent running single iteration", "blue")
            # Placeholder - would analyze charts
            return {"status": "analyzed", "charts": 3}

        except Exception as e:
            cprint(f"❌ Chart analysis agent iteration error: {e}", "red")
            return None

    def _run_funding_agent_iteration(self, agent_module):
        """Run a single iteration of the funding agent"""
        try:
            cprint("💰 Funding agent checking rates", "blue")
            # Placeholder - would check funding rates
            return {"status": "checked", "pairs": 5}

        except Exception as e:
            cprint(f"❌ Funding agent iteration error: {e}", "red")
            return None

    def _run_liquidation_agent_iteration(self, agent_module):
        """Run a single iteration of the liquidation agent"""
        try:
            cprint("💥 Liquidation agent monitoring", "blue")
            # Placeholder - would monitor liquidations
            return {"status": "monitored", "alerts": 0}

        except Exception as e:
            cprint(f"❌ Liquidation agent iteration error: {e}", "red")
            return None

    def _run_whale_agent_iteration(self, agent_module):
        """Run a single iteration of the whale agent"""
        try:
            cprint("🐋 Whale agent tracking movements", "blue")
            # Placeholder - would track whale movements
            return {"status": "tracked", "whales": 2}

        except Exception as e:
            cprint(f"❌ Whale agent iteration error: {e}", "red")
            return None

    def _run_coingecko_agent_iteration(self, agent_module):
        """Run a single iteration of the coingecko agent"""
        try:
            cprint("🪙 CoinGecko agent fetching data", "blue")
            # Placeholder - would fetch CoinGecko data
            return {"status": "fetched", "coins": 10}

        except Exception as e:
            cprint(f"❌ CoinGecko agent iteration error: {e}", "red")
            return None

    def _run_sniper_agent_iteration(self, agent_module):
        """Run a single iteration of the sniper agent"""
        try:
            cprint("🎯 Sniper agent monitoring launches", "blue")
            # Placeholder - would monitor new launches
            return {"status": "monitored", "launches": 0}

        except Exception as e:
            cprint(f"❌ Sniper agent iteration error: {e}", "red")
            return None

    def _run_strategy_agent_iteration(self, agent_module):
        """Run a single iteration of the strategy agent"""
        try:
            cprint("🎲 Strategy agent analyzing signals", "blue")
            # Placeholder - would analyze strategies
            return {"status": "analyzed", "strategies": 3}

        except Exception as e:
            cprint(f"❌ Strategy agent iteration error: {e}", "red")
            return None

    def _run_swarm_agent_iteration(self, agent_module):
        """Run a single iteration of the swarm agent"""
        try:
            cprint("🐝 Swarm agent consensus building", "blue")
            # Placeholder - would run swarm intelligence
            return {"status": "consensus", "agents": 6}

        except Exception as e:
            cprint(f"❌ Swarm agent iteration error: {e}", "red")
            return None

    def _run_websearch_agent_iteration(self, agent_module):
        """Run a single iteration of the websearch agent"""
        try:
            cprint("🌐 Web search agent gathering data", "blue")
            # Placeholder - would search web
            return {"status": "searched", "results": 5}

        except Exception as e:
            cprint(f"❌ Web search agent iteration error: {e}", "red")
            return None

    def _run_rbi_agent_iteration(self, agent_module):
        """Run a single iteration of the RBI agent"""
        try:
            cprint("🧠 RBI agent analyzing strategies", "blue")
            # Placeholder - would run RBI analysis
            return {"status": "analyzed", "strategies": 2}

        except Exception as e:
            cprint(f"❌ RBI agent iteration error: {e}", "red")
            return None

    def run_cycle(self):
        """Run one complete cycle of all active agents"""
        cycle_start_time = time.time()
        cycle_timestamp = datetime.now()

        cprint(f"\n🧠 NeuroFlux Cycle Starting - {cycle_timestamp.strftime('%Y-%m-%d %H:%M:%S')}", "cyan", attrs=['bold'])

        # Update flux state
        self._update_flux_state()

        # Sort agents by priority
        prioritized_agents = sorted(
            [name for name in ACTIVE_AGENTS.keys() if ACTIVE_AGENTS[name]],
            key=lambda x: AGENT_PRIORITIES.get(x, 0),
            reverse=True
        )

        results = {}
        cycle_errors = []

        for agent_name in prioritized_agents:
            if self._should_run_agent(agent_name):
                result = self._run_agent(agent_name)
                results[agent_name] = result

                # Track errors for analytics
                if result is None:
                    cycle_errors.append(agent_name)

                # Small delay between agents to prevent overwhelming
                time.sleep(0.5)
            else:
                cprint(f"⏭️  Skipping {agent_name} (conditions not met)", "yellow")
                results[agent_name] = 'skipped'

        cycle_execution_time = time.time() - cycle_start_time

        cprint(f"🧠 NeuroFlux Cycle Complete - {len(results)} agents executed", "cyan", attrs=['bold'])

        # Phase 4.3.6: Collect cycle-level analytics
        if self.analytics_engine:
            try:
                self._collect_cycle_analytics(results, cycle_errors, cycle_execution_time, cycle_timestamp)
            except Exception as e:
                cprint(f"⚠️  Cycle analytics collection failed: {e}", "yellow")

        return results

    def run_continuous(self, cycles: Optional[int] = None):
        """Run continuous cycles with sleep between them"""
        cycle_count = 0

        cprint("🚀 NeuroFlux Continuous Mode Started", "green", attrs=['bold'])
        cprint(f"🌊 Initial Flux Level: {self.flux_state['level']:.2f}", "yellow")
        cprint(f"⏰ Sleep between cycles: {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "cyan")

        try:
            while cycles is None or cycle_count < cycles:
                self.run_cycle()
                cycle_count += 1

                if cycles is None or cycle_count < cycles:
                    sleep_time = config.SLEEP_BETWEEN_RUNS_MINUTES * 60
                    cprint(f"😴 Sleeping for {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes...", "yellow")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            cprint("\n🛑 NeuroFlux stopped by user", "red", attrs=['bold'])

        except Exception as e:
            cprint(f"\n❌ NeuroFlux crashed: {e}", "red", attrs=['bold'])
            traceback.print_exc()

        cprint(f"🏁 NeuroFlux finished after {cycle_count} cycles", "green", attrs=['bold'])

    def get_status(self) -> Dict:
        """Get current system status"""
        status = {
            'flux_state': self.flux_state,
            'agent_status': self.agent_status,
            'last_run_times': {k: v.isoformat() if v else None for k, v in self.last_run_times.items()},
            'active_agents': [name for name, enabled in ACTIVE_AGENTS.items() if enabled],
            'total_agents': len(ACTIVE_AGENTS),
            'initialized_agents': len([s for s in self.agent_status.values() if s == 'initialized'])
        }

        # Phase 4.3.6: Add analytics status
        if self.analytics_engine:
            status['analytics'] = {
                'enabled': True,
                'status': 'active'
            }
        else:
            status['analytics'] = {
                'enabled': False,
                'status': 'unavailable'
            }

        return status

    def _collect_agent_analytics(self, agent_name: str, result: any, execution_time: float):
        """Phase 4.3.6: Collect analytics data after successful agent execution"""
        if not self.analytics_engine:
            return

        try:
            # Prepare agent execution data for analytics
            agent_data = {
                'agent_name': agent_name,
                'timestamp': datetime.now().isoformat(),
                'execution_time': execution_time,
                'success': result is not None,
                'result_summary': str(result)[:500] if result else None,  # Limit result size
                'flux_level': self.flux_state['level']
            }

            # Store in analytics system (this would be enhanced in future phases)
            # For now, just log that analytics collection occurred
            cprint(f"📊 Analytics collected for {agent_name}", "cyan")

        except Exception as e:
            cprint(f"⚠️  Failed to collect analytics for {agent_name}: {e}", "yellow")

    def _collect_agent_error_analytics(self, agent_name: str, error_message: str):
        """Phase 4.3.6: Collect analytics data for agent errors"""
        if not self.analytics_engine:
            return

        try:
            # Prepare error data for analytics
            error_data = {
                'agent_name': agent_name,
                'timestamp': datetime.now().isoformat(),
                'error_type': 'execution_error',
                'error_message': error_message[:500],  # Limit error message size
                'flux_level': self.flux_state['level']
            }

            # Store error analytics
            cprint(f"📊 Error analytics collected for {agent_name}", "yellow")

        except Exception as e:
            cprint(f"⚠️  Failed to collect error analytics: {e}", "yellow")

    def _collect_cycle_analytics(self, results: Dict, errors: List[str], execution_time: float, timestamp: datetime):
        """Phase 4.3.6: Collect analytics data for complete cycle execution"""
        if not self.analytics_engine:
            return

        try:
            # Prepare cycle analytics data
            cycle_data = {
                'cycle_timestamp': timestamp.isoformat(),
                'execution_time': execution_time,
                'agents_executed': len([r for r in results.values() if r != 'skipped']),
                'agents_skipped': len([r for r in results.values() if r == 'skipped']),
                'agents_failed': len(errors),
                'success_rate': (len(results) - len(errors)) / len(results) if results else 0,
                'flux_level': self.flux_state['level'],
                'market_stability': self.flux_state['market_stability'],
                'failed_agents': errors
            }

            # Store cycle analytics
            cprint(f"📊 Cycle analytics collected - Success rate: {cycle_data['success_rate']:.1%}", "cyan")

        except Exception as e:
            cprint(f"⚠️  Failed to collect cycle analytics: {e}", "yellow")

    def get_system_overview(self) -> Dict:
        """Phase 4.3.6: Get comprehensive system overview with analytics"""
        if self.analytics_engine:
            try:
                return self.analytics_engine.generate_system_overview()
            except Exception as e:
                cprint(f"⚠️  Analytics overview failed: {e}", "yellow")

        # Fallback to basic status if analytics unavailable
        return self.get_status()

    def get_agent_report(self, agent_name: str, hours_back: int = 24) -> Dict:
        """Phase 4.3.6: Get detailed agent report with analytics"""
        if self.analytics_engine:
            try:
                return self.analytics_engine.generate_agent_report(agent_name, hours_back)
            except Exception as e:
                cprint(f"⚠️  Agent analytics report failed for {agent_name}: {e}", "yellow")

        # Fallback to basic agent info
        return {
            'agent_name': agent_name,
            'status': self.agent_status.get(agent_name, 'unknown'),
            'last_run': self.last_run_times.get(agent_name).isoformat() if self.last_run_times.get(agent_name) else None,
            'analytics_available': False
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    cprint("🧠 NeuroFlux Multi-Agent Trading System", "cyan", attrs=['bold'])
    cprint("Built with love by Nyros Veil 🚀", "yellow")
    cprint("=" * 50, "white")

    # Check if agents are enabled
    if not AGENTS_ENABLED:
        cprint("❌ All agents are disabled in configuration", "red")
        return

    # Initialize orchestrator
    orchestrator = NeuroFluxOrchestrator()

    # Display status
    status = orchestrator.get_status()
    cprint(f"🌊 Flux Level: {status['flux_state']['level']:.2f}", "yellow")
    cprint(f"🤖 Active Agents: {status['initialized_agents']}/{status['total_agents']}", "cyan")
    cprint(f"📋 Agents: {', '.join(status['active_agents'])}", "white")

    # Run mode selection
    import argparse
    parser = argparse.ArgumentParser(description='NeuroFlux Multi-Agent Trading System')
    parser.add_argument('--cycles', type=int, help='Number of cycles to run (default: continuous)')
    parser.add_argument('--single', action='store_true', help='Run single cycle only')
    parser.add_argument('--status', action='store_true', help='Show status only')

    args = parser.parse_args()

    if args.status:
        # Just show status
        print(json.dumps(status, indent=2, default=str))
        return

    if args.single:
        # Run single cycle
        orchestrator.run_cycle()
    else:
        # Run continuous
        cycles = args.cycles if args.cycles else None
        orchestrator.run_continuous(cycles)


if __name__ == "__main__":
    main()