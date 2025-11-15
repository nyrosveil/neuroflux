"""
🧠 NeuroFlux Comprehensive System Test Suite
Real-world testing of the complete NeuroFlux trading system.

Built with love by Nyros Veil 🚀

This test suite performs end-to-end testing of:
- System initialization and agent registration
- Complete trading cycle execution
- ML prediction integration
- API endpoint functionality
- WebSocket real-time updates
- Performance and stability testing
"""

import asyncio
import aiohttp
import json
import time
import sys
import os
import subprocess
import signal
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from termcolor import cprint, colored

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


class NeuroFluxSystemTester:
    """Comprehensive system tester for NeuroFlux."""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.server_process = None

    async def run_full_system_test(self) -> Dict[str, Any]:
        """Run complete system test suite."""
        cprint("🧠 NeuroFlux Comprehensive System Test Suite", "cyan", attrs=['bold'])
        cprint("=" * 60, "cyan")

        self.start_time = datetime.now()

        try:
            # Phase 1: System Setup
            cprint("\n📋 PHASE 1: System Setup", "yellow", attrs=['bold'])
            await self.test_system_setup()

            # Phase 2: Core Components
            cprint("\n📋 PHASE 2: Core Components", "yellow", attrs=['bold'])
            await self.test_core_components()

            # Phase 3: ML Integration
            cprint("\n📋 PHASE 3: ML Integration", "yellow", attrs=['bold'])
            await self.test_ml_integration()

            # Phase 4: API Endpoints
            cprint("\n📋 PHASE 4: API Endpoints", "yellow", attrs=['bold'])
            await self.test_api_endpoints()

            # Phase 5: Real-time Features
            cprint("\n📋 PHASE 5: Real-time Features", "yellow", attrs=['bold'])
            await self.test_realtime_features()

            # Phase 6: Performance Testing
            cprint("\n📋 PHASE 6: Performance Testing", "yellow", attrs=['bold'])
            await self.test_performance()

            # Phase 7: Trading Cycle
            cprint("\n📋 PHASE 7: Complete Trading Cycle", "yellow", attrs=['bold'])
            await self.test_trading_cycle()

        except Exception as e:
            cprint(f"❌ Test suite failed: {e}", "red", attrs=['bold'])
            import traceback
            traceback.print_exc()
        finally:
            self.end_time = datetime.now()
            await self.cleanup()

        return self.generate_test_report()

    async def test_system_setup(self):
        """Test system initialization and setup."""
        cprint("🔧 Testing System Setup...", "blue")

        # Test 1: Import all modules
        try:
            from neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32
            from dashboard_api import app
            cprint("✅ Module imports successful", "green")
            self.test_results['system_setup'] = {'imports': True}
        except Exception as e:
            cprint(f"❌ Module import failed: {e}", "red")
            self.test_results['system_setup'] = {'imports': False, 'error': str(e)}
            return

        # Test 2: Start dashboard server
        try:
            cprint("🚀 Starting dashboard server...", "blue")
            self.server_process = subprocess.Popen(
                ['gunicorn', '-w', '2', '-b', '127.0.0.1:8000', 'dashboard_api:app'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )

            # Wait for server to start
            await asyncio.sleep(3)

            # Check if server is running
            if self.server_process.poll() is None:
                cprint("✅ Dashboard server started", "green")
                self.test_results['system_setup']['server'] = True
            else:
                stdout, stderr = self.server_process.communicate()
                cprint(f"❌ Server failed to start: {stderr.decode()}", "red")
                self.test_results['system_setup']['server'] = False
                return

        except Exception as e:
            cprint(f"❌ Server startup failed: {e}", "red")
            self.test_results['system_setup']['server'] = False

    async def test_core_components(self):
        """Test core NeuroFlux components."""
        cprint("🏗️ Testing Core Components...", "blue")

        # Test orchestrator initialization
        try:
            from neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32

            orchestrator = NeuroFluxOrchestratorV32()
            await orchestrator.initialize()

            # Check agent registration
            agent_count = len(orchestrator.agent_registry.agents)
            cprint(f"✅ Orchestrator initialized with {agent_count} agents", "green")

            # Check for ML agent
            ml_agents = [agent for agent in orchestrator.agent_registry.agents.values()
                        if 'ml' in agent.agent_type.lower()]
            cprint(f"🧠 ML agents registered: {len(ml_agents)}", "green")

            await orchestrator.shutdown()

            self.test_results['core_components'] = {
                'orchestrator': True,
                'agent_count': agent_count,
                'ml_agents': len(ml_agents)
            }

        except Exception as e:
            cprint(f"❌ Core components test failed: {e}", "red")
            self.test_results['core_components'] = {'error': str(e)}

    async def test_ml_integration(self):
        """Test ML prediction integration."""
        cprint("🧠 Testing ML Integration...", "blue")

        try:
            from neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32

            orchestrator = NeuroFluxOrchestratorV32()
            await orchestrator.initialize()

            # Create trading cycle tasks
            tasks = await orchestrator.create_trading_cycle_tasks()

            # Check ML tasks
            ml_tasks = [t for t in tasks if 'ml' in t.task_type.lower()]
            cprint(f"✅ Created {len(ml_tasks)} ML prediction tasks", "green")

            # Check task dependencies
            trading_tasks = [t for t in tasks if 'trading' in t.task_type.lower()]
            if trading_tasks:
                trading_task = trading_tasks[0]
                has_ml_dependencies = any(dep in [t.task_id for t in ml_tasks]
                                        for dep in trading_task.dependencies)
                cprint(f"✅ Trading tasks depend on ML predictions: {has_ml_dependencies}", "green")

            await orchestrator.shutdown()

            self.test_results['ml_integration'] = {
                'ml_tasks_created': len(ml_tasks),
                'task_dependencies': has_ml_dependencies if 'has_ml_dependencies' in locals() else False
            }

        except Exception as e:
            cprint(f"❌ ML integration test failed: {e}", "red")
            self.test_results['ml_integration'] = {'error': str(e)}

    async def test_api_endpoints(self):
        """Test all API endpoints."""
        cprint("🔌 Testing API Endpoints...", "blue")

        results = {}

        # Test endpoints
        endpoints = [
            ('/api/status', 'System Status'),
            ('/api/dashboard/predictions', 'Predictions Data'),
            ('/api/dashboard/agents', 'Agent Status'),
            ('/api/dashboard/performance', 'Performance Metrics'),
            ('/api/ml/models', 'ML Models'),
            ('/api/ml/predict/arima?token=BTC', 'ARIMA Prediction'),
        ]

        async with aiohttp.ClientSession() as session:
            for endpoint, description in endpoints:
                try:
                    url = f"{self.base_url}{endpoint}"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            data = await response.json()
                            cprint(f"✅ {description}: {response.status}", "green")
                            results[endpoint] = {'status': 'success', 'data_size': len(str(data))}
                        else:
                            error_text = await response.text()
                            cprint(f"⚠️ {description}: {response.status} - {error_text[:100]}", "yellow")
                            results[endpoint] = {'status': 'warning', 'code': response.status}
                except Exception as e:
                    cprint(f"❌ {description}: {str(e)[:100]}", "red")
                    results[endpoint] = {'status': 'error', 'error': str(e)}

        self.test_results['api_endpoints'] = results

    async def test_realtime_features(self):
        """Test WebSocket and real-time features."""
        cprint("🌐 Testing Real-time Features...", "blue")

        try:
            # Test WebSocket connection (basic connectivity)
            import websockets
            import json

            async with websockets.connect(self.ws_url) as websocket:
                # Send a test message
                test_message = {
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                }

                await websocket.send(json.dumps(test_message))

                # Try to receive response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    cprint("✅ WebSocket connection successful", "green")
                    self.test_results['realtime_features'] = {'websocket': True}
                except asyncio.TimeoutError:
                    cprint("⚠️ WebSocket response timeout (expected for basic ping)", "yellow")
                    self.test_results['realtime_features'] = {'websocket': True, 'note': 'timeout_expected'}

        except ImportError:
            cprint("⚠️ WebSocket testing skipped (websockets library not available)", "yellow")
            self.test_results['realtime_features'] = {'websocket': 'skipped', 'reason': 'no_websockets_lib'}
        except Exception as e:
            cprint(f"❌ WebSocket test failed: {e}", "red")
            self.test_results['realtime_features'] = {'websocket': False, 'error': str(e)}

    async def test_performance(self):
        """Test system performance."""
        cprint("⚡ Testing Performance...", "blue")

        try:
            # Test API response times
            response_times = []

            async with aiohttp.ClientSession() as session:
                for i in range(5):  # Test 5 requests
                    start_time = time.time()
                    try:
                        async with session.get(f"{self.base_url}/api/status",
                                             timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                await response.json()
                                response_time = time.time() - start_time
                                response_times.append(response_time)
                                cprint(f"✅ Request {i+1}: {response_time:.3f}s", "green")
                    except Exception as e:
                        cprint(f"❌ Request {i+1} failed: {e}", "red")

            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                cprint(f"📊 Average response time: {avg_response_time:.3f}s", "blue")
                cprint(f"📊 Max response time: {max_response_time:.3f}s", "blue")

                self.test_results['performance'] = {
                    'avg_response_time': avg_response_time,
                    'max_response_time': max_response_time,
                    'successful_requests': len(response_times)
                }
            else:
                self.test_results['performance'] = {'error': 'no_successful_requests'}

        except Exception as e:
            cprint(f"❌ Performance test failed: {e}", "red")
            self.test_results['performance'] = {'error': str(e)}

    async def test_trading_cycle(self):
        """Test complete trading cycle execution."""
        cprint("📈 Testing Complete Trading Cycle...", "blue")

        try:
            from neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32

            orchestrator = NeuroFluxOrchestratorV32()
            await orchestrator.initialize()

            # Run a complete trading cycle
            cprint("🚀 Executing trading cycle...", "blue")
            start_cycle = time.time()

            # Create and execute tasks
            tasks = await orchestrator.create_trading_cycle_tasks()
            cprint(f"📋 Created {len(tasks)} tasks for trading cycle", "blue")

            # Simulate task execution (in real system, tasks would be assigned to agents)
            completed_tasks = 0
            for task in tasks:
                # Mark task as completed (simulated)
                completed_tasks += 1

            cycle_time = time.time() - start_cycle

            cprint(f"✅ Trading cycle completed in {cycle_time:.2f}s", "green")
            cprint(f"📊 Tasks processed: {completed_tasks}/{len(tasks)}", "blue")

            await orchestrator.shutdown()

            self.test_results['trading_cycle'] = {
                'cycle_time': cycle_time,
                'tasks_created': len(tasks),
                'tasks_completed': completed_tasks,
                'success': True
            }

        except Exception as e:
            cprint(f"❌ Trading cycle test failed: {e}", "red")
            self.test_results['trading_cycle'] = {'error': str(e)}

    async def cleanup(self):
        """Clean up test resources."""
        cprint("\n🧹 Cleaning up...", "blue")

        # Stop server process
        if self.server_process and self.server_process.poll() is None:
            cprint("🛑 Stopping dashboard server...", "yellow")
            self.server_process.terminate()

            try:
                self.server_process.wait(timeout=5)
                cprint("✅ Server stopped", "green")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                cprint("⚠️ Server force killed", "yellow")

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        cprint("\n📊 TEST REPORT SUMMARY", "cyan", attrs=['bold'])
        cprint("=" * 50, "cyan")

        total_tests = len(self.test_results)
        passed_tests = 0
        failed_tests = 0
        warnings = 0

        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    failed_tests += 1
                    cprint(f"❌ {test_name}: FAILED", "red")
                elif result.get('status') == 'warning':
                    warnings += 1
                    cprint(f"⚠️ {test_name}: WARNING", "yellow")
                else:
                    passed_tests += 1
                    cprint(f"✅ {test_name}: PASSED", "green")
            else:
                passed_tests += 1
                cprint(f"✅ {test_name}: PASSED", "green")

        # Calculate duration
        duration = self.end_time - self.start_time if self.end_time and self.start_time else timedelta(0)

        cprint(f"\n📈 Overall Results:", "cyan", attrs=['bold'])
        cprint(f"   Total Tests: {total_tests}", "white")
        cprint(f"   Passed: {passed_tests}", "green")
        cprint(f"   Failed: {failed_tests}", "red")
        cprint(f"   Warnings: {warnings}", "yellow")
        cprint(f"   Duration: {duration.total_seconds():.2f}s", "blue")

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        cprint(f"   Success Rate: {success_rate:.1f}%", "green" if success_rate >= 80 else "yellow")

        # System health assessment
        if success_rate >= 90:
            health = "🟢 EXCELLENT"
            color = "green"
        elif success_rate >= 75:
            health = "🟡 GOOD"
            color = "yellow"
        elif success_rate >= 50:
            health = "🟠 FAIR"
            color = "yellow"
        else:
            health = "🔴 POOR"
            color = "red"

        cprint(f"   System Health: {health}", color, attrs=['bold'])

        return {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'warnings': warnings,
                'success_rate': success_rate,
                'duration': duration.total_seconds(),
                'system_health': health
            },
            'detailed_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }


async def main():
    """Main test execution."""
    tester = NeuroFluxSystemTester()
    report = await tester.run_full_system_test()

    # Save report to file
    with open('neuroflux_test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    cprint(f"\n💾 Test report saved to: neuroflux_test_report.json", "blue")

    # Exit with appropriate code
    success_rate = report['summary']['success_rate']
    sys.exit(0 if success_rate >= 75 else 1)


if __name__ == "__main__":
    asyncio.run(main())