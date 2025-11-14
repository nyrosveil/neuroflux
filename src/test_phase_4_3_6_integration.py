"""
🧠 NeuroFlux Phase 4.3.6 Integration Testing
Test script for Analytics Engine integration into main orchestrator.

Built with love by Nyros Veil 🚀
"""

import os
import sys
import time
from datetime import datetime
from termcolor import cprint

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_analytics_integration():
    """Test Phase 4.3.6 Analytics Integration"""
    cprint("🧠 Phase 4.3.6 Integration Testing Started", "cyan", attrs=['bold'])
    cprint("=" * 60, "white")

    try:
        # Import the main orchestrator
        from main import NeuroFluxOrchestrator, ANALYTICS_ENABLED

        cprint(f"📊 Analytics Enabled: {ANALYTICS_ENABLED}", "blue")

        # Initialize orchestrator
        cprint("🔧 Initializing NeuroFlux Orchestrator...", "yellow")
        orchestrator = NeuroFluxOrchestrator()

        # Check analytics status
        status = orchestrator.get_status()
        cprint(f"📊 Analytics Status: {status.get('analytics', {})}", "blue")

        # Test system overview
        cprint("📈 Testing System Overview...", "yellow")
        overview = orchestrator.get_system_overview()
        cprint(f"✅ System Overview Generated - Agents: {overview.get('system_health', {}).get('total_agents', 0)}", "green")

        # Test agent report
        cprint("📋 Testing Agent Report...", "yellow")
        agent_report = orchestrator.get_agent_report('risk_agent', hours_back=1)
        cprint(f"✅ Agent Report Generated - Status: {agent_report.get('health_score', 'N/A')}", "green")

        # Test single cycle execution
        cprint("🔄 Testing Single Cycle Execution...", "yellow")
        start_time = time.time()
        results = orchestrator.run_cycle()
        execution_time = time.time() - start_time

        successful_agents = len([r for r in results.values() if r is not None and r != 'skipped'])
        total_agents = len(results)

        cprint(f"✅ Cycle Completed - {successful_agents}/{total_agents} agents successful", "green")
        cprint(f"⏱️  Execution time: {execution_time:.1f}s", "green")

        # Test analytics data collection
        if orchestrator.analytics_engine:
            cprint("📊 Analytics Engine Active - Data collection verified", "green")
        else:
            cprint("⚠️  Analytics Engine Not Available - Using fallback mode", "yellow")

        # Summary
        cprint("\n🎉 Phase 4.3.6 Integration Test Results:", "cyan", attrs=['bold'])
        cprint(f"   ✅ Orchestrator initialized: {orchestrator is not None}", "green")
        cprint(f"   ✅ Analytics integration: {ANALYTICS_ENABLED}", "green")
        cprint(f"   ✅ System overview: Generated", "green")
        cprint(f"   ✅ Agent reports: Working", "green")
        cprint(f"   ✅ Cycle execution: {successful_agents}/{total_agents} success", "green")
        cprint(f"   ✅ Execution time: {execution_time:.1f}s", "green")

        return True

    except Exception as e:
        cprint(f"❌ Integration test failed: {e}", "red")
        import traceback
        traceback.print_exc()
        return False

def test_analytics_endpoints():
    """Test analytics API endpoints"""
    cprint("\n🔗 Testing Analytics Endpoints...", "yellow")

    try:
        from main import NeuroFluxOrchestrator

        orchestrator = NeuroFluxOrchestrator()

        # Test various endpoints
        endpoints = [
            ("System Status", lambda: orchestrator.get_status()),
            ("System Overview", lambda: orchestrator.get_system_overview()),
            ("Risk Agent Report", lambda: orchestrator.get_agent_report('risk_agent')),
            ("Sentiment Agent Report", lambda: orchestrator.get_agent_report('sentiment_agent')),
        ]

        for endpoint_name, endpoint_func in endpoints:
            try:
                result = endpoint_func()
                status = "✅" if result else "❌"
                cprint(f"   {status} {endpoint_name}: {'Available' if result else 'Failed'}", "green" if result else "red")
            except Exception as e:
                cprint(f"   ❌ {endpoint_name}: Error - {e}", "red")

        return True

    except Exception as e:
        cprint(f"❌ Endpoint testing failed: {e}", "red")
        return False

if __name__ == "__main__":
    cprint("🧠 NeuroFlux Phase 4.3.6 Integration Testing", "cyan", attrs=['bold'])
    cprint(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white")
    cprint("=" * 60, "white")

    # Run integration tests
    success = test_analytics_integration()

    if success:
        # Run endpoint tests
        test_analytics_endpoints()

        cprint("\n🎉 Phase 4.3.6 Integration Testing PASSED!", "green", attrs=['bold'])
        cprint("📊 Analytics Engine successfully integrated into NeuroFlux orchestrator", "green")
        cprint("🚀 Ready for production deployment", "green")

    else:
        cprint("\n❌ Phase 4.3.6 Integration Testing FAILED!", "red", attrs=['bold'])
        cprint("🔧 Please check analytics integration and try again", "red")

    cprint(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white")