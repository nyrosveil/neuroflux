"""
🧠 ML Integration Test
Test script to verify ML integration with NeuroFlux orchestrator.

Built with love by Nyros Veil 🚀
"""

import sys
import os
import asyncio
from termcolor import cprint

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

async def test_orchestrator_initialization():
    """Test if the orchestrator can initialize with ML agents."""
    try:
        cprint("🧠 Testing NeuroFlux Orchestrator initialization with ML integration", "cyan", attrs=['bold'])

        from src.neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32

        # Create orchestrator instance
        orchestrator = NeuroFluxOrchestratorV32()

        # Try to initialize
        cprint("📋 Initializing orchestrator...", "blue")
        await orchestrator.initialize()

        # Check system status
        status = orchestrator.get_system_status()
        cprint("✅ Orchestrator initialized successfully!", "green")
        cprint(f"🤖 Registered agents: {status['agents']['registered']}", "white")
        cprint(f"🧠 System running: {status['system']['running']}", "white")

        # Try to create a trading cycle
        cprint("📊 Creating trading cycle tasks...", "blue")
        tasks = await orchestrator.create_trading_cycle_tasks()
        cprint(f"✅ Created {len(tasks)} trading cycle tasks", "green")

        # Check for ML prediction tasks
        ml_tasks = [t for t in tasks if 'ml' in t.task_type.lower()]
        cprint(f"🧠 ML prediction tasks: {len(ml_tasks)}", "white")

        # Shutdown
        await orchestrator.shutdown()
        cprint("✅ Orchestrator shutdown complete", "green")

        return True

    except Exception as e:
        cprint(f"❌ Orchestrator test failed: {e}", "red")
        import traceback
        traceback.print_exc()
        return False

async def test_ml_agent_direct():
    """Test ML agent directly."""
    try:
        cprint("🧠 Testing ML Prediction Agent directly", "cyan", attrs=['bold'])

        # Import the agent
        from src.agents.ml_prediction_agent import MLPredictionAgent

        # Create a mock communication bus
        class MockCommBus:
            def __init__(self):
                self.messages = []

            async def send_message(self, message):
                self.messages.append(message)
                print(f"📤 Message sent: {message.payload.get('action', 'unknown')}")

        # Create agent
        agent = MLPredictionAgent("test_ml_agent", MockCommBus())

        # Initialize
        result = agent.initialize()
        if result:
            cprint("✅ ML Prediction Agent initialized", "green")
        else:
            raise Exception("ML Prediction Agent initialization failed")

        return True

    except Exception as e:
        cprint(f"❌ ML Agent test failed: {e}", "red")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    cprint("🧠 ML Integration Test Suite", "cyan", attrs=['bold'])
    cprint("=" * 40, "cyan")

    # Test 1: ML Agent direct test
    cprint("\n1️⃣ Testing ML Agent directly...", "yellow")
    agent_test = await test_ml_agent_direct()

    # Test 2: Orchestrator initialization
    cprint("\n2️⃣ Testing Orchestrator with ML integration...", "yellow")
    orchestrator_test = await test_orchestrator_initialization()

    # Summary
    cprint("\n📋 TEST RESULTS", "cyan", attrs=['bold'])
    cprint("=" * 20, "cyan")
    cprint(f"ML Agent Test: {'✅ PASS' if agent_test else '❌ FAIL'}", "green" if agent_test else "red")
    cprint(f"Orchestrator Test: {'✅ PASS' if orchestrator_test else '❌ FAIL'}", "green" if orchestrator_test else "red")

    if agent_test and orchestrator_test:
        cprint("\n🎉 ALL TESTS PASSED! ML integration is working!", "green", attrs=['bold'])
        return True
    else:
        cprint("\n⚠️ Some tests failed. Check the errors above.", "yellow")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)