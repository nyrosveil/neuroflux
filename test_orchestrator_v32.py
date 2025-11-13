#!/usr/bin/env python3
"""
🧪 Test Runner for NeuroFlux Orchestrator v3.2
Simple test to verify the new orchestration system works.

Built with love by Nyros Veil 🚀
"""

import sys
import os
import asyncio
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from neuroflux_orchestrator_v32 import NeuroFluxOrchestratorV32


async def test_orchestrator_initialization():
    """Test basic orchestrator initialization."""
    print("🧠 Testing NeuroFlux Orchestrator v3.2...")

    orchestrator = NeuroFluxOrchestratorV32()

    # Test initialization
    await orchestrator.initialize()
    print("✅ Orchestrator initialized successfully")

    # Get status
    status = orchestrator.get_system_status()
    print(f"📊 System Status: {status['agents']['registered']} agents registered")

    # Test single cycle
    print("🔄 Running single trading cycle...")
    cycle_result = await orchestrator.run_trading_cycle()
    print(f"✅ Cycle completed: {cycle_result['tasks_completed']}/{cycle_result['tasks_created']} tasks successful")

    # Shutdown
    await orchestrator.shutdown()
    print("✅ Orchestrator shutdown successfully")

    return True


async def main():
    """Run tests."""
    print("🚀 Starting NeuroFlux Orchestrator v3.2 Tests\n")

    try:
        success = await test_orchestrator_initialization()
        if success:
            print("\n🎊 ALL TESTS PASSED! NeuroFlux Orchestrator v3.2 is working!")
            return True
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)