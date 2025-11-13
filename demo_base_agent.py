#!/usr/bin/env python3
"""
🧠 NeuroFlux Base Agent Framework Demo
Demonstrates the BaseAgent framework with a simple example agent.

Built with love by Nyros Veil 🚀
"""

import time
import signal
import sys
from typing import Dict, Any
from termcolor import cprint

# Add the src directory to the path for imports
sys.path.insert(0, 'src')

try:
    from agents.base_agent import BaseAgent, AgentStatus
    from agents.example_agent import ExampleAgent
    from models.model_factory import ModelFactory
except ImportError as e:
    cprint(f"❌ Import error: {e}", "red")
    cprint("Please ensure you're running from the neuroflux directory", "yellow")
    sys.exit(1)

class AgentDemo:
    """Demo class for showcasing the Base Agent Framework."""

    def __init__(self):
        self.agent = None
        self.running = False

    def setup_agent(self):
        """Set up and initialize the example agent."""
        cprint("🚀 Setting up NeuroFlux Example Agent...", "cyan")

        # Create agent configuration
        config = {
            'cycle_interval': 5.0,  # Run every 5 seconds
            'max_iterations': 3,    # Stop after 3 cycles
            'task_message': 'Greetings from the NeuroFlux Base Agent Framework!'
        }

        # Create the agent with neuro-flux enhancements
        self.agent = ExampleAgent(
            agent_id="demo-agent-001",
            config=config,
            model_provider="claude",  # Will use Claude if available
            flux_level=0.6,           # Moderate flux level
            flux_sensitivity=0.8,     # High sensitivity to flux changes
            adaptive_behavior=True    # Enable adaptive behavior
        )

        # Initialize the agent
        if self.agent.initialize():
            cprint("✅ Agent initialized successfully!", "green")
            return True
        else:
            cprint("❌ Agent initialization failed!", "red")
            return False

    def display_agent_info(self):
        """Display comprehensive agent information."""
        if not self.agent:
            return

        cprint("\n📊 Agent Information:", "cyan")
        status = self.agent.get_status()

        print(f"  Agent ID: {status['agent_id']}")
        print(f"  Agent Type: {status['agent_type']}")
        print(f"  Status: {status['status']}")
        print(f"  Priority: {status['priority']}")
        print(f"  Flux Level: {status['flux_level']:.2f}")
        print(f"  Model Provider: {status['model_provider']}")
        print(f"  Uptime: {status['uptime']:.1f}s")

        metrics = status['metrics']
        print(f"  Requests: {metrics['total_requests']} total, {metrics['successful_requests']} successful")
        print(f"  Success Rate: {metrics['success_rate']:.1f}%")
        print(f"  Avg Response Time: {metrics['average_response_time']:.2f}s")

    def run_demo(self):
        """Run the agent demo."""
        if not self.setup_agent():
            return

        self.display_agent_info()

        cprint("\n🎯 Starting agent execution...", "cyan")

        # Start the agent
        if self.agent.start():
            cprint("✅ Agent started successfully!", "green")

            # Let the agent run for a bit
            cprint("⏳ Agent is running (will stop automatically after 3 cycles)...", "yellow")

            # Monitor agent status
            try:
                while self.agent.is_running:
                    time.sleep(1)

                    # Display periodic updates
                    if int(time.time()) % 10 == 0:  # Every 10 seconds
                        self.display_agent_info()

            except KeyboardInterrupt:
                cprint("\n⏹️  Received interrupt signal...", "yellow")

            # Stop the agent
            cprint("🛑 Stopping agent...", "cyan")
            self.agent.stop(timeout=5.0)

            cprint("✅ Agent stopped successfully!", "green")

        else:
            cprint("❌ Failed to start agent!", "red")

    def test_agent_features(self):
        """Test various agent features."""
        if not self.setup_agent():
            return

        cprint("\n🧪 Testing Agent Features:", "cyan")

        # Test status callbacks
        def status_callback(status: AgentStatus):
            cprint(f"📢 Status changed to: {status.value}", "blue")

        self.agent.add_status_callback(status_callback)

        # Test flux level updates
        cprint("🌊 Testing flux level adaptation...", "yellow")
        original_flux = self.agent.flux_level
        self.agent.update_flux_level(0.8)
        cprint(f"  Flux level changed: {original_flux:.2f} → {self.agent.flux_level:.2f}", "green")

        # Test pause/resume
        cprint("⏸️  Testing pause/resume functionality...", "yellow")
        if self.agent.pause():
            cprint("  Agent paused successfully", "green")
            time.sleep(2)
            if self.agent.resume():
                cprint("  Agent resumed successfully", "green")

        # Test state saving/loading
        cprint("💾 Testing state persistence...", "yellow")
        if self.agent.save_state():
            cprint("  State saved successfully", "green")

        # Clean up
        self.agent.stop()

def main():
    """Main demo function."""
    cprint("🧠 NeuroFlux Base Agent Framework Demo", "cyan", attrs=["bold"])
    cprint("=" * 50, "cyan")

    # Check if model factory is working
    cprint("🔍 Checking Model Factory...", "yellow")
    try:
        providers = ModelFactory.get_available_providers()
        cprint(f"✅ Available providers: {', '.join(providers)}", "green")

        # Test model creation (won't actually connect without API keys)
        cprint("🧪 Testing model creation...", "yellow")
        try:
            test_model = ModelFactory.create_model("claude")
            cprint(f"✅ Model created: {test_model.__class__.__name__}", "green")
        except Exception as e:
            cprint(f"⚠️  Model creation failed (expected without API keys): {e}", "yellow")

    except Exception as e:
        cprint(f"❌ Model Factory error: {e}", "red")
        return

    # Run the agent demo
    demo = AgentDemo()
    demo.run_demo()

    # Run feature tests
    cprint("\n🧪 Running Feature Tests...", "cyan")
    demo.test_agent_features()

    cprint("\n🎉 Demo completed successfully!", "green", attrs=["bold"])

if __name__ == "__main__":
    main()