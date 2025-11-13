"""
🧠 NeuroFlux Example Agent
Demonstration agent showing how to inherit from BaseAgent.

Built with love by Nyros Veil 🚀

This is a simple example agent that demonstrates the BaseAgent framework.
"""

from typing import Dict, Any, Optional
from .base_agent import BaseAgent

class ExampleAgent(BaseAgent):
    """
    Example agent demonstrating the BaseAgent framework.

    This agent performs simple periodic tasks and demonstrates:
    - Agent lifecycle management
    - Neuro-flux integration
    - Performance monitoring
    - Configuration handling
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Set default config for example agent
        if config is None:
            config = {
                'cycle_interval': 10.0,  # Run every 10 seconds
                'max_iterations': 5,     # Stop after 5 cycles
                'task_message': 'Hello from NeuroFlux Example Agent!'
            }

        super().__init__(agent_id, config, **kwargs)

        # Agent-specific state
        self.iteration_count = 0
        self.max_iterations = config.get('max_iterations', 5)

    def _initialize_agent(self) -> bool:
        """Initialize example agent specific components."""
        try:
            self.logger.info(f"🎯 Initializing Example Agent with config: {self.config}")

            # Example initialization - could be database connections, API clients, etc.
            self.task_message = self.config.get('task_message', 'Default task message')

            self.logger.info("✅ Example Agent initialization complete")
            return True

        except Exception as e:
            self.logger.error(f"❌ Example Agent initialization failed: {e}")
            return False

    def _execute_agent_cycle(self):
        """Execute one cycle of agent logic."""
        try:
            self.iteration_count += 1
            self.logger.info(f"🔄 Executing cycle {self.iteration_count}/{self.max_iterations}")

            # Example task: Generate a response using the LLM
            system_prompt = f"You are a helpful AI assistant running in NeuroFlux. Current flux level: {self.flux_level:.2f}"
            user_content = f"{self.task_message} This is iteration {self.iteration_count}."

            response = self.generate_response(system_prompt, user_content)

            if response.get('success'):
                self.logger.info(f"🤖 LLM Response: {response['response'][:100]}...")
            else:
                self.logger.warning(f"⚠️ LLM call failed: {response.get('error_message', 'Unknown error')}")

            # Check if we should stop
            if self.iteration_count >= self.max_iterations:
                self.logger.info(f"🎯 Reached maximum iterations ({self.max_iterations}), requesting stop")
                self.should_stop = True

        except Exception as e:
            self.logger.error(f"❌ Error in agent cycle: {e}")
            self.metrics.record_request(False)

    def _update_flux_level(self):
        """Update flux level based on agent performance."""
        # Simple example: increase flux if success rate is high
        success_rate = self.metrics.get_success_rate()
        if success_rate > 80:
            new_flux = min(0.8, self.flux_level + 0.1)
        elif success_rate < 50:
            new_flux = max(0.2, self.flux_level - 0.1)
        else:
            new_flux = self.flux_level

        if abs(new_flux - self.flux_level) > 0.01:
            self.update_flux_level(new_flux)

    def _cleanup_agent(self):
        """Cleanup agent-specific resources."""
        self.logger.info("🧹 Cleaning up Example Agent resources")
        # Example cleanup - close connections, save final state, etc.
        self.save_state()

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent-specific information."""
        return {
            **super().get_status(),
            'iteration_count': self.iteration_count,
            'max_iterations': self.max_iterations,
            'task_message': self.task_message,
            'completion_percentage': (self.iteration_count / self.max_iterations) * 100 if self.max_iterations > 0 else 0
        }