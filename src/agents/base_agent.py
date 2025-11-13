"""
🧠 NeuroFlux Base Agent Framework
Foundation for all NeuroFlux trading and research agents.

Built with love by Nyros Veil 🚀

Features:
- Neuro-flux enhanced LLM integration
- Standardized agent lifecycle management
- Performance monitoring and metrics
- Error handling and recovery
- Configuration management
- Multi-exchange support foundation
"""

import os
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv

from ..models.model_factory import ModelFactory

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent lifecycle status enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class AgentPriority(Enum):
    """Agent execution priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentMetrics:
    """Performance metrics for agent monitoring."""
    start_time: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    average_response_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    flux_level: float = 0.5
    model_provider: str = "unknown"

    def update_response_time(self, response_time: float):
        """Update average response time using exponential moving average."""
        alpha = 0.1  # Smoothing factor
        self.average_response_time = alpha * response_time + (1 - alpha) * self.average_response_time

    def record_request(self, success: bool, tokens_used: int = 0, response_time: float = 0.0):
        """Record a request and update metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.error_count += 1

        self.total_tokens_used += tokens_used
        self.update_response_time(response_time)

    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'total_tokens_used': self.total_tokens_used,
            'average_response_time': round(self.average_response_time, 3),
            'error_count': self.error_count,
            'last_error': self.last_error,
            'flux_level': self.flux_level,
            'model_provider': self.model_provider,
            'success_rate': round(self.get_success_rate(), 2),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

class BaseAgent(ABC):
    """
    Abstract base class for all NeuroFlux agents.

    Provides common functionality:
    - Neuro-flux enhanced LLM integration
    - Lifecycle management
    - Performance monitoring
    - Error handling and recovery
    - Configuration management
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent instance
            config: Agent-specific configuration
            **kwargs: Additional initialization parameters
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.kwargs = kwargs

        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.priority = AgentPriority.MEDIUM
        self.is_running = False
        self.should_stop = False

        # Neuro-flux parameters
        self.flux_level = kwargs.get('flux_level', 0.5)
        self.flux_sensitivity = kwargs.get('flux_sensitivity', 0.8)
        self.adaptive_behavior = kwargs.get('adaptive_behavior', True)

        # LLM configuration
        self.model_provider = kwargs.get('model_provider', 'claude')
        self.model_name = kwargs.get('model_name', None)
        self.llm_model = None

        # Performance monitoring
        self.metrics = AgentMetrics()
        self.metrics.flux_level = self.flux_level
        self.metrics.model_provider = self.model_provider

        # Threading and synchronization
        self._lock = threading.RLock()
        self._status_callbacks: List[Callable[[AgentStatus], None]] = []

        # Data directories
        self.data_dir = f"src/data/{self.__class__.__name__.lower()}"
        os.makedirs(self.data_dir, exist_ok=True)

        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")

        logger.info(f"🧠 Initializing {self.__class__.__name__} agent: {agent_id}")

    def initialize(self) -> bool:
        """
        Initialize the agent and its dependencies.

        Returns:
            bool: True if initialization successful
        """
        try:
            with self._lock:
                if self.status != AgentStatus.INITIALIZING:
                    logger.warning(f"Agent {self.agent_id} already initialized")
                    return True

                # Initialize LLM model
                self._initialize_llm_model()

                # Agent-specific initialization
                if not self._initialize_agent():
                    logger.error(f"Agent-specific initialization failed for {self.agent_id}")
                    self.status = AgentStatus.ERROR
                    return False

                self.status = AgentStatus.READY
                self._notify_status_change()
                logger.info(f"✅ Agent {self.agent_id} initialized successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            self.metrics.last_error = str(e)
            return False

    def _initialize_llm_model(self):
        """Initialize the LLM model with neuro-flux enhancements."""
        try:
            self.llm_model = ModelFactory.create_flux_optimized_model(
                self.model_provider,
                self.model_name,
                flux_level=self.flux_level
            )
            logger.info(f"🧠 Initialized {self.model_provider} model for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM model: {e}")
            raise

    @abstractmethod
    def _initialize_agent(self) -> bool:
        """
        Agent-specific initialization logic.

        Returns:
            bool: True if initialization successful
        """
        pass

    def start(self) -> bool:
        """
        Start the agent execution.

        Returns:
            bool: True if started successfully
        """
        try:
            with self._lock:
                if self.status != AgentStatus.READY:
                    logger.error(f"Cannot start agent {self.agent_id} in status {self.status.value}")
                    return False

                self.status = AgentStatus.RUNNING
                self.is_running = True
                self.should_stop = False
                self._notify_status_change()

                # Start agent in background thread
                agent_thread = threading.Thread(
                    target=self._run_agent_loop,
                    name=f"{self.__class__.__name__}-{self.agent_id}",
                    daemon=True
                )
                agent_thread.start()

                logger.info(f"🚀 Agent {self.agent_id} started successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to start agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            self.metrics.last_error = str(e)
            return False

    def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the agent execution.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            bool: True if stopped successfully
        """
        try:
            with self._lock:
                if self.status in [AgentStatus.STOPPED, AgentStatus.ERROR]:
                    return True

                self.status = AgentStatus.STOPPING
                self.should_stop = True
                self._notify_status_change()

                # Wait for agent to stop
                start_time = time.time()
                while self.is_running and (time.time() - start_time) < timeout:
                    time.sleep(0.1)

                if self.is_running:
                    logger.warning(f"Agent {self.agent_id} did not stop gracefully within {timeout}s")

                self.status = AgentStatus.STOPPED
                self.is_running = False
                self._notify_status_change()

                # Cleanup
                self._cleanup_agent()

                logger.info(f"🛑 Agent {self.agent_id} stopped successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Error stopping agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            return False

    def pause(self) -> bool:
        """
        Pause agent execution.

        Returns:
            bool: True if paused successfully
        """
        with self._lock:
            if self.status != AgentStatus.RUNNING:
                return False

            self.status = AgentStatus.PAUSED
            self._notify_status_change()
            logger.info(f"⏸️ Agent {self.agent_id} paused")
            return True

    def resume(self) -> bool:
        """
        Resume agent execution.

        Returns:
            bool: True if resumed successfully
        """
        with self._lock:
            if self.status != AgentStatus.PAUSED:
                return False

            self.status = AgentStatus.RUNNING
            self._notify_status_change()
            logger.info(f"▶️ Agent {self.agent_id} resumed")
            return True

    def _run_agent_loop(self):
        """Main agent execution loop."""
        try:
            self._on_agent_start()

            while not self.should_stop and self.is_running:
                try:
                    # Update flux level if adaptive
                    if self.adaptive_behavior:
                        self._update_flux_level()

                    # Execute agent logic
                    self._execute_agent_cycle()

                    # Sleep between cycles
                    time.sleep(self._get_cycle_interval())

                except Exception as e:
                    logger.error(f"Error in agent cycle: {e}")
                    self.metrics.record_request(False, response_time=0.0)
                    self.metrics.last_error = str(e)

                    # Implement backoff strategy
                    time.sleep(min(60.0, 2 ** self.metrics.error_count))

            self._on_agent_stop()

        except Exception as e:
            logger.error(f"Critical error in agent loop: {e}")
            self.status = AgentStatus.ERROR
            self.metrics.last_error = str(e)
        finally:
            self.is_running = False

    @abstractmethod
    def _execute_agent_cycle(self):
        """Execute one cycle of agent logic."""
        pass

    def _update_flux_level(self):
        """Update flux level based on market conditions or agent performance."""
        # Default implementation - can be overridden by subclasses
        # This could integrate with market data, performance metrics, etc.
        pass

    def _get_cycle_interval(self) -> float:
        """Get the interval between agent cycles in seconds."""
        return self.config.get('cycle_interval', 5.0)

    def _on_agent_start(self):
        """Called when agent starts running."""
        logger.info(f"Agent {self.agent_id} execution started")

    def _on_agent_stop(self):
        """Called when agent stops running."""
        logger.info(f"Agent {self.agent_id} execution stopped")

    @abstractmethod
    def _cleanup_agent(self):
        """Agent-specific cleanup logic."""
        pass

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the agent's LLM model with neuro-flux adaptation.

        Args:
            system_prompt: System instruction prompt
            user_content: User message content
            temperature: Optional temperature override
            **kwargs: Additional parameters

        Returns:
            Dict containing response data
        """
        if not self.llm_model:
            return {'error': True, 'error_message': 'LLM model not initialized'}

        try:
            start_time = time.time()

            # Apply neuro-flux adaptation
            adapted_temp = temperature
            if adapted_temp is None:
                adapted_temp = self.llm_model.apply_flux_adaptation(0.7, self.flux_level)

            # Generate response
            response = self.llm_model.generate_response(
                system_prompt,
                user_content,
                temperature=adapted_temp,
                flux_level=self.flux_level,
                **kwargs
            )

            response_time = time.time() - start_time
            tokens_used = response.get('tokens_used', 0)

            # Update metrics
            success = not response.get('error', False)
            self.metrics.record_request(success, tokens_used, response_time)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.metrics.record_request(False, response_time=time.time() - time.time())
            return {'error': True, 'error_message': str(e)}

    def update_flux_level(self, new_flux_level: float):
        """
        Update the agent's flux level and adapt behavior accordingly.

        Args:
            new_flux_level: New flux level (0-1)
        """
        self.flux_level = max(0.0, min(1.0, new_flux_level))
        self.metrics.flux_level = self.flux_level

        # Update LLM model if it supports flux adaptation
        if self.llm_model and hasattr(self.llm_model, 'apply_flux_adaptation'):
            logger.info(f"🌊 Agent {self.agent_id} flux level updated to {self.flux_level:.2f}")

    def add_status_callback(self, callback: Callable[[AgentStatus], None]):
        """Add a callback for status changes."""
        with self._lock:
            self._status_callbacks.append(callback)

    def remove_status_callback(self, callback: Callable[[AgentStatus], None]):
        """Remove a status callback."""
        with self._lock:
            self._status_callbacks.remove(callback)

    def _notify_status_change(self):
        """Notify all status callbacks of status change."""
        for callback in self._status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status information."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.__class__.__name__,
            'status': self.status.value,
            'priority': self.priority.value,
            'flux_level': self.flux_level,
            'model_provider': self.model_provider,
            'is_running': self.is_running,
            'metrics': self.metrics.to_dict(),
            'config': self.config,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds()
        }

    def save_state(self, filepath: Optional[str] = None) -> bool:
        """
        Save agent state to file.

        Args:
            filepath: Optional custom filepath

        Returns:
            bool: True if saved successfully
        """
        try:
            if not filepath:
                filepath = f"{self.data_dir}/agent_state_{self.agent_id}.json"

            state = {
                'agent_id': self.agent_id,
                'agent_type': self.__class__.__name__,
                'status': self.status.value,
                'config': self.config,
                'flux_level': self.flux_level,
                'metrics': self.metrics.to_dict(),
                'saved_at': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"💾 Agent state saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            return False

    def load_state(self, filepath: Optional[str] = None) -> bool:
        """
        Load agent state from file.

        Args:
            filepath: Optional custom filepath

        Returns:
            bool: True if loaded successfully
        """
        try:
            if not filepath:
                filepath = f"{self.data_dir}/agent_state_{self.agent_id}.json"

            if not os.path.exists(filepath):
                logger.warning(f"State file not found: {filepath}")
                return False

            with open(filepath, 'r') as f:
                state = json.load(f)

            # Restore state
            self.flux_level = state.get('flux_level', 0.5)
            self.metrics.flux_level = self.flux_level

            logger.info(f"📂 Agent state loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, status={self.status.value}, flux={self.flux_level:.2f})"