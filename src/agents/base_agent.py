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

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import ModelFactory
try:
    from cache_system import neuroflux_cache, cached
except ImportError:
    # Fallback if cache system not available
    neuroflux_cache = None
    cached = lambda *args, **kwargs: lambda func: func

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
    """Enhanced performance metrics for agent monitoring."""
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

    # Enhanced metrics
    peak_memory_usage: float = 0.0
    cpu_time_used: float = 0.0
    cycle_count: int = 0
    last_cycle_time: float = 0.0
    average_cycle_time: float = 0.0
    flux_changes: int = 0
    recovery_events: int = 0
    alert_count: int = 0

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

        # Update enhanced metrics
        self._update_enhanced_metrics()

        # Check for alerts
        self._check_alerts()

    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0

    def _update_enhanced_metrics(self):
        """Update enhanced performance metrics."""
        try:
            import psutil
            import os

            # Update memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.peak_memory_usage = max(self.peak_memory_usage, memory_mb)

            # Update CPU time
            cpu_times = process.cpu_times()
            self.cpu_time_used = cpu_times.user + cpu_times.system

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Error updating enhanced metrics: {e}")

    def _check_alerts(self):
        """Check for alert conditions and trigger alerts."""
        alerts = []

        # Error rate alert
        if self.total_requests > 10:
            error_rate = self.failed_requests / self.total_requests
            if error_rate > 0.3:  # 30% error rate
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f'Error rate {error_rate:.1%} exceeds threshold',
                    'value': error_rate
                })

        # Response time alert
        if self.average_response_time > 30.0:  # 30 seconds
            alerts.append({
                'type': 'slow_response',
                'severity': 'warning',
                'message': f'Average response time {self.average_response_time:.1f}s too high',
                'value': self.average_response_time
            })

        # Memory usage alert
        if self.peak_memory_usage > 500:  # 500MB
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f'Peak memory usage {self.peak_memory_usage:.1f}MB exceeds threshold',
                'value': self.peak_memory_usage
            })

        # Trigger alerts
        for alert in alerts:
            self.alert_count += 1
            self._trigger_alert(alert)

    def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert (can be overridden by subclasses)."""
        severity = alert.get('severity', 'info')
        message = alert.get('message', 'Alert triggered')

        if severity == 'critical':
            logger.error(f"🚨 CRITICAL ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"⚠️ WARNING ALERT: {message}")
        else:
            logger.info(f"ℹ️ INFO ALERT: {message}")

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
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            # Enhanced metrics
            'peak_memory_usage': round(self.peak_memory_usage, 2),
            'cpu_time_used': round(self.cpu_time_used, 2),
            'cycle_count': self.cycle_count,
            'last_cycle_time': round(self.last_cycle_time, 3),
            'average_cycle_time': round(self.average_cycle_time, 3),
            'flux_changes': self.flux_changes,
            'recovery_events': self.recovery_events,
            'alert_count': self.alert_count
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
        Pause agent execution with validation.

        Returns:
            bool: True if paused successfully
        """
        with self._lock:
            if self.status not in [AgentStatus.RUNNING, AgentStatus.READY]:
                logger.warning(f"Cannot pause agent {self.agent_id} in status {self.status.value}")
                return False

            old_status = self.status
            self.status = AgentStatus.PAUSED
            self._notify_status_change()
            logger.info(f"⏸️ Agent {self.agent_id} paused from {old_status.value}")
            return True

    def resume(self) -> bool:
        """
        Resume agent execution with validation.

        Returns:
            bool: True if resumed successfully
        """
        with self._lock:
            if self.status != AgentStatus.PAUSED:
                logger.warning(f"Cannot resume agent {self.agent_id} in status {self.status.value}")
                return False

            self.status = AgentStatus.RUNNING
            self._notify_status_change()
            logger.info(f"▶️ Agent {self.agent_id} resumed")
            return True

    def restart(self) -> bool:
        """
        Restart the agent completely.

        Returns:
            bool: True if restarted successfully
        """
        logger.info(f"🔄 Restarting agent {self.agent_id}")

        # Stop current execution
        if not self.stop():
            logger.error(f"Failed to stop agent {self.agent_id} for restart")
            return False

        # Brief pause
        time.sleep(1.0)

        # Reinitialize
        if not self.initialize():
            logger.error(f"Failed to reinitialize agent {self.agent_id}")
            return False

        # Start again
        if not self.start():
            logger.error(f"Failed to restart agent {self.agent_id}")
            return False

        logger.info(f"✅ Agent {self.agent_id} restarted successfully")
        return True

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check.

        Returns:
            Dict containing health status
        """
        health = {
            'agent_id': self.agent_id,
            'status': self.status.value,
            'healthy': True,
            'issues': [],
            'checks': {}
        }

        # Status check
        health['checks']['status'] = {
            'healthy': self.status not in [AgentStatus.ERROR],
            'value': self.status.value
        }

        # Memory check
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            health['checks']['memory'] = {
                'healthy': memory_mb < 1000,  # 1GB threshold
                'value': round(memory_mb, 1)
            }
        except:
            health['checks']['memory'] = {'healthy': True, 'value': 'unknown'}

        # Performance check
        success_rate = self.metrics.get_success_rate()
        health['checks']['performance'] = {
            'healthy': success_rate > 80.0,
            'value': round(success_rate, 1)
        }

        # Error check
        recent_errors = len([e for e in getattr(self, '_error_history', [])
                           if time.time() - e['timestamp'] < 300])
        health['checks']['errors'] = {
            'healthy': recent_errors < 5,
            'value': recent_errors
        }

        # Overall health
        health['healthy'] = all(check['healthy'] for check in health['checks'].values())
        health['issues'] = [name for name, check in health['checks'].items() if not check['healthy']]

        return health

    def _run_agent_loop(self):
        """Main agent execution loop."""
        try:
            self._on_agent_start()

            while not self.should_stop and self.is_running:
                cycle_start = time.time()
                try:
                    # Update flux level if adaptive
                    if self.adaptive_behavior:
                        self._update_flux_level()

                    # Execute agent logic
                    self._execute_agent_cycle()

                    # Track cycle metrics
                    cycle_time = time.time() - cycle_start
                    self._record_cycle_metrics(cycle_time)

                    # Sleep between cycles
                    time.sleep(self._get_cycle_interval())

                except Exception as e:
                    error_msg = f"Error in agent cycle: {e}"
                    logger.error(error_msg)
                    self.metrics.record_request(False, response_time=0.0)
                    self.metrics.last_error = str(e)

                    # Enhanced error handling and recovery
                    self._handle_cycle_error(e)

                    # Implement adaptive backoff strategy
                    backoff_time = self._calculate_backoff_time()
                    logger.info(f"Backing off for {backoff_time:.1f} seconds")
                    time.sleep(backoff_time)

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
        """Advanced flux level adaptation based on multiple factors."""
        try:
            current_time = time.time()
            # Throttle updates to avoid excessive computation
            if hasattr(self, '_last_flux_update') and (current_time - self._last_flux_update) < 30.0:
                return

            self._last_flux_update = current_time

            # Calculate flux factors
            performance_factor = self._calculate_performance_factor()
            market_factor = self._calculate_market_factor()
            error_factor = self._calculate_error_factor()
            load_factor = self._calculate_load_factor()

            # Combine factors with weights
            weights = {
                'performance': 0.4,
                'market': 0.3,
                'error': 0.2,
                'load': 0.1
            }

            new_flux = (
                performance_factor * weights['performance'] +
                market_factor * weights['market'] +
                error_factor * weights['error'] +
                load_factor * weights['load']
            )

            # Apply smoothing to prevent drastic changes
            smoothing_factor = 0.1
            new_flux = (smoothing_factor * new_flux) + ((1 - smoothing_factor) * self.flux_level)

            # Clamp to valid range
            new_flux = max(0.0, min(1.0, new_flux))

            # Only update if change is significant
            if abs(new_flux - self.flux_level) > 0.05:
                old_flux = self.flux_level
                self.update_flux_level(new_flux)
                logger.info(f"🌊 Agent {self.agent_id} flux adapted: {old_flux:.2f} → {new_flux:.2f} "
                           f"(perf:{performance_factor:.2f}, market:{market_factor:.2f}, "
                           f"error:{error_factor:.2f}, load:{load_factor:.2f})")

        except Exception as e:
            logger.error(f"Error updating flux level: {e}")

    def _calculate_performance_factor(self) -> float:
        """Calculate flux factor based on agent performance."""
        if self.metrics.total_requests == 0:
            return 0.5  # Neutral

        success_rate = self.metrics.get_success_rate()
        avg_response_time = self.metrics.average_response_time

        # Higher success rate and lower response time = higher flux tolerance
        performance_score = (success_rate / 100.0) * 0.7 + (1.0 / (1.0 + avg_response_time)) * 0.3
        return performance_score

    def _calculate_market_factor(self) -> float:
        """Calculate flux factor based on market conditions."""
        # Default implementation - subclasses can override with actual market data
        # This could integrate with market volatility, trend strength, etc.
        return 0.5  # Neutral - subclasses should override

    def _calculate_error_factor(self) -> float:
        """Calculate flux factor based on error patterns."""
        if self.metrics.total_requests == 0:
            return 0.5

        error_rate = self.metrics.failed_requests / self.metrics.total_requests

        # Higher error rate = lower flux (more conservative)
        if error_rate > 0.2:
            return 0.2  # Very conservative
        elif error_rate > 0.1:
            return 0.4  # Conservative
        elif error_rate > 0.05:
            return 0.6  # Moderate
        else:
            return 0.8  # Aggressive

    def _calculate_load_factor(self) -> float:
        """Calculate flux factor based on system load."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            # High load = lower flux (reduce computational intensity)
            load_score = 1.0 - ((cpu_percent + memory_percent) / 200.0)
            return max(0.1, min(1.0, load_score))

        except ImportError:
            # psutil not available
            return 0.5
        except Exception as e:
            logger.warning(f"Error calculating load factor: {e}")
            return 0.5

    def _handle_cycle_error(self, error: Exception):
        """Handle errors that occur during agent cycles with recovery strategies."""
        error_type = type(error).__name__

        # Track error patterns
        if not hasattr(self, '_error_history'):
            self._error_history = []
        self._error_history.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': str(error)
        })

        # Keep only recent errors (last 10)
        self._error_history = self._error_history[-10:]

        # Analyze error patterns
        recent_errors = [e for e in self._error_history
                        if time.time() - e['timestamp'] < 300]  # Last 5 minutes

        # Implement recovery strategies based on error patterns
        if len(recent_errors) >= 3:
            # Multiple recent errors - implement recovery
            self.record_recovery_event()
            self._implement_error_recovery(error_type, recent_errors)

        # Log error details
        logger.warning(f"Agent {self.agent_id} cycle error: {error_type} - {error}")

    def _implement_error_recovery(self, error_type: str, recent_errors: list):
        """Implement specific recovery strategies based on error patterns."""
        try:
            if error_type in ['ConnectionError', 'TimeoutError']:
                # Network issues - reduce frequency and implement retry
                logger.warning(f"Network errors detected, reducing cycle frequency")
                self._temporary_reduce_frequency()

            elif error_type in ['RateLimitError', 'TooManyRequests']:
                # Rate limiting - increase delays
                logger.warning(f"Rate limiting detected, increasing delays")
                self._increase_delays()

            elif error_type in ['MemoryError', 'OSError']:
                # Resource issues - reduce flux and cleanup
                logger.warning(f"Resource issues detected, reducing flux and cleaning up")
                self.update_flux_level(max(0.1, self.flux_level * 0.8))
                self._emergency_cleanup()

            elif len(recent_errors) >= 5:
                # Persistent errors - pause agent temporarily
                logger.error(f"Persistent errors detected, pausing agent temporarily")
                self._temporary_pause()

        except Exception as e:
            logger.error(f"Error in recovery implementation: {e}")

    def _calculate_backoff_time(self) -> float:
        """Calculate adaptive backoff time based on error patterns."""
        base_backoff = 5.0  # Base 5 seconds

        # Exponential backoff based on error rate
        if self.metrics.total_requests > 0:
            error_rate = self.metrics.failed_requests / self.metrics.total_requests
            # Higher error rate = longer backoff
            backoff_multiplier = 1 + (error_rate * 5)  # Up to 6x base backoff
        else:
            backoff_multiplier = 1.0

        # Cap at 300 seconds (5 minutes)
        backoff_time = min(300.0, base_backoff * backoff_multiplier)

        # Adjust based on flux level (higher flux = shorter backoff)
        flux_adjustment = 1.0 - (self.flux_level * 0.5)
        backoff_time *= flux_adjustment

        return max(1.0, backoff_time)

    def _temporary_reduce_frequency(self):
        """Temporarily reduce agent cycle frequency."""
        if not hasattr(self, '_original_cycle_interval'):
            self._original_cycle_interval = self._get_cycle_interval()
        # Double the cycle interval temporarily
        self.config['cycle_interval'] = self._original_cycle_interval * 2.0
        logger.info(f"Temporarily increased cycle interval to {self.config['cycle_interval']}s")

    def _increase_delays(self):
        """Increase delays between operations."""
        if not hasattr(self, '_original_delays'):
            self._original_delays = {}
        # Increase various delays by 50%
        for key in ['api_delay', 'request_delay', 'cycle_delay']:
            if key in self.config:
                self._original_delays[key] = self.config[key]
                self.config[key] *= 1.5
        logger.info("Increased delays due to rate limiting")

    def _emergency_cleanup(self):
        """Perform emergency cleanup to free resources."""
        try:
            # Force garbage collection
            import gc
            collected = gc.collect()
            logger.info(f"Emergency cleanup: collected {collected} objects")

            # Reduce flux temporarily to ease load
            if self.flux_level > 0.3:
                self.update_flux_level(self.flux_level * 0.8)

        except Exception as e:
            logger.error(f"Error in emergency cleanup: {e}")

    def _record_cycle_metrics(self, cycle_time: float):
        """Record cycle execution metrics."""
        self.metrics.cycle_count += 1
        self.metrics.last_cycle_time = cycle_time

        # Update average cycle time using exponential moving average
        alpha = 0.1
        self.metrics.average_cycle_time = (
            alpha * cycle_time +
            (1 - alpha) * self.metrics.average_cycle_time
        )

    def record_flux_change(self):
        """Record a flux level change."""
        self.metrics.flux_changes += 1

    def record_recovery_event(self):
        """Record a recovery event."""
        self.metrics.recovery_events += 1

    def _temporary_pause(self, duration: float = 60.0):
        """Temporarily pause agent execution."""
        def resume_agent():
            time.sleep(duration)
            if self.status == AgentStatus.PAUSED:
                self.resume()
                logger.info(f"Agent {self.agent_id} automatically resumed after {duration}s pause")

        # Start resume thread
        resume_thread = threading.Thread(target=resume_agent, daemon=True)
        resume_thread.start()

        # Pause the agent
        self.pause()
        logger.warning(f"Agent {self.agent_id} temporarily paused for {duration}s due to persistent errors")

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
        old_flux = self.flux_level
        self.flux_level = max(0.0, min(1.0, new_flux_level))
        self.metrics.flux_level = self.flux_level

        # Record flux change if significant
        if abs(self.flux_level - old_flux) > 0.01:
            self.record_flux_change()

        # Update LLM model if it supports flux adaptation
        if self.llm_model and hasattr(self.llm_model, 'apply_flux_adaptation'):
            logger.info(f"🌊 Agent {self.agent_id} flux level updated: {old_flux:.2f} → {self.flux_level:.2f}")

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