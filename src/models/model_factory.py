"""
🧠 NeuroFlux Model Factory
LLM provider abstraction layer with neuro-flux enhancements.

Built with love by Nyros Veil 🚀

Unified interface for multiple AI providers with adaptive parameters.
Supports Claude, GPT-4, DeepSeek, Groq, Gemini, Ollama with flux-aware adjustments.
"""

import os
import time
from typing import Dict, Any, Optional, List
from termcolor import cprint
from dotenv import load_dotenv

# Neuro-flux enhanced model classes will be imported dynamically

# Load environment variables
load_dotenv()

class ModelFactory:
    """Enhanced factory class for creating neuro-flux enhanced LLM model instances with cost tracking and fallback."""

    # Default model configurations
    DEFAULT_MODELS = {
        'claude': 'claude-3-haiku-20240307',
        'openai': 'gpt-4',
        'deepseek': 'deepseek-chat',
        'groq': 'mixtral-8x7b-32768',
        'gemini': 'gemini-pro',
        'ollama': 'llama2'
    }

    # Cost per 1K tokens (approximate, update as needed)
    COST_PER_1K_TOKENS = {
        'claude': {'input': 0.0015, 'output': 0.005},  # Claude 3 Haiku
        'openai': {'input': 0.0015, 'output': 0.002},   # GPT-4
        'deepseek': {'input': 0.0001, 'output': 0.0002}, # DeepSeek
        'groq': {'input': 0.0002, 'output': 0.0002},    # Mixtral
        'gemini': {'input': 0.00025, 'output': 0.0005}, # Gemini Pro
        'ollama': {'input': 0.0, 'output': 0.0}         # Local, free
    }

    # Fallback priority order (most reliable to least)
    FALLBACK_ORDER = ['claude', 'openai', 'groq', 'deepseek', 'gemini', 'ollama']

    def __init__(self):
        """Initialize the model factory with enhanced state tracking."""
        self.active_model = None
        self.active_model_name = None
        self.created_models = {}
        self.cost_tracking = {
            'total_cost': 0.0,
            'provider_costs': {},
            'monthly_usage': {},
            'last_reset': time.time()
        }
        self.error_tracking = {
            'provider_errors': {},
            'recent_failures': [],
            'circuit_breakers': {}
        }

    @staticmethod
    def create_model(provider: str, model_name: Optional[str] = None, **kwargs):
        """
        Create a neuro-flux enhanced model instance for the specified provider with enhanced error handling.

        Args:
            provider (str): Provider name (claude, openai, deepseek, groq, gemini, ollama)
            model_name (str, optional): Specific model name, uses default if not provided
            **kwargs: Additional configuration parameters (flux_sensitivity, adaptive_temperature, etc.)

        Returns:
            BaseModel: Neuro-flux enhanced model instance

        Raises:
            ValueError: If provider is unsupported
            ImportError: If model class cannot be imported
            RuntimeError: If model initialization fails
        """
        provider = provider.lower()
        model_name = model_name or ModelFactory.DEFAULT_MODELS.get(provider)

        if not model_name:
            raise ValueError(f"No default model found for provider: {provider}")

        # Check circuit breaker
        if ModelFactory._is_circuit_breaker_tripped(provider):
            raise RuntimeError(f"Circuit breaker tripped for provider: {provider}")

        # Dynamically import and create neuro-flux enhanced model instance
        try:
            if provider == 'claude':
                from .claude_model import ClaudeModel
                model_class = ClaudeModel
            elif provider == 'openai':
                from .openai_model import OpenAIModel
                model_class = OpenAIModel
            elif provider == 'deepseek':
                from .deepseek_model import DeepSeekModel
                model_class = DeepSeekModel
            elif provider == 'groq':
                from .groq_model import GroqModel
                model_class = GroqModel
            elif provider == 'gemini':
                from .gemini_model import GeminiModel
                model_class = GeminiModel
            elif provider == 'ollama':
                from .ollama_model import OllamaModel
                model_class = OllamaModel
            else:
                raise ValueError(f"Unsupported provider: {provider}. Supported: {list(ModelFactory.DEFAULT_MODELS.keys())}")

            # Create model instance
            model = model_class(model_name, **kwargs)

            # Validate model
            if not model.validate_api_key():
                raise RuntimeError(f"Invalid API key for provider: {provider}")

            return model

        except ImportError as e:
            ModelFactory._record_provider_error(provider, 'import_error', str(e))
            raise ImportError(f"Failed to import model class for provider '{provider}': {e}")
        except Exception as e:
            ModelFactory._record_provider_error(provider, 'initialization_error', str(e))
            raise RuntimeError(f"Failed to initialize model for provider '{provider}': {e}")

    @staticmethod
    def get_available_providers() -> list:
        """Get list of available providers."""
        return list(ModelFactory.DEFAULT_MODELS.keys())

    @staticmethod
    def test_model(provider: str, model_name: Optional[str] = None) -> bool:
        """
        Test if a neuro-flux model is available and working with enhanced validation.

        Args:
            provider (str): Provider name
            model_name (str, optional): Model name

        Returns:
            bool: True if model is working
        """
        try:
            model = ModelFactory.create_model(provider, model_name)
            response = model.generate_response(
                "You are a test assistant",
                "Respond with 'OK' if you can read this.",
                temperature=0.1,
                max_tokens=10
            )
            success = response.get('success', False) and 'OK' in response.get('response', '').upper()

            # Track test result
            if success:
                ModelFactory._record_provider_success(provider)
            else:
                ModelFactory._record_provider_error(provider, 'test_failure', 'Test response validation failed')

            return success
        except Exception as e:
            ModelFactory._record_provider_error(provider, 'test_error', str(e))
            return False

    @staticmethod
    def _is_circuit_breaker_tripped(provider: str) -> bool:
        """Check if circuit breaker is tripped for a provider."""
        # Simple circuit breaker: trip if 5+ errors in last 5 minutes
        errors = ModelFactory._get_recent_provider_errors(provider, 300)  # 5 minutes
        return len(errors) >= 5

    @staticmethod
    def _record_provider_error(provider: str, error_type: str, error_message: str):
        """Record a provider error for tracking and circuit breaker logic."""
        if not hasattr(ModelFactory, '_error_tracking'):
            ModelFactory._error_tracking = {
                'provider_errors': {},
                'recent_failures': []
            }

        # Record provider-specific error
        if provider not in ModelFactory._error_tracking['provider_errors']:
            ModelFactory._error_tracking['provider_errors'][provider] = []

        ModelFactory._error_tracking['provider_errors'][provider].append({
            'timestamp': time.time(),
            'error_type': error_type,
            'error_message': error_message
        })

        # Keep only last 50 errors per provider
        ModelFactory._error_tracking['provider_errors'][provider] = \
            ModelFactory._error_tracking['provider_errors'][provider][-50:]

        # Record recent failure
        ModelFactory._error_tracking['recent_failures'].append({
            'timestamp': time.time(),
            'provider': provider,
            'error_type': error_type,
            'error_message': error_message
        })

        # Keep only last 100 recent failures
        ModelFactory._error_tracking['recent_failures'] = \
            ModelFactory._error_tracking['recent_failures'][-100:]

    @staticmethod
    def _record_provider_success(provider: str):
        """Record a successful provider operation."""
        # Clear recent errors for this provider (circuit breaker recovery)
        if hasattr(ModelFactory, '_error_tracking'):
            recent_failures = ModelFactory._error_tracking.get('recent_failures', [])
            # Remove failures older than 10 minutes for successful providers
            cutoff_time = time.time() - 600  # 10 minutes
            ModelFactory._error_tracking['recent_failures'] = [
                f for f in recent_failures
                if f['provider'] != provider or f['timestamp'] > cutoff_time
            ]

    @staticmethod
    def _get_recent_provider_errors(provider: str, time_window: float) -> List[Dict]:
        """Get recent errors for a provider within time window."""
        if not hasattr(ModelFactory, '_error_tracking'):
            return []

        cutoff_time = time.time() - time_window
        provider_errors = ModelFactory._error_tracking.get('provider_errors', {}).get(provider, [])
        return [e for e in provider_errors if e['timestamp'] > cutoff_time]

    @staticmethod
    def create_flux_optimized_model(provider: str, model_name: Optional[str] = None,
                                   flux_level: float = 0.5):
        """
        Create a model optimized for specific flux conditions with cost consideration.

        Args:
            provider (str): Provider name
            model_name (str, optional): Model name
            flux_level (float): Current market flux level (0-1)

        Returns:
            BaseModel: Flux-optimized model instance
        """
        # Adjust parameters based on flux level
        flux_sensitivity = 0.8
        adaptive_temperature = True

        # High flux: more creative, lower flux: more consistent
        if flux_level > 0.7:
            flux_sensitivity = 1.0  # Maximum adaptation
        elif flux_level < 0.3:
            flux_sensitivity = 0.5  # Reduced adaptation

        return ModelFactory.create_model(
            provider,
            model_name,
            flux_sensitivity=flux_sensitivity,
            adaptive_temperature=adaptive_temperature,
            rate_limit_delay=0.5 if flux_level > 0.5 else 1.0  # Faster in high flux
        )

    @staticmethod
    def create_with_fallback(primary_provider: str, model_name: Optional[str] = None,
                           flux_level: float = 0.5, max_fallbacks: int = 2):
        """
        Create a model with intelligent fallback to alternative providers.

        Args:
            primary_provider (str): Preferred provider
            model_name (str, optional): Model name
            flux_level (float): Current flux level
            max_fallbacks (int): Maximum fallback attempts

        Returns:
            BaseModel: Working model instance (may be from fallback provider)

        Raises:
            RuntimeError: If no providers are available
        """
        providers_to_try = [primary_provider.lower()] + [
            p for p in ModelFactory.FALLBACK_ORDER
            if p != primary_provider.lower()
        ][:max_fallbacks]

        last_error = None

        for provider in providers_to_try:
            try:
                cprint(f"🧠 Trying provider: {provider}", "cyan")
                model = ModelFactory.create_flux_optimized_model(provider, model_name, flux_level)

                # Test the model quickly
                if ModelFactory.test_model(provider, model_name):
                    cprint(f"✅ Successfully created model with provider: {provider}", "green")
                    return model
                else:
                    cprint(f"❌ Provider {provider} test failed", "yellow")

            except Exception as e:
                error_msg = f"Provider {provider} failed: {e}"
                cprint(error_msg, "red")
                last_error = e
                continue

        # All providers failed
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    @staticmethod
    def track_cost(provider: str, input_tokens: int, output_tokens: int):
        """
        Track API usage costs.

        Args:
            provider (str): Provider name
            input_tokens (int): Number of input tokens used
            output_tokens (int): Number of output tokens used
        """
        if not hasattr(ModelFactory, '_cost_tracking'):
            ModelFactory._cost_tracking = {
                'total_cost': 0.0,
                'provider_costs': {},
                'monthly_usage': {}
            }

        provider = provider.lower()
        costs = ModelFactory.COST_PER_1K_TOKENS.get(provider, {'input': 0.0, 'output': 0.0})

        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        total_cost = input_cost + output_cost

        # Update total cost
        ModelFactory._cost_tracking['total_cost'] += total_cost

        # Update provider costs
        if provider not in ModelFactory._cost_tracking['provider_costs']:
            ModelFactory._cost_tracking['provider_costs'][provider] = 0.0
        ModelFactory._cost_tracking['provider_costs'][provider] += total_cost

        # Update monthly usage
        current_month = time.strftime('%Y-%m')
        if current_month not in ModelFactory._cost_tracking['monthly_usage']:
            ModelFactory._cost_tracking['monthly_usage'][current_month] = 0.0
        ModelFactory._cost_tracking['monthly_usage'][current_month] += total_cost

    @staticmethod
    def get_cost_summary() -> Dict[str, Any]:
        """Get cost tracking summary."""
        if not hasattr(ModelFactory, '_cost_tracking'):
            return {'total_cost': 0.0, 'provider_costs': {}, 'monthly_usage': {}}

        return ModelFactory._cost_tracking.copy()

    @staticmethod
    def reset_cost_tracking():
        """Reset cost tracking (typically monthly)."""
        if hasattr(ModelFactory, '_cost_tracking'):
            ModelFactory._cost_tracking = {
                'total_cost': 0.0,
                'provider_costs': {},
                'monthly_usage': {},
                'last_reset': time.time()
            }

    @staticmethod
    def get_provider_health() -> Dict[str, Any]:
        """Get health status of all providers."""
        health = {}

        for provider in ModelFactory.DEFAULT_MODELS.keys():
            recent_errors = ModelFactory._get_recent_provider_errors(provider, 300)  # 5 minutes
            circuit_tripped = ModelFactory._is_circuit_breaker_tripped(provider)

            health[provider] = {
                'available': not circuit_tripped,
                'recent_errors': len(recent_errors),
                'circuit_breaker_tripped': circuit_tripped,
                'last_test_success': ModelFactory.test_model(provider)
            }

        return health

    def get_model(self, provider: str = 'claude', model_name: Optional[str] = None,
                  use_fallback: bool = True):
        """
        Get a neuro-flux enhanced model instance with fallback support.

        Args:
            provider (str): Provider name (defaults to 'claude')
            model_name (str, optional): Specific model name
            use_fallback (bool): Whether to use fallback providers if primary fails

        Returns:
            BaseModel: Neuro-flux enhanced model instance
        """
        try:
            if use_fallback:
                model = ModelFactory.create_with_fallback(provider, model_name)
            else:
                model = self.create_model(provider, model_name)

            self.active_model = model
            self.active_model_name = getattr(model, 'model_name',
                                           f"{provider}:{model_name or self.DEFAULT_MODELS.get(provider, 'unknown')}")
            return model

        except Exception as e:
            cprint(f"❌ Failed to get model for provider {provider}: {e}", "red")
            raise

    def get_active_model_name(self) -> str:
        """
        Get the name of the currently active model.

        Returns:
            str: Active model name
        """
        if self.active_model_name:
            return self.active_model_name
        return "claude:claude-3-haiku-20240307"  # Default fallback

# Initialize model factory
if __name__ == "__main__":
    cprint("🧠 NeuroFlux Enhanced Model Factory", "cyan")
    cprint("Available providers:", "white")

    # Test all providers
    for provider in ModelFactory.get_available_providers():
        status = "✅" if ModelFactory.test_model(provider) else "❌"
        cprint(f"  {provider}: {status}", "green" if status == "✅" else "red")

    # Show cost summary
    cprint("\n💰 Cost Tracking Summary:", "yellow")
    costs = ModelFactory.get_cost_summary()
    cprint(f"  Total Cost: ${costs['total_cost']:.4f}", "white")

    # Show provider health
    cprint("\n🏥 Provider Health:", "yellow")
    health = ModelFactory.get_provider_health()
    for provider, status in health.items():
        health_status = "✅" if status['available'] else "❌"
        cprint(f"  {provider}: {health_status} (errors: {status['recent_errors']})",
               "green" if status['available'] else "red")