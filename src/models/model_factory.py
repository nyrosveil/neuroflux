"""
🧠 NeuroFlux Model Factory
LLM provider abstraction layer with neuro-flux enhancements.

Built with love by Nyros Veil 🚀

Unified interface for multiple AI providers with adaptive parameters.
Supports Claude, GPT-4, DeepSeek, Groq, Gemini, Ollama with flux-aware adjustments.
"""

import os
from typing import Dict, Any, Optional
from termcolor import cprint
from dotenv import load_dotenv

# Neuro-flux enhanced model classes will be imported dynamically

# Load environment variables
load_dotenv()

class ModelFactory:
    """Factory class for creating neuro-flux enhanced LLM model instances."""

    # Default model configurations
    DEFAULT_MODELS = {
        'claude': 'claude-3-haiku-20240307',
        'openai': 'gpt-4',
        'deepseek': 'deepseek-chat',
        'groq': 'mixtral-8x7b-32768',
        'gemini': 'gemini-pro',
        'ollama': 'llama2'
    }

    def __init__(self):
        """Initialize the model factory with state tracking."""
        self.active_model = None
        self.active_model_name = None
        self.created_models = {}

    @staticmethod
    def create_model(provider: str, model_name: Optional[str] = None, **kwargs):
        """
        Create a neuro-flux enhanced model instance for the specified provider.

        Args:
            provider (str): Provider name (claude, openai, deepseek, groq, gemini, ollama)
            model_name (str, optional): Specific model name, uses default if not provided
            **kwargs: Additional configuration parameters (flux_sensitivity, adaptive_temperature, etc.)

        Returns:
            BaseModel: Neuro-flux enhanced model instance
        """
        provider = provider.lower()
        model_name = model_name or ModelFactory.DEFAULT_MODELS.get(provider)

        if not model_name:
            raise ValueError(f"No default model found for provider: {provider}")

        # Dynamically import and create neuro-flux enhanced model instance
        try:
            if provider == 'claude':
                from .claude_model import ClaudeModel
                return ClaudeModel(model_name, **kwargs)
            elif provider == 'openai':
                from .openai_model import OpenAIModel
                return OpenAIModel(model_name, **kwargs)
            elif provider == 'deepseek':
                from .deepseek_model import DeepSeekModel
                return DeepSeekModel(model_name, **kwargs)
            elif provider == 'groq':
                from .groq_model import GroqModel
                return GroqModel(model_name, **kwargs)
            elif provider == 'gemini':
                from .gemini_model import GeminiModel
                return GeminiModel(model_name, **kwargs)
            elif provider == 'ollama':
                from .ollama_model import OllamaModel
                return OllamaModel(model_name, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}. Supported: {list(ModelFactory.DEFAULT_MODELS.keys())}")
        except ImportError as e:
            raise ImportError(f"Failed to import model class for provider '{provider}': {e}")

    @staticmethod
    def get_available_providers() -> list:
        """Get list of available providers."""
        return list(ModelFactory.DEFAULT_MODELS.keys())

    @staticmethod
    def test_model(provider: str, model_name: Optional[str] = None) -> bool:
        """
        Test if a neuro-flux model is available and working.

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
            return response.get('success', False) and 'OK' in response.get('response', '').upper()
        except Exception:
            return False

    @staticmethod
    def create_flux_optimized_model(provider: str, model_name: Optional[str] = None,
                                  flux_level: float = 0.5):
        """
        Create a model optimized for specific flux conditions.

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

    def get_model(self, provider: str = 'claude', model_name: Optional[str] = None):
        """
        Get a neuro-flux enhanced model instance, maintaining state.

        Args:
            provider (str): Provider name (defaults to 'claude')
            model_name (str, optional): Specific model name

        Returns:
            BaseModel: Neuro-flux enhanced model instance
        """
        model = self.create_model(provider, model_name)
        self.active_model = model
        self.active_model_name = getattr(model, 'model_name', f"{provider}:{model_name or self.DEFAULT_MODELS.get(provider, 'unknown')}")
        return model

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
    cprint("🧠 NeuroFlux Model Factory", "cyan")
    cprint("Available providers:", "white")
    for provider in ModelFactory.get_available_providers():
        status = "✅" if ModelFactory.test_model(provider) else "❌"
        cprint(f"  {provider}: {status}", "green" if status == "✅" else "red")