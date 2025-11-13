"""
🧠 NeuroFlux Base Model
Abstract base class for all LLM model implementations.

Built with love by Nyros Veil 🚀

Provides common functionality and neuro-flux adaptations for all models.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseModel(ABC):
    """Abstract base class for neuro-flux enhanced LLM models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the base model.

        Args:
            model_name: Name/identifier of the model
            api_key: API key for the model provider
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.flux_sensitivity = kwargs.get('flux_sensitivity', 0.8)
        self.adaptive_temperature = kwargs.get('adaptive_temperature', True)
        self.rate_limit_delay = kwargs.get('rate_limit_delay', 1.0)
        self.last_request_time = 0

        # Neuro-flux state
        self.flux_level = kwargs.get('flux_level', 0.5)
        self.adaptation_history = []

    @abstractmethod
    def generate_response(self,
                         system_prompt: str,
                         user_content: str,
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the model.

        Args:
            system_prompt: System/instruction prompt
            user_content: User message content
            temperature: Base temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Dict containing response data and metadata
        """
        pass

    def adapt_temperature(self, base_temperature: float, flux_level: Optional[float] = None) -> float:
        """
        Adapt temperature based on neuro-flux conditions.

        Args:
            base_temperature: Base temperature setting
            flux_level: Current market flux level (0-1)

        Returns:
            float: Adapted temperature
        """
        if not self.adaptive_temperature:
            return base_temperature

        flux = flux_level if flux_level is not None else self.flux_level

        # High flux: more creative (higher temperature)
        # Low flux: more consistent (lower temperature)
        if flux > 0.7:
            adapted_temp = min(1.5, base_temperature * (1 + self.flux_sensitivity * 0.5))
        elif flux < 0.3:
            adapted_temp = max(0.1, base_temperature * (1 - self.flux_sensitivity * 0.3))
        else:
            adapted_temp = base_temperature

        return adapted_temp

    def apply_rate_limit(self):
        """Apply rate limiting to prevent API abuse."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def validate_api_key(self) -> bool:
        """Validate that the API key is present and valid format."""
        if not self.api_key:
            return False

        # Basic validation - can be overridden by subclasses
        return len(self.api_key.strip()) > 10

    def handle_error(self, error: Exception, error_type: str) -> Dict[str, Any]:
        """
        Handle and format errors consistently.

        Args:
            error: The exception that occurred
            error_type: Type/category of the error

        Returns:
            Dict with error information
        """
        error_info = {
            'success': False,
            'error': str(error),
            'error_type': error_type,
            'timestamp': time.time(),
            'model_name': self.model_name
        }

        cprint(f"❌ Model error ({error_type}): {str(error)}", "red")
        return error_info

    def update_flux_state(self, new_flux_level: float):
        """
        Update the model's flux state and adaptation history.

        Args:
            new_flux_level: New flux level (0-1)
        """
        self.flux_level = max(0.0, min(1.0, new_flux_level))
        self.adaptation_history.append({
            'timestamp': time.time(),
            'flux_level': self.flux_level,
            'model_name': self.model_name
        })

        # Keep only last 100 adaptations
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model state."""
        return {
            'model_name': self.model_name,
            'flux_level': self.flux_level,
            'flux_sensitivity': self.flux_sensitivity,
            'adaptive_temperature': self.adaptive_temperature,
            'rate_limit_delay': self.rate_limit_delay,
            'adaptation_count': len(self.adaptation_history)
        }