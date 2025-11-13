"""
🧠 NeuroFlux Ollama Model
Local Ollama integration with neuro-flux adaptations.

Built with love by Nyros Veil 🚀
"""

import os
from typing import Dict, Any, Optional
from termcolor import cprint
from dotenv import load_dotenv

from .base_model import BaseModel

# Load environment variables
load_dotenv()

class OllamaModel(BaseModel):
    """Ollama model implementation with neuro-flux enhancements."""

    def __init__(self, model_name: str = "llama2", **kwargs):
        # No API key needed for local models
        super().__init__(model_name, api_key="", **kwargs)

    def generate_response(self,
                         system_prompt: str,
                         user_content: str,
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a response using local Ollama API with neuro-flux adaptations.
        """
        try:
            flux_level = kwargs.get('flux_level', 0.5)
            adapted_temperature = self.apply_flux_adaptation(temperature, flux_level)
            self.rate_limit_wait()

            import ollama

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                options={
                    "temperature": adapted_temperature,
                    "num_predict": max_tokens
                }
            )

            response_text = response['message']['content']

            return {
                'success': True,
                'response': response_text,
                'model': self.model_name,
                'temperature_used': adapted_temperature,
                'flux_level': flux_level,
                'tokens_used': 0,  # Ollama doesn't provide token counts easily
                'metadata': self.get_model_info()
            }

        except Exception as e:
            cprint(f"❌ Ollama API error: {str(e)}", "red")
            return self.handle_error(e, "generate_response")

    def validate_api_key(self) -> bool:
        """Ollama doesn't require API key validation."""
        return True