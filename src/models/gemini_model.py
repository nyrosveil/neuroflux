"""
🧠 NeuroFlux Gemini Model
Google Gemini integration with neuro-flux adaptations.

Built with love by Nyros Veil 🚀
"""

import os
from typing import Dict, Any, Optional
from termcolor import cprint
from dotenv import load_dotenv

from .base_model import BaseModel

# Load environment variables
load_dotenv()

class GeminiModel(BaseModel):
    """Gemini model implementation with neuro-flux enhancements."""

    def __init__(self, model_name: str = "gemini-pro", **kwargs):
        api_key = kwargs.get('api_key') or os.getenv("GEMINI_KEY")
        super().__init__(model_name, api_key, **kwargs)

    def generate_response(self,
                         system_prompt: str,
                         user_content: str,
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a response using Gemini API with neuro-flux adaptations.
        """
        try:
            if not self.validate_api_key():
                return self.handle_error(ValueError("Invalid API key"), "api_key_validation")

            flux_level = kwargs.get('flux_level', 0.5)
            adapted_temperature = self.apply_flux_adaptation(temperature, flux_level)
            self.rate_limit_wait()

            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                f"{system_prompt}\n\n{user_content}",
                generation_config=genai.types.GenerationConfig(
                    temperature=adapted_temperature,
                    max_output_tokens=max_tokens
                )
            )

            response_text = response.text

            return {
                'success': True,
                'response': response_text,
                'model': self.model_name,
                'temperature_used': adapted_temperature,
                'flux_level': flux_level,
                'tokens_used': 0,  # Gemini doesn't provide token counts easily
                'metadata': self.get_model_info()
            }

        except Exception as e:
            cprint(f"❌ Gemini API error: {str(e)}", "red")
            return self.handle_error(e, "generate_response")