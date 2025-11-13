"""
🧠 NeuroFlux Groq Model
Groq fast inference integration with neuro-flux adaptations.

Built with love by Nyros Veil 🚀
"""

import os
from typing import Dict, Any, Optional
from termcolor import cprint
from dotenv import load_dotenv

from .base_model import BaseModel

# Load environment variables
load_dotenv()

class GroqModel(BaseModel):
    """Groq model implementation with neuro-flux enhancements."""

    def __init__(self, model_name: str = "mixtral-8x7b-32768", **kwargs):
        api_key = kwargs.get('api_key') or os.getenv("GROQ_API_KEY")
        super().__init__(model_name, api_key, **kwargs)

    def generate_response(self,
                         system_prompt: str,
                         user_content: str,
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a response using Groq API with neuro-flux adaptations.
        """
        try:
            if not self.validate_api_key():
                return self.handle_error(ValueError("Invalid API key"), "api_key_validation")

            flux_level = kwargs.get('flux_level', 0.5)
            adapted_temperature = self.apply_flux_adaptation(temperature, flux_level)
            self.rate_limit_wait()

            import groq
            client = groq.Groq(api_key=self.api_key)

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.model_name,
                temperature=adapted_temperature,
                max_tokens=max_tokens
            )

            response_text = response.choices[0].message.content

            return {
                'success': True,
                'response': response_text,
                'model': self.model_name,
                'temperature_used': adapted_temperature,
                'flux_level': flux_level,
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0,
                'metadata': self.get_model_info()
            }

        except Exception as e:
            cprint(f"❌ Groq API error: {str(e)}", "red")
            return self.handle_error(e, "generate_response")