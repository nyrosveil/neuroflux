"""
🧠 NeuroFlux Claude Model
Anthropic Claude integration with neuro-flux adaptations.

Built with love by Nyros Veil 🚀
"""

import os
from typing import Dict, Any, Optional
from termcolor import cprint
from dotenv import load_dotenv

from .base_model import BaseModel

# Load environment variables
load_dotenv()

class ClaudeModel(BaseModel):
    """Claude model implementation with neuro-flux enhancements."""

    def __init__(self, model_name: str = "claude-3-haiku-20240307", **kwargs):
        # Get API key from environment or kwargs
        api_key = kwargs.get('api_key') or os.getenv("ANTHROPIC_KEY")
        super().__init__(model_name, api_key, **kwargs)

    def generate_response(self,
                         system_prompt: str,
                         user_content: str,
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a response using Claude API with neuro-flux adaptations.

        Args:
            system_prompt: System/instruction prompt
            user_content: User message content
            temperature: Base temperature (will be adapted by flux)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (flux_level, etc.)

        Returns:
            Dict containing response data and metadata
        """
        try:
            # Validate API key
            if not self.validate_api_key():
                return self.handle_error(ValueError("Invalid API key"), "api_key_validation")

            # Apply neuro-flux adaptation to temperature
            flux_level = kwargs.get('flux_level', 0.5)
            adapted_temperature = self.apply_flux_adaptation(temperature, flux_level)

            # Rate limiting
            self.rate_limit_wait()

            # Import here to avoid import errors if not installed
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            # Make API call
            message = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=adapted_temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )

            response_text = message.content[0].text

            # Return structured response
            return {
                'success': True,
                'response': response_text,
                'model': self.model_name,
                'temperature_used': adapted_temperature,
                'flux_level': flux_level,
                'tokens_used': getattr(message, 'usage', {}).get('total_tokens', 0),
                'metadata': self.get_model_info()
            }

        except Exception as e:
            cprint(f"❌ Claude API error: {str(e)}", "red")
            return self.handle_error(e, "generate_response")

    def generate_structured_output(self,
                                  system_prompt: str,
                                  user_content: str,
                                  output_schema: Dict[str, Any],
                                  temperature: float = 0.3,
                                  **kwargs) -> Dict[str, Any]:
        """
        Generate structured JSON output using Claude.

        Args:
            system_prompt: System prompt with JSON formatting instructions
            user_content: User content
            output_schema: Schema for structured output
            temperature: Temperature for generation
            **kwargs: Additional parameters

        Returns:
            Dict containing parsed JSON response
        """
        # Enhanced system prompt for JSON output
        json_prompt = f"{system_prompt}\n\nIMPORTANT: Respond with valid JSON only, matching this schema: {output_schema}"

        response = self.generate_response(
            json_prompt,
            user_content,
            temperature=temperature,
            max_tokens=2000,
            **kwargs
        )

        if not response.get('success'):
            return response

        # Try to parse JSON from response
        import json
        import re

        response_text = response['response']

        try:
            # Try direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or text
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass

            # Fallback: extract any JSON-like structure
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            # Last resort: return error
            return {
                'error': True,
                'error_message': 'Failed to parse JSON from response',
                'raw_response': response_text
            }