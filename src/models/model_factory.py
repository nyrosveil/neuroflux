"""
🧠 NeuroFlux Model Factory
LLM provider abstraction layer with neuro-flux enhancements.

Built with love by Nyros Veil 🚀

Unified interface for multiple AI providers with adaptive parameters.
Supports Claude, GPT-4, DeepSeek, Groq, Gemini, Ollama with flux-aware adjustments.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseModel(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key()

    @abstractmethod
    def _get_api_key(self) -> str:
        """Get API key for the provider."""
        pass

    @abstractmethod
    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate a text response."""
        pass

    def generate_structured_output(self, system_prompt: str, user_content: str,
                                  output_schema: Dict[str, Any],
                                  temperature: float = 0.3) -> Dict[str, Any]:
        """Generate structured JSON output."""
        # Default implementation - can be overridden by providers that support it
        response = self.generate_response(system_prompt, user_content, temperature, 2000)

        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            # Last resort: return basic structure
            return {"response": response, "parsed": False}

class AnthropicModel(BaseModel):
    """Claude model implementation."""

    def _get_api_key(self) -> str:
        return os.getenv("ANTHROPIC_KEY", "")

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            message = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_content}
                ]
            )

            return message.content[0].text

        except Exception as e:
            cprint(f"❌ Anthropic API error: {str(e)}", "red")
            return f"Error: {str(e)}"

class OpenAIModel(BaseModel):
    """OpenAI GPT model implementation."""

    def _get_api_key(self) -> str:
        return os.getenv("OPENAI_KEY", "")

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            cprint(f"❌ OpenAI API error: {str(e)}", "red")
            return f"Error: {str(e)}"

class DeepSeekModel(BaseModel):
    """DeepSeek model implementation."""

    def _get_api_key(self) -> str:
        return os.getenv("DEEPSEEK_KEY", "")

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            cprint(f"❌ DeepSeek API error: {str(e)}", "red")
            return f"Error: {str(e)}"

class GroqModel(BaseModel):
    """Groq fast inference model implementation."""

    def _get_api_key(self) -> str:
        return os.getenv("GROQ_API_KEY", "")

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            import groq
            client = groq.Groq(api_key=self.api_key)

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            cprint(f"❌ Groq API error: {str(e)}", "red")
            return f"Error: {str(e)}"

class GeminiModel(BaseModel):
    """Google Gemini model implementation."""

    def _get_api_key(self) -> str:
        return os.getenv("GEMINI_KEY", "")

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                f"{system_prompt}\\n\\n{user_content}",
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )

            return response.text

        except Exception as e:
            cprint(f"❌ Gemini API error: {str(e)}", "red")
            return f"Error: {str(e)}"

class OllamaModel(BaseModel):
    """Local Ollama model implementation."""

    def _get_api_key(self) -> str:
        return ""  # No API key needed for local models

    def generate_response(self, system_prompt: str, user_content: str,
                         temperature: float = 0.7, max_tokens: int = 1000) -> str:
        try:
            import ollama

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )

            return response['message']['content']

        except Exception as e:
            cprint(f"❌ Ollama API error: {str(e)}", "red")
            return f"Error: {str(e)}"

class ModelFactory:
    """Factory class for creating LLM model instances."""

    # Default model configurations
    DEFAULT_MODELS = {
        'anthropic': 'claude-3-haiku-20240307',
        'openai': 'gpt-4',
        'deepseek': 'deepseek-chat',
        'groq': 'mixtral-8x7b-32768',
        'gemini': 'gemini-pro',
        'ollama': 'llama2'
    }

    @staticmethod
    def create_model(provider: str, model_name: Optional[str] = None) -> BaseModel:
        """
        Create a model instance for the specified provider.

        Args:
            provider (str): Provider name (anthropic, openai, deepseek, groq, gemini, ollama)
            model_name (str, optional): Specific model name, uses default if not provided

        Returns:
            BaseModel: Model instance
        """
        provider = provider.lower()
        model_name = model_name or ModelFactory.DEFAULT_MODELS.get(provider)

        if not model_name:
            raise ValueError(f"No default model found for provider: {provider}")

        # Create model instance based on provider
        if provider == 'anthropic':
            return AnthropicModel(model_name)
        elif provider == 'openai':
            return OpenAIModel(model_name)
        elif provider == 'deepseek':
            return DeepSeekModel(model_name)
        elif provider == 'groq':
            return GroqModel(model_name)
        elif provider == 'gemini':
            return GeminiModel(model_name)
        elif provider == 'ollama':
            return OllamaModel(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_available_providers() -> list:
        """Get list of available providers."""
        return list(ModelFactory.DEFAULT_MODELS.keys())

    @staticmethod
    def test_model(provider: str, model_name: Optional[str] = None) -> bool:
        """
        Test if a model is available and working.

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
            return 'OK' in response.upper()
        except Exception:
            return False

# Initialize model factory
if __name__ == "__main__":
    cprint("🧠 NeuroFlux Model Factory", "cyan")
    cprint("Available providers:", "white")
    for provider in ModelFactory.get_available_providers():
        status = "✅" if ModelFactory.test_model(provider) else "❌"
        cprint(f"  {provider}: {status}", "green" if status == "✅" else "red")