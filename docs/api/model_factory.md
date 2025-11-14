# 🧠 Model Factory API

## Overview

The `ModelFactory` class provides unified access to multiple LLM providers with neuro-flux enhancements, enabling adaptive model selection and configuration based on market conditions.

**Location:** `src/models/model_factory.py`

**Supported Providers:** Claude, OpenAI, DeepSeek, Groq, Gemini, Ollama

---

## Class Architecture

```python
ModelFactory
├── Model Creation
├── Flux Adaptation
├── Provider Management
├── Performance Testing
└── State Management
```

---

## Core Methods

### Model Creation

#### `create_model(provider: str, model_name: Optional[str] = None, **kwargs) → BaseModel`
**Description:** Create a standard LLM model instance for the specified provider

**Parameters:**
- `provider` (str): Provider name ('claude', 'openai', 'deepseek', 'groq', 'gemini', 'ollama')
- `model_name` (str, optional): Specific model name, uses default if not provided
- `**kwargs`: Additional provider-specific parameters

**Returns:**
- `BaseModel`: Configured model instance

**Raises:**
- `ValueError`: If provider is unsupported
- `ImportError`: If required dependencies are missing

**Example:**
```python
from src.models.model_factory import ModelFactory

# Create Claude model
claude_model = ModelFactory.create_model('claude', 'claude-3-haiku-20240307')

# Create OpenAI model with custom settings
openai_model = ModelFactory.create_model(
    'openai',
    'gpt-4',
    temperature=0.7,
    max_tokens=1000
)
```

#### `create_flux_optimized_model(provider: str, model_name: Optional[str] = None, flux_level: float = 0.5) → BaseModel`
**Description:** Create a model optimized for specific neuro-flux conditions

**Parameters:**
- `provider` (str): Provider name
- `model_name` (str, optional): Specific model name
- `flux_level` (float): Current market flux level (0.0 to 1.0)

**Returns:**
- `BaseModel`: Flux-optimized model instance

**Flux Adaptations:**
- **High Flux (0.7+)**: Increased creativity, faster rate limits
- **Low Flux (0.3-)**: Reduced creativity, standard rate limits
- **Normal Flux**: Balanced parameters

**Example:**
```python
# High flux conditions - more creative responses
high_flux_model = ModelFactory.create_flux_optimized_model(
    'claude',
    flux_level=0.8
)

# Low flux conditions - more conservative responses
low_flux_model = ModelFactory.create_flux_optimized_model(
    'openai',
    flux_level=0.2
)
```

### Provider Management

#### `get_available_providers() → list`
**Description:** Get list of all supported LLM providers

**Returns:**
- `list`: List of provider names

**Example:**
```python
providers = ModelFactory.get_available_providers()
print(providers)  # ['claude', 'openai', 'deepseek', 'groq', 'gemini', 'ollama']
```

#### `test_model(provider: str, model_name: Optional[str] = None) → bool`
**Description:** Test if a model provider is available and working

**Parameters:**
- `provider` (str): Provider to test
- `model_name` (str, optional): Specific model to test

**Returns:**
- `bool`: True if model is accessible and functional

**Example:**
```python
# Test Claude availability
if ModelFactory.test_model('claude'):
    print("Claude is available")
else:
    print("Claude is not configured")
```

### State Management

#### `get_model(provider: str = 'claude', model_name: Optional[str] = None) → BaseModel`
**Description:** Get a model instance with state management

**Parameters:**
- `provider` (str): Provider name (default: 'claude')
- `model_name` (str, optional): Specific model name

**Returns:**
- `BaseModel`: Model instance with state tracking

**Example:**
```python
factory = ModelFactory()
model = factory.get_model('claude')
active_name = factory.get_active_model_name()
print(f"Active model: {active_name}")
```

#### `get_active_model_name() → str`
**Description:** Get the name of the currently active model

**Returns:**
- `str`: Active model identifier

---

## Default Model Configurations

```python
DEFAULT_MODELS = {
    'claude': 'claude-3-haiku-20240307',
    'openai': 'gpt-4',
    'deepseek': 'deepseek-chat',
    'groq': 'mixtral-8x7b-32768',
    'gemini': 'gemini-pro',
    'ollama': 'llama2'
}
```

---

## Usage Examples

### Basic Model Usage

```python
from src.models.model_factory import ModelFactory

# Create and use a model
model = ModelFactory.create_model('claude')

response = model.generate_response(
    system_prompt="You are a trading analyst",
    user_content="Analyze BTC price action",
    temperature=0.7
)

print(response['content'])
```

### Flux-Adaptive Model Selection

```python
def select_model_for_conditions(market_flux: float):
    """Select appropriate model based on market conditions"""

    if market_flux > 0.8:
        # High volatility - use creative model
        return ModelFactory.create_flux_optimized_model('groq', flux_level=market_flux)
    elif market_flux < 0.3:
        # Low volatility - use precise model
        return ModelFactory.create_flux_optimized_model('claude', flux_level=market_flux)
    else:
        # Normal conditions - use balanced model
        return ModelFactory.create_flux_optimized_model('openai', flux_level=market_flux)

# Usage
current_flux = 0.7
model = select_model_for_conditions(current_flux)
```

### Multi-Provider Fallback

```python
def get_working_model(preferred_providers: list):
    """Get first available model from preference list"""

    for provider in preferred_providers:
        if ModelFactory.test_model(provider):
            return ModelFactory.create_model(provider)

    # Fallback to basic Claude
    return ModelFactory.create_model('claude')

# Usage
providers = ['claude', 'openai', 'groq']
model = get_working_model(providers)
```

### Batch Model Testing

```python
def test_all_providers():
    """Test availability of all supported providers"""

    results = {}
    for provider in ModelFactory.get_available_providers():
        try:
            available = ModelFactory.test_model(provider)
            results[provider] = "✅ Available" if available else "❌ Unavailable"
        except Exception as e:
            results[provider] = f"❌ Error: {e}"

    return results

# Usage
status = test_all_providers()
for provider, status in status.items():
    print(f"{provider}: {status}")
```

---

## Provider-Specific Features

### Claude (Anthropic)
- **Models**: claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
- **Strengths**: Balanced performance, good reasoning
- **Use Case**: General analysis and strategy development

### OpenAI
- **Models**: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- **Strengths**: Strong reasoning, good for complex analysis
- **Use Case**: Detailed market research and predictions

### Groq
- **Models**: mixtral-8x7b-32768, llama2-70b-4096
- **Strengths**: Fast inference, cost-effective
- **Use Case**: High-frequency operations, real-time analysis

### DeepSeek
- **Models**: deepseek-chat, deepseek-coder
- **Strengths**: Code generation, technical analysis
- **Use Case**: Strategy implementation, backtesting

### Gemini (Google)
- **Models**: gemini-pro, gemini-pro-vision
- **Strengths**: Multimodal, good for visual analysis
- **Use Case**: Chart analysis, pattern recognition

### Ollama
- **Models**: llama2, codellama, mistral
- **Strengths**: Local execution, privacy
- **Use Case**: Offline development, sensitive operations

---

## Error Handling

```python
try:
    model = ModelFactory.create_model('unsupported_provider')
except ValueError as e:
    print(f"Unsupported provider: {e}")

try:
    model = ModelFactory.create_model('claude')
    # Model creation succeeded but API might fail
    response = model.generate_response("test", "test")
except ImportError as e:
    print(f"Missing dependencies: {e}")
except Exception as e:
    print(f"Model error: {e}")
```

---

## Performance Considerations

- **Rate Limits**: Different providers have different rate limits
- **Cost**: Consider token costs for production use
- **Latency**: Groq offers fastest inference, Claude most balanced
- **Flux Adaptation**: Higher flux levels may increase API costs
- **Caching**: Consider response caching for repeated queries

---

## Configuration

Models can be configured through environment variables:

```bash
# Claude
ANTHROPIC_API_KEY=your_key_here

# OpenAI
OPENAI_API_KEY=your_key_here

# DeepSeek
DEEPSEEK_API_KEY=your_key_here

# Groq
GROQ_API_KEY=your_key_here

# Gemini
GOOGLE_API_KEY=your_key_here
```

---

## Integration with Base Agent

```python
from src.agents.base_agent import BaseAgent
from src.models.model_factory import ModelFactory

class CustomAgent(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id=agent_id, flux_level=0.5)
        # Model will be initialized automatically in _initialize_agent

    def _initialize_agent(self) -> bool:
        # Model factory handles provider selection
        return True

    def _execute_agent_cycle(self):
        # Use the inherited generate_response method
        response = self.generate_response(
            system_prompt="You are a trading assistant",
            user_content="What is the current market sentiment?"
        )
        print(f"Analysis: {response['content']}")
```

---

## Related APIs

- **[Base Agent](base_agent.md)** - Agent lifecycle management
- **[Base Model](models/base_model.md)** - Model interface definition
- **[Task Orchestrator](task_orchestrator.md)** - Agent coordination

---

*Built with ❤️ by Nyros Veil* | [Back to API Index](../README.md)