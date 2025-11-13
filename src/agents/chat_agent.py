"""
🧠 NeuroFlux's Chat Agent
Conversational AI for trading questions and guidance.

Built with love by Nyros Veil 🚀

Answers user queries about markets, agents, strategies with neuro-enhanced responses.
Provides explanations, recommendations, and educational content.
Integrates with other agents for comprehensive trading assistance.
"""

import os
import json
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Output directory for chat logs
OUTPUT_DIR = "src/data/chat_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_system_prompt():
    """
    Get the system prompt for the chat agent with neuro-flux context.

    Returns:
        str: System prompt
    """
    return f"""You are NeuroFlux's Chat Agent, an advanced AI assistant for cryptocurrency trading.

Your capabilities:
- Answer questions about NeuroFlux agents and their functions
- Provide trading advice based on market analysis
- Explain technical concepts and strategies
- Help with strategy development and backtesting
- Offer risk management guidance
- Assist with exchange setup and configuration

Key NeuroFlux features:
- Neuro-inspired algorithms with flux-adaptive learning
- Multi-exchange support (Solana, Hyperliquid, Extended)
- 48+ specialized agents for comprehensive trading
- Risk-first approach with circuit breakers
- Real-time flux monitoring and adaptation

Guidelines:
- Be helpful, accurate, and educational
- Always emphasize risk management and caution
- Provide actionable advice when appropriate
- Reference specific agents when relevant
- Maintain professional yet friendly tone
- Use flux-awareness in recommendations

Remember: This is experimental software. Always backtest strategies and never risk more than you can afford to lose."""

def get_agent_info():
    """
    Get information about all NeuroFlux agents for reference.

    Returns:
        dict: Agent information
    """
    return {
        "risk_agent": "Circuit breaker that monitors portfolio risk and enforces limits",
        "trading_agent": "Core trading agent with neuro-decision making",
        "chat_agent": "Conversational AI for questions and guidance (you)",
        "rbi_agent": "Research-Based Inference for automated backtesting",
        "sentiment_agent": "Market sentiment analysis from social media",
        "whale_agent": "Large wallet movement tracking",
        "funding_agent": "Perpetual funding rate analysis",
        "liquidation_agent": "Liquidation event monitoring",
        "swarm_agent": "Multi-agent coordination with neuro-consensus",
        "strategy_agent": "Custom strategy execution",
        "copybot_agent": "Copy trading from successful wallets"
    }

def generate_response(user_input, conversation_history=None):
    """
    Generate a neuro-enhanced response to user input.

    Args:
        user_input (str): User's question or message
        conversation_history (list): Previous conversation turns

    Returns:
        str: AI response
    """
    # Placeholder for AI response generation
    # In full implementation, this would use ModelFactory

    # Simple rule-based responses for demo
    user_input_lower = user_input.lower()

    if "risk" in user_input_lower or "safety" in user_input_lower:
        response = """🛡️ Risk management is crucial in NeuroFlux:

• **Risk Agent** runs first before any trading
• Circuit breakers: Max loss/gain limits, minimum balance
• Flux-adaptive risk: Adjusts limits based on market volatility
• AI confirmation for emergency closures
• Never risk more than you can afford to lose

Always enable risk_agent in your ACTIVE_AGENTS configuration!"""

    elif "trading" in user_input_lower or "buy" in user_input_lower or "sell" in user_input_lower:
        response = """📈 NeuroFlux trading capabilities:

• **Trading Agent**: Neuro-decision making with flux adaptation
• Multi-exchange support: Solana, Hyperliquid, Extended
• Confidence-based execution with risk integration
• Real-time market analysis with neural networks
• Adaptive strategies based on market conditions

Remember: All trading involves risk. Backtest strategies first!"""

    elif "backtest" in user_input_lower or "rbi" in user_input_lower:
        response = """🧪 Backtesting in NeuroFlux:

• **RBI Agent**: Automated backtesting from videos/PDFs/text
• Uses AI to extract and code trading strategies
• Tests across multiple market datasets
• Neuro-optimization for better performance
• Generates complete backtesting.py compatible code

Start with: `python src/agents/rbi_agent.py`"""

    elif "agent" in user_input_lower or "list" in user_input_lower:
        agents = get_agent_info()
        response = "🤖 NeuroFlux Agents:\n\n" + "\n".join([f"• **{name}**: {desc}" for name, desc in agents.items()])

    elif "flux" in user_input_lower or "neuro" in user_input_lower:
        response = """🧠 NeuroFlux innovations:

• **Neuro-Inspired Algorithms**: Neural networks for pattern recognition
• **Flux-Based Adaptation**: Real-time market volatility response
• **Adaptive Learning**: Strategies evolve with market conditions
• **Swarm Intelligence**: Multi-agent consensus decisions
• **Risk-Aware Execution**: Flux-adjusted position sizing

These features provide superior trading performance in dynamic markets."""

    elif "setup" in user_input_lower or "install" in user_input_lower:
        response = """⚙️ NeuroFlux setup:

1. **Environment**: `conda create -n neuro python=3.10.9`
2. **Dependencies**: `pip install -r requirements.txt`
3. **Configuration**: Copy `.env_example` to `.env`, add API keys
4. **Run**: `python src/main.py` for orchestrator

Required API keys: Anthropic (or other AI provider), exchange APIs"""

    elif "help" in user_input_lower or "what" in user_input_lower:
        response = """💡 How can I help you with NeuroFlux?

• Explain agents and their functions
• Trading strategy guidance
• Risk management advice
• Setup and configuration help
• Market analysis questions
• Backtesting assistance

Ask me anything about NeuroFlux trading system!"""

    else:
        response = f"""🤔 I understand you're asking about: "{user_input}"

As NeuroFlux's Chat Agent, I'm here to help with:
• Agent explanations and usage
• Trading strategy guidance
• Risk management advice
• Setup and configuration
• Market analysis questions

Could you please rephrase your question or ask about a specific aspect of NeuroFlux?"""

    return response

def save_conversation(user_input, response):
    """
    Save conversation for analysis and improvement.

    Args:
        user_input (str): User's message
        response (str): AI response
    """
    conversation = {
        'timestamp': datetime.now().isoformat(),
        'user_input': user_input,
        'response': response,
        'agent': 'chat_agent'
    }

    # Save latest conversation
    with open(f"{OUTPUT_DIR}/latest_conversation.json", 'w') as f:
        json.dump(conversation, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/conversation_history.jsonl", 'a') as f:
        f.write(json.dumps(conversation, default=str) + '\n')

def main():
    """Main chat loop for interactive assistance."""
    cprint("🧠 NeuroFlux Chat Agent starting...", "cyan")
    cprint("Conversational AI for trading questions and guidance", "yellow")
    cprint("Type 'quit' or 'exit' to stop", "white")

    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input("\n🗣️  You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                cprint("👋 Goodbye! Happy trading with NeuroFlux!", "green")
                break

            if not user_input:
                continue

            # Generate response
            response = generate_response(user_input, conversation_history)

            # Display response
            cprint(f"\n🤖 NeuroFlux: {response}", "green")

            # Save conversation
            save_conversation(user_input, response)

            # Add to history
            conversation_history.append({
                'user': user_input,
                'assistant': response
            })

            # Keep history manageable
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except KeyboardInterrupt:
            cprint("\n👋 Chat Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Chat Agent error: {str(e)}", "red")

if __name__ == "__main__":
    main()