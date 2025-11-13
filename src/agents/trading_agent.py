"""
🧠 NeuroFlux's Trading Agent
Core trading agent with neuro-decision making and flux-adaptive strategies.

Built with love by Nyros Veil 🚀

Makes buy/sell decisions based on market analysis with neural network enhancements.
Supports multi-exchange trading with real-time flux monitoring.
Integrates with risk_agent.py for safety checks.
"""

import os
import time
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

# Output directory for trading decisions
OUTPUT_DIR = "src/data/trading_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_market_data(token_address=None):
    """
    Get market data for analysis.

    Args:
        token_address (str): Token address or symbol

    Returns:
        dict: Market data including price, volume, indicators
    """
    # Placeholder - integrate with exchange APIs
    return {
        'price': 1.0,
        'volume_24h': 1000000,
        'price_change_24h': 5.2,
        'rsi': 65,
        'macd': 0.02,
        'support': 0.95,
        'resistance': 1.05,
        'sentiment_score': 0.7,
        'flux_level': 0.3
    }

def analyze_market_neuro(data):
    """
    Neuro-enhanced market analysis using flux-adaptive AI.

    Args:
        data (dict): Market data

    Returns:
        dict: Analysis results with confidence scores
    """
    # Placeholder for neuro-analysis
    # In full implementation, this would use neural networks
    # and flux-adaptive AI models

    analysis = {
        'signal': 'HOLD',  # BUY, SELL, HOLD
        'confidence': 0.0,
        'reasoning': '',
        'flux_adjustment': 1.0
    }

    # Basic analysis logic (placeholder)
    if data['rsi'] < 30 and data['flux_level'] < FLUX_SENSITIVITY:
        analysis['signal'] = 'BUY'
        analysis['confidence'] = 0.8
        analysis['reasoning'] = 'Oversold with low flux - good entry'
    elif data['rsi'] > 70 and data['flux_level'] < FLUX_SENSITIVITY:
        analysis['signal'] = 'SELL'
        analysis['confidence'] = 0.8
        analysis['reasoning'] = 'Overbought with low flux - good exit'
    else:
        analysis['signal'] = 'HOLD'
        analysis['confidence'] = 0.5
        analysis['reasoning'] = 'Neutral conditions or high flux'

    # Flux adjustment
    if data['flux_level'] > FLUX_SENSITIVITY:
        analysis['confidence'] *= (1 - data['flux_level'])
        analysis['flux_adjustment'] = 0.5  # Reduce confidence in high flux

    return analysis

def execute_trade(signal, token, amount, analysis):
    """
    Execute trade with neuro-flux confirmation.

    Args:
        signal (str): BUY, SELL, HOLD
        token (str): Token symbol
        amount (float): Trade amount in USD
        analysis (dict): Analysis results

    Returns:
        dict: Trade execution result
    """
    if signal == 'HOLD':
        return {'status': 'skipped', 'reason': 'Hold signal'}

    if analysis['confidence'] < 0.6:
        return {'status': 'skipped', 'reason': 'Low confidence'}

    # Placeholder - integrate with exchange APIs
    cprint(f"🧠 Executing {signal} for {token} with ${amount} at {analysis['confidence']:.2f} confidence", "green")

    # Simulate trade execution
    result = {
        'status': 'executed',
        'signal': signal,
        'token': token,
        'amount': amount,
        'confidence': analysis['confidence'],
        'timestamp': datetime.now().isoformat(),
        'flux_adjusted': analysis['flux_adjustment'] < 1.0
    }

    return result

def save_decision(data, analysis, result):
    """
    Save trading decision and result.

    Args:
        data (dict): Market data
        analysis (dict): Analysis results
        result (dict): Execution result
    """
    decision = {
        'timestamp': datetime.now().isoformat(),
        'market_data': data,
        'analysis': analysis,
        'execution': result
    }

    # Save latest decision
    with open(f"{OUTPUT_DIR}/latest_decision.json", 'w') as f:
        json.dump(decision, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/trading_history.jsonl", 'a') as f:
        f.write(json.dumps(decision, default=str) + '\n')

def check_risk_status():
    """
    Check risk status from risk_agent before trading.

    Returns:
        bool: True if trading allowed
    """
    try:
        with open("src/data/risk_agent/latest_report.json", 'r') as f:
            report = json.load(f)
            return report['risk_check']['ok']
    except FileNotFoundError:
        cprint("⚠️  Risk report not found - assuming safe", "yellow")
        return True

def main():
    """Main trading loop with neuro-flux decision making."""
    cprint("🧠 NeuroFlux Trading Agent starting...", "cyan")
    cprint("Neuro-decision making with flux-adaptive strategies", "yellow")

    while True:
        try:
            # Check risk status first
            if not check_risk_status():
                cprint("🚫 Risk check failed - skipping trading cycle", "red")
                time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)
                continue

            # Get tokens to monitor
            tokens = get_active_tokens()
            cprint(f"📊 Monitoring {len(tokens)} tokens", "blue")

            for token in tokens:
                # Get market data
                data = get_market_data(token)
                cprint(f"📈 {token}: ${data['price']:.4f} | RSI: {data['rsi']:.1f} | Flux: {data['flux_level']:.3f}", "white")

                # Neuro-enhanced analysis
                analysis = analyze_market_neuro(data)

                # Determine trade amount
                trade_amount = min(usd_size, max_usd_order_size)

                # Execute trade if signal
                result = execute_trade(analysis['signal'], token, trade_amount, analysis)

                # Save decision
                save_decision(data, analysis, result)

                # Brief pause between tokens
                time.sleep(1)

            cprint(f"✅ Trading cycle complete - sleeping {SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Trading Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Trading Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()