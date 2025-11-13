"""
🧠 NeuroFlux's Funding Agent
Monitors funding rates with neuro-prediction.

Built with love by Nyros Veil 🚀

Analyzes funding rates across perpetual markets to predict market direction.
Uses neuro-flux adaptation for enhanced prediction accuracy.
"""

import os
import time
import json
import requests
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Output directory for funding analysis
OUTPUT_DIR = "src/data/funding_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_funding_rates():
    """
    Get current funding rates from exchanges.

    Returns:
        dict: Funding rates by symbol
    """
    funding_rates = {}

    try:
        # Hyperliquid funding rates
        if hasattr(config, 'EXCHANGE') and config.EXCHANGE == 'hyperliquid':
            # Placeholder - integrate with Hyperliquid API
            # In full implementation: query Hyperliquid perpetuals API

            mock_rates = {
                'BTC': 0.0001,   # 0.01% positive funding
                'ETH': -0.0002,  # -0.02% negative funding
                'SOL': 0.0003    # 0.03% positive funding
            }
            funding_rates.update(mock_rates)

        # Extended Exchange funding rates
        # Add API calls for other exchanges

        return funding_rates

    except Exception as e:
        cprint(f"❌ Error fetching funding rates: {str(e)}", "red")
        return {}

def analyze_funding_rates(funding_rates, market_data):
    """
    Analyze funding rates with neuro-flux prediction.

    Args:
        funding_rates (dict): Current funding rates
        market_data (dict): Market data for flux analysis

    Returns:
        dict: Analysis results with predictions
    """
    analysis = {
        'average_funding': 0.0,
        'positive_count': 0,
        'negative_count': 0,
        'flux_level': market_data.get('flux_level', 0.0),
        'funding_signal': 'NEUTRAL',
        'confidence': 0.5,
        'reasoning': '',
        'predicted_direction': 'SIDEWAYS'
    }

    if not funding_rates:
        analysis['reasoning'] = 'No funding rate data available'
        return analysis

    # Calculate statistics
    rates = list(funding_rates.values())
    analysis['average_funding'] = sum(rates) / len(rates)

    for rate in rates:
        if rate > 0:
            analysis['positive_count'] += 1
        elif rate < 0:
            analysis['negative_count'] += 1

    # Neuro-flux analysis
    flux_adjustment = analysis['flux_level'] * config.FLUX_SENSITIVITY

    # Funding rate interpretation:
    # Positive funding = longs pay shorts = bullish sentiment
    # Negative funding = shorts pay longs = bearish sentiment

    positive_ratio = analysis['positive_count'] / len(rates)

    # Adjust thresholds based on flux
    bullish_threshold = 0.6 + flux_adjustment  # Higher threshold in high flux
    bearish_threshold = 0.4 - flux_adjustment  # Lower threshold in high flux

    if positive_ratio > bullish_threshold:
        analysis['funding_signal'] = 'BULLISH'
        analysis['confidence'] = min(0.9, 0.5 + (positive_ratio - bullish_threshold))
        analysis['reasoning'] = f"High positive funding ratio {positive_ratio:.2f} > {bullish_threshold:.2f}"
        analysis['predicted_direction'] = 'UP'
    elif positive_ratio < bearish_threshold:
        analysis['funding_signal'] = 'BEARISH'
        analysis['confidence'] = min(0.9, 0.5 + (bearish_threshold - positive_ratio))
        analysis['reasoning'] = f"Low positive funding ratio {positive_ratio:.2f} < {bearish_threshold:.2f}"
        analysis['predicted_direction'] = 'DOWN'
    else:
        analysis['reasoning'] = f"Neutral funding sentiment, ratio {positive_ratio:.2f} in neutral zone"

    # Reduce confidence in high flux
    if analysis['flux_level'] > config.FLUX_SENSITIVITY:
        analysis['confidence'] *= (1 - analysis['flux_level'])
        analysis['reasoning'] += f" | Confidence reduced by {analysis['flux_level']:.1f} flux level"

    return analysis

def get_market_data():
    """
    Get current market data for flux analysis.

    Returns:
        dict: Market data
    """
    # Placeholder - integrate with market data APIs
    return {
        'flux_level': 0.3,
        'volatility': 0.05
    }

def save_funding_analysis(funding_rates, analysis):
    """
    Save funding rate analysis results.

    Args:
        funding_rates (dict): Raw funding rates
        analysis (dict): Analysis results
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'funding_rates': funding_rates,
        'analysis': analysis
    }

    # Save latest analysis
    with open(f"{OUTPUT_DIR}/latest_funding_analysis.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/funding_history.jsonl", 'a') as f:
        f.write(json.dumps(result, default=str) + '\n')

def main():
    """Main funding rate monitoring loop with neuro-prediction."""
    cprint("🧠 NeuroFlux Funding Agent starting...", "cyan")
    cprint("Monitoring funding rates with neuro-prediction", "yellow")

    while True:
        try:
            # Get funding rates
            funding_rates = get_funding_rates()

            if funding_rates:
                cprint(f"📊 Funding rates for {len(funding_rates)} symbols", "blue")
                for symbol, rate in funding_rates.items():
                    cprint(f"💰 {symbol}: {rate:.6f} ({rate*100:.4f}%)", "white")
            else:
                cprint("📊 No funding rate data available", "yellow")

            # Get market data for flux analysis
            market_data = get_market_data()

            # Analyze funding rates
            analysis = analyze_funding_rates(funding_rates, market_data)

            cprint(f"🎯 Funding Signal: {analysis['funding_signal']} | Confidence: {analysis['confidence']:.2f}", "green")
            cprint(f"📈 Predicted Direction: {analysis['predicted_direction']} | Avg Funding: {analysis['average_funding']:.6f}", "yellow")

            # Save analysis
            save_funding_analysis(funding_rates, analysis)

            cprint(f"✅ Funding analysis complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Funding Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Funding Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()