"""
🧠 NeuroFlux's Liquidation Agent
Tracks liquidation data with AI analysis.

Built with love by Nyros Veil 🚀

Monitors liquidation events across perpetual markets for market impact analysis.
Uses neuro-flux adaptation to detect liquidation cascades and sentiment shifts.
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

# Output directory for liquidation analysis
OUTPUT_DIR = "src/data/liquidation_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_liquidation_data(hours_back=24):
    """
    Get recent liquidation data from exchanges.

    Args:
        hours_back (int): Hours of history to fetch

    Returns:
        list: List of liquidation events
    """
    liquidations = []

    try:
        # Hyperliquid liquidations
        if hasattr(config, 'EXCHANGE') and config.EXCHANGE == 'hyperliquid':
            # Placeholder - integrate with Hyperliquid API
            # In full implementation: query liquidation events

            mock_liquidations = [
                {
                    'symbol': 'BTC',
                    'side': 'long',  # or 'short'
                    'size': 1000000,  # USD value
                    'price': 45000,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'symbol': 'ETH',
                    'side': 'short',
                    'size': 500000,
                    'price': 2500,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            liquidations.extend(mock_liquidations)

        # Extended Exchange liquidations
        # Add API calls for other exchanges

        return liquidations

    except Exception as e:
        cprint(f"❌ Error fetching liquidation data: {str(e)}", "red")
        return []

def analyze_liquidations(liquidations, market_data):
    """
    Analyze liquidation data with neuro-flux AI analysis.

    Args:
        liquidations (list): List of liquidation events
        market_data (dict): Market data for flux analysis

    Returns:
        dict: Analysis results with market impact assessment
    """
    analysis = {
        'total_liquidations': len(liquidations),
        'total_volume': 0,
        'long_liquidations': 0,
        'short_liquidations': 0,
        'long_volume': 0,
        'short_volume': 0,
        'flux_level': market_data.get('flux_level', 0.0),
        'liquidation_signal': 'NEUTRAL',
        'confidence': 0.5,
        'reasoning': '',
        'market_impact': 'LOW',
        'cascade_risk': 'LOW'
    }

    if not liquidations:
        analysis['reasoning'] = 'No liquidation data available'
        return analysis

    # Calculate statistics
    for liq in liquidations:
        analysis['total_volume'] += liq['size']
        if liq['side'] == 'long':
            analysis['long_liquidations'] += 1
            analysis['long_volume'] += liq['size']
        elif liq['side'] == 'short':
            analysis['short_liquidations'] += 1
            analysis['short_volume'] += liq['size']

    # Neuro-flux analysis
    flux_adjustment = analysis['flux_level'] * config.FLUX_SENSITIVITY

    # Liquidation analysis:
    # High long liquidations = bearish (forced selling)
    # High short liquidations = bullish (forced buying)

    long_ratio = analysis['long_liquidations'] / max(analysis['total_liquidations'], 1)

    # Adjust thresholds based on flux
    bearish_threshold = 0.6 + flux_adjustment  # Higher threshold in high flux
    bullish_threshold = 0.4 - flux_adjustment  # Lower threshold in high flux

    if long_ratio > bearish_threshold:
        analysis['liquidation_signal'] = 'BEARISH'
        analysis['confidence'] = min(0.9, 0.5 + (long_ratio - bearish_threshold))
        analysis['reasoning'] = f"High long liquidation ratio {long_ratio:.2f} > {bearish_threshold:.2f} - forced selling pressure"
    elif long_ratio < bullish_threshold:
        analysis['liquidation_signal'] = 'BULLISH'
        analysis['confidence'] = min(0.9, 0.5 + (bullish_threshold - long_ratio))
        analysis['reasoning'] = f"Low long liquidation ratio {long_ratio:.2f} < {bullish_threshold:.2f} - forced buying pressure"
    else:
        analysis['reasoning'] = f"Balanced liquidations, ratio {long_ratio:.2f} in neutral zone"

    # Assess market impact
    if analysis['total_volume'] > 10000000:  # $10M+
        analysis['market_impact'] = 'HIGH'
    elif analysis['total_volume'] > 1000000:  # $1M+
        analysis['market_impact'] = 'MEDIUM'

    # Cascade risk assessment
    if analysis['total_liquidations'] > 50 and analysis['flux_level'] > config.FLUX_SENSITIVITY:
        analysis['cascade_risk'] = 'HIGH'
    elif analysis['total_liquidations'] > 20:
        analysis['cascade_risk'] = 'MEDIUM'

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

def save_liquidation_analysis(liquidations, analysis):
    """
    Save liquidation analysis results.

    Args:
        liquidations (list): Raw liquidation data
        analysis (dict): Analysis results
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'liquidations': liquidations,
        'analysis': analysis
    }

    # Save latest analysis
    with open(f"{OUTPUT_DIR}/latest_liquidation_analysis.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/liquidation_history.jsonl", 'a') as f:
        f.write(json.dumps(result, default=str) + '\n')

def main():
    """Main liquidation monitoring loop with AI analysis."""
    cprint("🧠 NeuroFlux Liquidation Agent starting...", "cyan")
    cprint("Tracking liquidation data with AI analysis", "yellow")

    while True:
        try:
            # Get liquidation data
            liquidations = get_liquidation_data(hours_back=24)

            if liquidations:
                cprint(f"💥 Found {len(liquidations)} liquidation events", "red")
                total_volume = sum(l['size'] for l in liquidations)
                cprint(f"💰 Total liquidated: ${total_volume:,.0f}", "yellow")
            else:
                cprint("✅ No significant liquidations detected", "green")

            # Get market data for flux analysis
            market_data = get_market_data()

            # Analyze liquidations
            analysis = analyze_liquidations(liquidations, market_data)

            cprint(f"🎯 Liquidation Signal: {analysis['liquidation_signal']} | Confidence: {analysis['confidence']:.2f}", "green")
            cprint(f"📊 Impact: {analysis['market_impact']} | Cascade Risk: {analysis['cascade_risk']}", "yellow")
            cprint(f"📈 Long Liqs: {analysis['long_liquidations']} (${analysis['long_volume']:,.0f}) | Short Liqs: {analysis['short_liquidations']} (${analysis['short_volume']:,.0f})", "blue")

            # Save analysis
            save_liquidation_analysis(liquidations, analysis)

            cprint(f"✅ Liquidation analysis complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Liquidation Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Liquidation Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()