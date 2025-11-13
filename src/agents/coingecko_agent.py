"""
🧠 NeuroFlux's CoinGecko Agent
Fetches token metadata with flux enrichment.

Built with love by Nyros Veil 🚀

Retrieves comprehensive token information from CoinGecko API.
Enriches data with neuro-flux analysis for market insights.
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

# Output directory for CoinGecko data
OUTPUT_DIR = "src/data/coingecko_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CoinGecko API base URL
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

def get_coingecko_token_data(token_symbol):
    """
    Fetch token data from CoinGecko API.

    Args:
        token_symbol (str): Token symbol (e.g., 'bitcoin', 'ethereum')

    Returns:
        dict: Token metadata and market data
    """
    try:
        # First, search for the coin ID
        search_url = f"{COINGECKO_BASE_URL}/search"
        params = {'query': token_symbol}
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        search_data = response.json()

        if not search_data.get('coins'):
            return None

        # Get the first matching coin
        coin_id = search_data['coins'][0]['id']

        # Fetch detailed coin data
        coin_url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'false',
            'developer_data': 'false',
            'sparkline': 'false'
        }

        response = requests.get(coin_url, params=params, timeout=10)
        response.raise_for_status()
        coin_data = response.json()

        # Extract relevant information
        token_info = {
            'id': coin_data['id'],
            'symbol': coin_data['symbol'].upper(),
            'name': coin_data['name'],
            'market_cap_rank': coin_data.get('market_cap_rank'),
            'current_price': coin_data['market_data']['current_price'].get('usd', 0),
            'market_cap': coin_data['market_data']['market_cap'].get('usd', 0),
            'total_volume': coin_data['market_data']['total_volume'].get('usd', 0),
            'price_change_24h': coin_data['market_data']['price_change_percentage_24h'],
            'price_change_7d': coin_data['market_data']['price_change_percentage_7d'],
            'price_change_30d': coin_data['market_data']['price_change_percentage_30d'],
            'circulating_supply': coin_data['market_data']['circulating_supply'],
            'total_supply': coin_data['market_data']['total_supply'],
            'max_supply': coin_data['market_data']['max_supply'],
            'ath': coin_data['market_data']['ath'].get('usd', 0),
            'ath_date': coin_data['market_data']['ath_date'].get('usd'),
            'atl': coin_data['market_data']['atl'].get('usd', 0),
            'atl_date': coin_data['market_data']['atl_date'].get('usd'),
            'last_updated': coin_data['last_updated']
        }

        return token_info

    except requests.RequestException as e:
        cprint(f"❌ CoinGecko API error for {token_symbol}: {str(e)}", "red")
        return None
    except Exception as e:
        cprint(f"❌ Error processing CoinGecko data for {token_symbol}: {str(e)}", "red")
        return None

def analyze_token_fundamentals(token_data, market_data):
    """
    Analyze token fundamentals with neuro-flux enrichment.

    Args:
        token_data (dict): Token data from CoinGecko
        market_data (dict): Current market data

    Returns:
        dict: Enhanced analysis with flux-adjusted insights
    """
    if not token_data:
        return {
            'signal': 'UNKNOWN',
            'confidence': 0.0,
            'reasoning': 'No token data available',
            'flux_adjusted': False
        }

    analysis = {
        'market_cap_rank': token_data.get('market_cap_rank', 9999),
        'price_vs_ath': 0.0,
        'supply_health': 'UNKNOWN',
        'momentum_signal': 'NEUTRAL',
        'fundamental_score': 0.0,
        'flux_level': market_data.get('flux_level', 0.0)
    }

    # Calculate price vs ATH
    if token_data.get('ath', 0) > 0:
        analysis['price_vs_ath'] = token_data['current_price'] / token_data['ath']

    # Supply health assessment
    circulating = token_data.get('circulating_supply', 0)
    total = token_data.get('total_supply', 0)
    max_supply = token_data.get('max_supply', 0)

    if max_supply and max_supply > 0:
        supply_ratio = circulating / max_supply
        if supply_ratio < 0.5:
            analysis['supply_health'] = 'HEALTHY'
        elif supply_ratio < 0.8:
            analysis['supply_health'] = 'MODERATE'
        else:
            analysis['supply_health'] = 'DILUTED'
    elif total and total > 0:
        analysis['supply_health'] = 'UNCAPPED'

    # Momentum analysis
    price_change_7d = token_data.get('price_change_7d', 0)
    price_change_30d = token_data.get('price_change_30d', 0)

    if price_change_7d > 10 and price_change_30d > 20:
        analysis['momentum_signal'] = 'STRONG_BULLISH'
    elif price_change_7d > 5 and price_change_30d > 10:
        analysis['momentum_signal'] = 'BULLISH'
    elif price_change_7d < -10 and price_change_30d < -20:
        analysis['momentum_signal'] = 'STRONG_BEARISH'
    elif price_change_7d < -5 and price_change_30d < -10:
        analysis['momentum_signal'] = 'BEARISH'

    # Fundamental score (0-1)
    score = 0.0

    # Market cap rank bonus
    if analysis['market_cap_rank'] <= 100:
        score += 0.3
    elif analysis['market_cap_rank'] <= 500:
        score += 0.2
    elif analysis['market_cap_rank'] <= 1000:
        score += 0.1

    # Supply health bonus
    if analysis['supply_health'] == 'HEALTHY':
        score += 0.2
    elif analysis['supply_health'] == 'MODERATE':
        score += 0.1

    # Momentum bonus
    if 'BULLISH' in analysis['momentum_signal']:
        score += 0.3
    elif 'BEARISH' in analysis['momentum_signal']:
        score -= 0.2

    # Price vs ATH consideration
    if analysis['price_vs_ath'] < 0.3:
        score += 0.2  # Potential for growth

    analysis['fundamental_score'] = max(0.0, min(1.0, score))

    # Neuro-flux adjustment
    flux_adjustment = analysis['flux_level'] * config.FLUX_SENSITIVITY
    analysis['adjusted_score'] = analysis['fundamental_score'] * (1 - flux_adjustment)
    analysis['flux_adjusted'] = flux_adjustment > 0.1

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

def save_coingecko_analysis(token_data, analysis):
    """
    Save CoinGecko analysis results.

    Args:
        token_data (dict): Raw token data
        analysis (dict): Analysis results
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'token_data': token_data,
        'analysis': analysis
    }

    # Save latest analysis
    with open(f"{OUTPUT_DIR}/latest_coingecko_analysis.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/coingecko_history.jsonl", 'a') as f:
        f.write(json.dumps(result, default=str) + '\n')

def main():
    """Main CoinGecko data fetching loop with neuro-flux enrichment."""
    cprint("🧠 NeuroFlux CoinGecko Agent starting...", "cyan")
    cprint("Fetching token metadata with flux enrichment", "yellow")

    while True:
        try:
            # Get tokens to analyze
            tokens = config.get_active_tokens()
            cprint(f"📊 Fetching CoinGecko data for {len(tokens)} tokens", "blue")

            for token in tokens:
                # Convert token symbol to CoinGecko format (lowercase)
                coingecko_symbol = token.lower()

                # Fetch token data
                token_data = get_coingecko_token_data(coingecko_symbol)

                if token_data:
                    cprint(f"📈 {token}: Rank #{token_data.get('market_cap_rank', 'N/A')} | ${token_data.get('current_price', 0):.4f}", "green")

                    # Get market data for flux analysis
                    market_data = get_market_data()

                    # Analyze fundamentals
                    analysis = analyze_token_fundamentals(token_data, market_data)

                    cprint(f"🎯 Momentum: {analysis['momentum_signal']} | Supply: {analysis['supply_health']} | Score: {analysis['adjusted_score']:.2f}", "yellow")

                    # Save analysis
                    save_coingecko_analysis(token_data, analysis)
                else:
                    cprint(f"❌ No CoinGecko data found for {token}", "red")

                # Brief pause between tokens (respect API rate limits)
                time.sleep(2)

            cprint(f"✅ CoinGecko analysis complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 CoinGecko Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ CoinGecko Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()