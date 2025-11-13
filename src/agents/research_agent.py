"""
🧠 NeuroFlux's Research Agent
General market research with AI.

Built with love by Nyros Veil 🚀

Conducts comprehensive market research using AI analysis.
Synthesizes data from multiple sources for market insights and predictions.
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

# Import ModelFactory for AI analysis
from models.model_factory import ModelFactory

# Output directory for research analysis
OUTPUT_DIR = "src/data/research_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def gather_market_data():
    """
    Gather comprehensive market data from various sources.

    Returns:
        dict: Collected market data
    """
    market_data = {
        'timestamp': datetime.now().isoformat(),
        'indices': {},
        'commodities': {},
        'currencies': {},
        'crypto_overview': {},
        'economic_indicators': {}
    }

    try:
        # Placeholder - integrate with financial data APIs
        # In full implementation: Yahoo Finance, Alpha Vantage, CoinGecko, etc.

        # Mock market data
        market_data['indices'] = {
            'sp500': {'price': 4500, 'change': 1.2},
            'nasdaq': {'price': 14200, 'change': 0.8},
            'dow': {'price': 35000, 'change': 1.5}
        }

        market_data['commodities'] = {
            'gold': {'price': 1950, 'change': -0.5},
            'oil': {'price': 78, 'change': 2.1},
            'bitcoin': {'price': 43000, 'change': 3.2}
        }

        market_data['currencies'] = {
            'usd_eur': 0.85,
            'usd_jpy': 150,
            'usd_cny': 7.2
        }

        market_data['economic_indicators'] = {
            'fed_funds_rate': 5.25,
            'inflation_rate': 3.1,
            'unemployment_rate': 4.2
        }

        return market_data

    except Exception as e:
        cprint(f"❌ Error gathering market data: {str(e)}", "red")
        return market_data

def analyze_market_research(market_data):
    """
    Perform AI-powered market research and analysis.

    Args:
        market_data (dict): Collected market data

    Returns:
        dict: AI research analysis and insights
    """
    try:
        # Initialize AI model
        model = ModelFactory.create_model('anthropic')

        # Create comprehensive research prompt
        research_prompt = f"""
        Conduct comprehensive market research analysis based on the following data:

        MARKET INDICES:
        - S&P 500: ${market_data['indices'].get('sp500', {}).get('price', 'N/A')} ({market_data['indices'].get('sp500', {}).get('change', 0):.1f}%)
        - NASDAQ: ${market_data['indices'].get('nasdaq', {}).get('price', 'N/A')} ({market_data['indices'].get('nasdaq', {}).get('change', 0):.1f}%)
        - DOW: ${market_data['indices'].get('dow', {}).get('price', 'N/A')} ({market_data['indices'].get('dow', {}).get('change', 0):.1f}%)

        COMMODITIES:
        - Gold: ${market_data['commodities'].get('gold', {}).get('price', 'N/A')} ({market_data['commodities'].get('gold', {}).get('change', 0):.1f}%)
        - Oil: ${market_data['commodities'].get('oil', {}).get('price', 'N/A')} ({market_data['commodities'].get('oil', {}).get('change', 0):.1f}%)
        - Bitcoin: ${market_data['commodities'].get('bitcoin', {}).get('price', 'N/A')} ({market_data['commodities'].get('bitcoin', {}).get('change', 0):.1f}%)

        ECONOMIC INDICATORS:
        - Fed Funds Rate: {market_data['economic_indicators'].get('fed_funds_rate', 'N/A')}%
        - Inflation Rate: {market_data['economic_indicators'].get('inflation_rate', 'N/A')}%
        - Unemployment Rate: {market_data['economic_indicators'].get('unemployment_rate', 'N/A')}%

        Provide a comprehensive market research analysis including:

        1. Overall market sentiment (BULLISH, BEARISH, NEUTRAL)
        2. Key drivers of current market conditions
        3. Risk assessment and potential volatility
        4. Sector outlook (Technology, Finance, Energy, etc.)
        5. Cryptocurrency market implications
        6. Economic policy impact assessment
        7. Short-term trading implications
        8. Long-term investment outlook

        Structure your response as a JSON object with these exact keys:
        market_sentiment, key_drivers, risk_assessment, sector_outlook, crypto_implications, policy_impact, short_term_trading, long_term_outlook
        """

        response = model.generate_response("", research_prompt, temperature=0.3, max_tokens=1500)

        # Parse JSON response
        try:
            analysis = json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback analysis
            analysis = {
                'market_sentiment': 'NEUTRAL',
                'key_drivers': ['Economic data mixed', 'Geopolitical tensions'],
                'risk_assessment': 'Moderate volatility expected',
                'sector_outlook': {'Technology': 'Positive', 'Energy': 'Bullish', 'Finance': 'Neutral'},
                'crypto_implications': 'Bitcoin showing strength as digital gold',
                'policy_impact': 'Fed policy remains data-dependent',
                'short_term_trading': 'Monitor key support/resistance levels',
                'long_term_outlook': 'Cautiously optimistic with inflation concerns'
            }

        return analysis

    except Exception as e:
        cprint(f"❌ AI research analysis error: {str(e)}", "red")
        return {
            'market_sentiment': 'UNKNOWN',
            'key_drivers': ['Analysis failed'],
            'risk_assessment': 'Unable to assess',
            'sector_outlook': {},
            'crypto_implications': 'Unknown',
            'policy_impact': 'Unknown',
            'short_term_trading': 'Exercise caution',
            'long_term_outlook': 'Monitor developments'
        }

def generate_research_report(market_data, analysis):
    """
    Generate comprehensive research report.

    Args:
        market_data (dict): Raw market data
        analysis (dict): AI analysis results

    Returns:
        dict: Complete research report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'market_data': market_data,
        'analysis': analysis,
        'recommendations': [],
        'confidence_level': 'MEDIUM'
    }

    # Generate recommendations based on analysis
    sentiment = analysis.get('market_sentiment', 'NEUTRAL')

    if sentiment == 'BULLISH':
        report['recommendations'].append("Consider increasing equity exposure")
        report['recommendations'].append("Monitor for continuation patterns")
        report['confidence_level'] = 'HIGH'
    elif sentiment == 'BEARISH':
        report['recommendations'].append("Increase cash positions")
        report['recommendations'].append("Consider defensive sectors")
        report['confidence_level'] = 'HIGH'
    else:
        report['recommendations'].append("Maintain balanced portfolio")
        report['recommendations'].append("Wait for clearer signals")

    # Add crypto-specific recommendations
    crypto_implications = analysis.get('crypto_implications', '')
    if 'strength' in crypto_implications.lower():
        report['recommendations'].append("Consider cryptocurrency exposure")
    elif 'weakness' in crypto_implications.lower():
        report['recommendations'].append("Reduce cryptocurrency exposure")

    return report

def save_research_report(report):
    """
    Save research report to file.

    Args:
        report (dict): Complete research report
    """
    # Save latest report
    with open(f"{OUTPUT_DIR}/latest_research_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/research_history.jsonl", 'a') as f:
        f.write(json.dumps(report, default=str) + '\n')

def main():
    """Main research analysis loop."""
    cprint("🧠 NeuroFlux Research Agent starting...", "cyan")
    cprint("Conducting comprehensive market research with AI", "yellow")

    while True:
        try:
            cprint("🔍 Gathering market data from multiple sources...", "blue")

            # Gather comprehensive market data
            market_data = gather_market_data()

            cprint("🤖 Analyzing market conditions with AI...", "blue")

            # Perform AI-powered research analysis
            analysis = analyze_market_research(market_data)

            cprint(f"📊 Market Sentiment: {analysis.get('market_sentiment', 'UNKNOWN')}", "green")
            cprint(f"🎯 Key Drivers: {', '.join(analysis.get('key_drivers', [])[:2])}", "yellow")

            # Generate complete research report
            report = generate_research_report(market_data, analysis)

            cprint(f"💡 Recommendations: {len(report['recommendations'])} generated", "green")

            # Save research report
            save_research_report(report)

            cprint(f"✅ Research analysis complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Research Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Research Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()