"""
🧠 NeuroFlux's Whale Agent
Tracks large wallet movements with flux detection.

Built with love by Nyros Veil 🚀

Monitors whale wallets and large transactions for market impact analysis.
Uses neuro-flux adaptation to detect unusual activity patterns.
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

# Output directory for whale tracking
OUTPUT_DIR = "src/data/whale_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known whale wallets to monitor (add more as needed)
WHALE_WALLETS = [
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC mint authority
    "So11111111111111111111111111111111111111111",   # SOL wrapped
    # Add more whale addresses here
]

def get_whale_transactions(wallet_address, days_back=1):
    """
    Get recent transactions for a whale wallet.

    Args:
        wallet_address (str): Wallet address to monitor
        days_back (int): Days of history to fetch

    Returns:
        list: List of large transactions
    """
    # Placeholder - integrate with Solana RPC or BirdEye API
    # In full implementation, this would query blockchain for transactions

    try:
        # Simulate API call to get transactions
        # For Solana: use getProgramAccounts or getTokenAccounts
        # For other chains: use appropriate APIs

        # Mock large transactions for demonstration
        mock_transactions = [
            {
                'signature': 'mock_tx_1',
                'timestamp': datetime.now().isoformat(),
                'amount': 1000000,  # Large amount in USD
                'token': 'SOL',
                'type': 'transfer',
                'direction': 'outgoing',
                'wallet': wallet_address
            },
            {
                'signature': 'mock_tx_2',
                'timestamp': datetime.now().isoformat(),
                'amount': 500000,
                'token': 'USDC',
                'type': 'swap',
                'direction': 'incoming',
                'wallet': wallet_address
            }
        ]

        # Filter for large transactions (> $100k)
        large_transactions = [tx for tx in mock_transactions if tx['amount'] > 100000]

        return large_transactions

    except Exception as e:
        cprint(f"❌ Error fetching whale transactions for {wallet_address}: {str(e)}", "red")
        return []

def analyze_whale_activity(transactions, market_data):
    """
    Analyze whale activity for market impact using neuro-flux detection.

    Args:
        transactions (list): List of whale transactions
        market_data (dict): Current market data

    Returns:
        dict: Analysis results with flux-adjusted signals
    """
    analysis = {
        'total_volume': 0,
        'transaction_count': len(transactions),
        'large_buy_volume': 0,
        'large_sell_volume': 0,
        'flux_level': market_data.get('flux_level', 0.0),
        'whale_signal': 'NEUTRAL',
        'confidence': 0.5,
        'reasoning': '',
        'market_impact': 'LOW'
    }

    # Calculate volumes
    for tx in transactions:
        analysis['total_volume'] += tx['amount']
        if tx['direction'] == 'incoming':
            analysis['large_buy_volume'] += tx['amount']
        elif tx['direction'] == 'outgoing':
            analysis['large_sell_volume'] += tx['amount']

    # Neuro-flux analysis
    buy_sell_ratio = analysis['large_buy_volume'] / max(analysis['large_sell_volume'], 1)

    # Adjust thresholds based on flux
    flux_adjustment = analysis['flux_level'] * config.FLUX_SENSITIVITY
    buy_threshold = 2.0 + flux_adjustment  # Higher threshold in high flux
    sell_threshold = 0.5 - flux_adjustment  # Lower threshold in high flux

    if buy_sell_ratio > buy_threshold:
        analysis['whale_signal'] = 'BULLISH'
        analysis['confidence'] = min(0.9, 0.5 + (buy_sell_ratio - buy_threshold) / 2)
        analysis['reasoning'] = f"High buy volume ratio {buy_sell_ratio:.2f} > {buy_threshold:.2f}"
        analysis['market_impact'] = 'HIGH' if analysis['total_volume'] > 5000000 else 'MEDIUM'
    elif buy_sell_ratio < sell_threshold:
        analysis['whale_signal'] = 'BEARISH'
        analysis['confidence'] = min(0.9, 0.5 + (sell_threshold - buy_sell_ratio) / 2)
        analysis['reasoning'] = f"High sell volume ratio {buy_sell_ratio:.2f} < {sell_threshold:.2f}"
        analysis['market_impact'] = 'HIGH' if analysis['total_volume'] > 5000000 else 'MEDIUM'
    else:
        analysis['reasoning'] = f"Balanced whale activity, ratio {buy_sell_ratio:.2f} in neutral zone"

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
        'price': 1.0,
        'volume_24h': 1000000,
        'flux_level': 0.3,
        'volatility': 0.05
    }

def save_whale_analysis(whale_data, analysis):
    """
    Save whale analysis results.

    Args:
        whale_data (dict): Whale transaction data
        analysis (dict): Analysis results
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'whale_data': whale_data,
        'analysis': analysis
    }

    # Save latest analysis
    with open(f"{OUTPUT_DIR}/latest_whale_analysis.json", 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/whale_history.jsonl", 'a') as f:
        f.write(json.dumps(result, default=str) + '\n')

def main():
    """Main whale monitoring loop with neuro-flux detection."""
    cprint("🧠 NeuroFlux Whale Agent starting...", "cyan")
    cprint("Tracking large wallet movements with flux detection", "yellow")

    while True:
        try:
            all_whale_transactions = []
            cprint(f"📊 Monitoring {len(WHALE_WALLETS)} whale wallets", "blue")

            # Get market data for flux analysis
            market_data = get_market_data()

            # Monitor each whale wallet
            for wallet in WHALE_WALLETS:
                transactions = get_whale_transactions(wallet, days_back=1)
                if transactions:
                    cprint(f"🐋 {wallet[:8]}...: {len(transactions)} large transactions", "white")
                    all_whale_transactions.extend(transactions)
                else:
                    cprint(f"🐋 {wallet[:8]}...: No large transactions", "white")

            # Analyze combined whale activity
            if all_whale_transactions:
                analysis = analyze_whale_activity(all_whale_transactions, market_data)

                cprint(f"📈 Whale Signal: {analysis['whale_signal']} | Confidence: {analysis['confidence']:.2f} | Impact: {analysis['market_impact']}", "green")
                cprint(f"💰 Total Volume: ${analysis['total_volume']:,.0f} | Buy: ${analysis['large_buy_volume']:,.0f} | Sell: ${analysis['large_sell_volume']:,.0f}", "yellow")

                # Save analysis
                whale_data = {
                    'wallets_monitored': len(WHALE_WALLETS),
                    'transactions_found': len(all_whale_transactions),
                    'total_volume': analysis['total_volume']
                }
                save_whale_analysis(whale_data, analysis)
            else:
                cprint("🐋 No significant whale activity detected", "blue")

            cprint(f"✅ Whale monitoring cycle complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Whale Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Whale Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()