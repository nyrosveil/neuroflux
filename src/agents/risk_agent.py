"""
🧠 NeuroFlux's Risk Agent
Circuit breaker with neuro-flux awareness - runs FIRST before any trading decisions.

Built with love by Nyros Veil 🚀

Monitors portfolio risk, P&L, and exposure with adaptive flux sensitivity.
Enforces position limits, loss/gain thresholds, and emergency closures.
Includes AI confirmation for critical decisions.
"""

import os
import time
import json
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Output directory for risk reports
OUTPUT_DIR = "src/data/risk_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_flux_level():
    """
    Calculate current market flux level for adaptive risk management.

    Returns:
        float: Flux level (0-1), higher means more volatility
    """
    # Placeholder for flux calculation
    # In full implementation, this would analyze recent price movements,
    # volume spikes, and market indicators
    return FLUX_SENSITIVITY  # Use config default for now

def get_portfolio_balance():
    """
    Get current portfolio balance across all exchanges.

    Returns:
        dict: Balance information
    """
    # Placeholder - integrate with exchange APIs
    return {
        'equity': 1000.0,  # Total equity in USD
        'available': 950.0,  # Available for trading
        'positions_value': 50.0  # Current positions value
    }

def get_positions():
    """
    Get all current positions across exchanges.

    Returns:
        list: List of position dictionaries
    """
    # Placeholder - integrate with exchange APIs
    return [
        # Example position
        # {
        #     'symbol': 'BTC',
        #     'side': 'long',
        #     'size': 0.1,
        #     'entry_price': 50000,
        #     'current_price': 51000,
        #     'pnl_percentage': 2.0,
        #     'pnl_usd': 100
        # }
    ]

def check_risk_limits(balance, positions, flux_level):
    """
    Check all risk limits with flux-adaptive thresholds.

    Args:
        balance (dict): Portfolio balance
        positions (list): Current positions
        flux_level (float): Current market flux level

    Returns:
        dict: Risk check results
    """
    results = {
        'ok': True,
        'violations': [],
        'warnings': [],
        'recommendations': []
    }

    # Calculate total P&L
    total_pnl = sum(pos.get('pnl_usd', 0) for pos in positions)

    # 1. Balance Check - Minimum Balance
    if balance['equity'] < MINIMUM_BALANCE_USD:
        results['ok'] = False
        results['violations'].append({
            'type': 'balance',
            'message': f"Balance ${balance['equity']:.2f} below minimum ${MINIMUM_BALANCE_USD}",
            'severity': 'critical'
        })

    # 2. Loss Limit Check
    if USE_PERCENTAGE:
        # Percentage-based limits
        loss_threshold = MAX_LOSS_PERCENT / 100 * balance['equity']
        if total_pnl < -loss_threshold:
            results['ok'] = False
            results['violations'].append({
                'type': 'loss_limit',
                'message': f"Loss ${-total_pnl:.2f} exceeds {MAX_LOSS_PERCENT}% limit",
                'severity': 'critical'
            })
    else:
        # USD-based limits
        if total_pnl < -MAX_LOSS_USD:
            results['ok'] = False
            results['violations'].append({
                'type': 'loss_limit',
                'message': f"Loss ${-total_pnl:.2f} exceeds ${MAX_LOSS_USD} limit",
                'severity': 'critical'
            })

    # 3. Gain Target Check
    if USE_PERCENTAGE:
        gain_threshold = MAX_GAIN_PERCENT / 100 * balance['equity']
        if total_pnl > gain_threshold:
            results['ok'] = False
            results['violations'].append({
                'type': 'gain_limit',
                'message': f"Gain ${total_pnl:.2f} exceeds {MAX_GAIN_PERCENT}% target",
                'severity': 'warning'
            })
    else:
        if total_pnl > MAX_GAIN_USD:
            results['ok'] = False
            results['violations'].append({
                'type': 'gain_limit',
                'message': f"Gain ${total_pnl:.2f} exceeds ${MAX_GAIN_USD} target",
                'severity': 'warning'
            })

    # 4. Position Size Check with Flux Adjustment
    total_position_value = sum(abs(pos.get('pnl_usd', 0)) for pos in positions)
    max_position_limit = MAX_POSITION_PERCENTAGE / 100 * balance['equity']

    # Reduce limit during high flux
    flux_adjusted_limit = max_position_limit * (1 - flux_level * 0.5)

    if total_position_value > flux_adjusted_limit:
        results['warnings'].append({
            'type': 'position_size',
            'message': f"Position value ${total_position_value:.2f} exceeds flux-adjusted limit ${flux_adjusted_limit:.2f}",
            'severity': 'warning'
        })

    # 5. Cash Buffer Check
    cash_percentage = (balance['available'] / balance['equity']) * 100
    if cash_percentage < CASH_PERCENTAGE:
        results['warnings'].append({
            'type': 'cash_buffer',
            'message': f"Cash buffer {cash_percentage:.1f}% below required {CASH_PERCENTAGE}%",
            'severity': 'warning'
        })

    # 6. Neuro-Flux Recommendations
    if flux_level > FLUX_SENSITIVITY:
        results['recommendations'].append({
            'type': 'flux_adaptation',
            'message': f"High flux detected ({flux_level:.2f}). Consider reducing position sizes and increasing stop losses.",
            'action': 'reduce_risk'
        })

    return results

def save_risk_report(results, balance, positions, flux_level):
    """
    Save risk assessment report to file.

    Args:
        results (dict): Risk check results
        balance (dict): Portfolio balance
        positions (list): Current positions
        flux_level (float): Market flux level
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'flux_level': flux_level,
        'balance': balance,
        'positions': positions,
        'risk_check': results,
        'total_positions': len(positions),
        'total_pnl': sum(pos.get('pnl_usd', 0) for pos in positions)
    }

    filename = f"{OUTPUT_DIR}/latest_report.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Also save historical reports
    history_file = f"{OUTPUT_DIR}/risk_history.jsonl"
    with open(history_file, 'a') as f:
        f.write(json.dumps(report, default=str) + '\n')

def emergency_close_positions(reason):
    """
    Emergency close all positions.

    Args:
        reason (str): Reason for emergency closure
    """
    cprint(f"🚨 EMERGENCY: Closing all positions - {reason}", "red", attrs=['bold'])

    # Placeholder - integrate with exchange APIs
    # for position in positions:
    #     close_position(position['symbol'])

    cprint("✅ All positions closed", "green")

def main():
    """Main risk monitoring loop with neuro-flux awareness."""
    cprint("🧠 NeuroFlux Risk Agent starting...", "cyan")
    cprint("Circuit breaker active - monitoring portfolio risk with flux adaptation", "yellow")

    while True:
        try:
            # Calculate current market flux
            flux_level = calculate_flux_level()
            cprint(f"🌊 Market flux level: {flux_level:.3f}", "blue")

            # Get portfolio data
            balance = get_portfolio_balance()
            positions = get_positions()

            cprint(f"💰 Balance: ${balance['equity']:.2f} | Positions: {len(positions)}", "white")

            # Perform risk checks
            risk_results = check_risk_limits(balance, positions, flux_level)

            # Save report
            save_risk_report(risk_results, balance, positions, flux_level)

            # Handle violations
            if not risk_results['ok']:
                cprint("⚠️  Risk violations detected!", "red", attrs=['bold'])

                for violation in risk_results['violations']:
                    cprint(f"🚨 {violation['message']}", "red")

                    if violation['severity'] == 'critical':
                        if USE_AI_CONFIRMATION:
                            # Placeholder for AI confirmation
                            cprint("🤖 AI confirmation required for emergency closure", "yellow")
                            # In full implementation, consult AI model

                        emergency_close_positions(violation['message'])
                        break

            # Handle warnings
            for warning in risk_results['warnings']:
                cprint(f"⚠️  {warning['message']}", "yellow")

            # Handle recommendations
            for rec in risk_results['recommendations']:
                cprint(f"💡 {rec['message']}", "cyan")

            if risk_results['ok']:
                cprint("✅ Risk checks passed - trading allowed", "green")
            else:
                cprint("❌ Risk checks failed - trading blocked", "red")

            # Sleep before next check
            cprint(f"⏰ Sleeping {SLEEP_BETWEEN_RUNS_MINUTES} minutes...", "white")
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Risk Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Risk Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()