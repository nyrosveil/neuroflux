"""
🧠 NeuroFlux's Strategy Agent
Executes user-defined strategies with flux adaptation.

Built with love by Nyros Veil 🚀

Loads and executes trading strategies from src/strategies/ directory.
Each strategy is analyzed with neuro-flux adaptation for optimal performance.
"""

import os
import time
import json
import importlib.util
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Output directory for strategy decisions
OUTPUT_DIR = "src/data/strategy_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_strategies():
    """
    Dynamically load all strategy classes from src/strategies/ directory.

    Returns:
        list: List of instantiated strategy objects
    """
    strategies = []
    strategies_dir = "src/strategies/"

    if not os.path.exists(strategies_dir):
        cprint(f"⚠️  Strategies directory {strategies_dir} not found", "yellow")
        return strategies

    # Import each .py file as a module
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            file_path = os.path.join(strategies_dir, filename)

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find strategy classes (classes that end with 'Strategy')
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        attr_name.endswith('Strategy') and
                        hasattr(attr, 'analyze')):
                        # Instantiate the strategy
                        strategy_instance = attr(config)
                        strategies.append(strategy_instance)
                        cprint(f"📈 Loaded strategy: {strategy_instance.name}", "blue")

            except Exception as e:
                cprint(f"❌ Failed to load strategy {filename}: {str(e)}", "red")

    return strategies

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
        'macd_signal': 0.015,
        'macd_hist': 0.005,
        'support': 0.95,
        'resistance': 1.05,
        'sentiment_score': 0.7,
        'flux_level': 0.3
    }

def execute_strategy_signals(strategies, token, market_data):
    """
    Execute signals from all loaded strategies for a token.

    Args:
        strategies (list): List of strategy instances
        token (str): Token symbol
        market_data (dict): Market data

    Returns:
        dict: Aggregated strategy results
    """
    results = {
        'token': token,
        'strategies': [],
        'consensus_signal': 'HOLD',
        'avg_confidence': 0.0,
        'total_strategies': len(strategies)
    }

    buy_signals = 0
    sell_signals = 0
    total_confidence = 0.0

    for strategy in strategies:
        try:
            analysis = strategy.analyze(market_data)

            # Check if strategy signals action
            should_buy = strategy.should_buy(market_data)
            should_sell = strategy.should_sell(market_data)

            strategy_result = {
                'name': strategy.name,
                'signal': analysis['signal'],
                'confidence': analysis['confidence'],
                'reasoning': analysis['reasoning'],
                'flux_adjusted': analysis['flux_adjusted'],
                'will_buy': should_buy,
                'will_sell': should_sell
            }

            results['strategies'].append(strategy_result)

            if should_buy:
                buy_signals += 1
            elif should_sell:
                sell_signals += 1

            total_confidence += analysis['confidence']

        except Exception as e:
            cprint(f"❌ Strategy {strategy.name} error: {str(e)}", "red")
            results['strategies'].append({
                'name': strategy.name,
                'error': str(e)
            })

    # Calculate consensus
    if buy_signals > sell_signals:
        results['consensus_signal'] = 'BUY'
    elif sell_signals > buy_signals:
        results['consensus_signal'] = 'SELL'

    results['avg_confidence'] = total_confidence / len(strategies) if strategies else 0.0

    return results

def execute_trade(signal, token, amount, consensus_result):
    """
    Execute trade based on strategy consensus.

    Args:
        signal (str): BUY, SELL, HOLD
        token (str): Token symbol
        amount (float): Trade amount in USD
        consensus_result (dict): Strategy consensus results

    Returns:
        dict: Trade execution result
    """
    if signal == 'HOLD':
        return {'status': 'skipped', 'reason': 'Hold consensus'}

    if consensus_result['avg_confidence'] < 0.6:
        return {'status': 'skipped', 'reason': 'Low average confidence'}

    # Placeholder - integrate with exchange APIs
    cprint(f"🧠 Executing {signal} for {token} with ${amount} based on {consensus_result['total_strategies']} strategies", "green")

    # Simulate trade execution
    result = {
        'status': 'executed',
        'signal': signal,
        'token': token,
        'amount': amount,
        'avg_confidence': consensus_result['avg_confidence'],
        'strategies_used': consensus_result['total_strategies'],
        'timestamp': datetime.now().isoformat()
    }

    return result

def save_strategy_decision(market_data, consensus_result, result):
    """
    Save strategy decision and result.

    Args:
        market_data (dict): Market data
        consensus_result (dict): Strategy consensus
        result (dict): Execution result
    """
    decision = {
        'timestamp': datetime.now().isoformat(),
        'market_data': market_data,
        'consensus': consensus_result,
        'execution': result
    }

    # Save latest decision
    with open(f"{OUTPUT_DIR}/latest_decision.json", 'w') as f:
        json.dump(decision, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/strategy_history.jsonl", 'a') as f:
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
    """Main strategy execution loop with neuro-flux adaptation."""
    cprint("🧠 NeuroFlux Strategy Agent starting...", "cyan")
    cprint("Loading and executing user-defined strategies", "yellow")

    # Load strategies once at startup
    strategies = load_strategies()
    if not strategies:
        cprint("❌ No strategies loaded - exiting", "red")
        return

    cprint(f"✅ Loaded {len(strategies)} strategies", "green")

    while True:
        try:
            # Check risk status first
            if not check_risk_status():
                cprint("🚫 Risk check failed - skipping strategy cycle", "red")
                time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)
                continue

            # Get tokens to monitor
            tokens = config.get_active_tokens()
            cprint(f"📊 Monitoring {len(tokens)} tokens with {len(strategies)} strategies", "blue")

            for token in tokens:
                # Get market data
                data = get_market_data(token)
                cprint(f"📈 {token}: ${data['price']:.4f} | RSI: {data['rsi']:.1f} | Flux: {data['flux_level']:.3f}", "white")

                # Execute all strategies
                consensus = execute_strategy_signals(strategies, token, data)

                # Determine trade amount
                trade_amount = min(config.usd_size, config.max_usd_order_size)

                # Execute trade if consensus
                result = execute_trade(consensus['consensus_signal'], token, trade_amount, consensus)

                # Save decision
                save_strategy_decision(data, consensus, result)

                # Brief pause between tokens
                time.sleep(1)

            cprint(f"✅ Strategy cycle complete - sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Strategy Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Strategy Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()