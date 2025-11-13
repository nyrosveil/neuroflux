"""
🧠 NeuroFlux's Swarm Agent
Multi-agent coordination with neuro-consensus decision making.

Built with love by Nyros Veil 🚀

Orchestrates multiple AI agents simultaneously for collective intelligence.
Uses neuro-inspired consensus algorithms for superior decision making.
Coordinates trading decisions across different strategies and timeframes.
"""

import os
import json
import time
import threading
import queue
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Output directory for swarm coordination
OUTPUT_DIR = "src/data/swarm_agent/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AgentWorker(threading.Thread):
    """
    Worker thread for running individual agents in parallel.
    """

    def __init__(self, agent_name, task_queue, result_queue):
        super().__init__()
        self.agent_name = agent_name
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.daemon = True

    def run(self):
        """Execute agent tasks from the queue."""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:  # Sentinel value to stop
                    break

                # Simulate agent execution (placeholder)
                result = self.simulate_agent_execution(task)
                self.result_queue.put((self.agent_name, result))

                self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put((self.agent_name, {'error': str(e)}))

    def simulate_agent_execution(self, task):
        """
        Simulate agent execution for the given task.

        Args:
            task (dict): Task parameters

        Returns:
            dict: Agent response
        """
        # Placeholder for actual agent integration
        # In full implementation, this would call actual agent functions

        agent_responses = {
            'trading_agent': {
                'signal': 'BUY',
                'confidence': 0.75,
                'reasoning': 'Technical indicators show bullish momentum',
                'token': task.get('token', 'BTC'),
                'timeframe': task.get('timeframe', '1H')
            },
            'sentiment_agent': {
                'sentiment_score': 0.2,
                'category': 'bullish',
                'confidence': 0.8,
                'sources': ['twitter', 'news', 'reddit']
            },
            'risk_agent': {
                'risk_ok': True,
                'current_pnl': 2.5,
                'max_drawdown': 8.2,
                'recommendation': 'Continue trading'
            },
            'rbi_agent': {
                'strategy_score': 85,
                'backtest_return': 24.5,
                'sharpe_ratio': 1.8,
                'recommendation': 'Implement strategy'
            }
        }

        return agent_responses.get(self.agent_name, {'status': 'unknown_agent'})

def neuro_consensus_algorithm(agent_responses, flux_level):
    """
    Neuro-inspired consensus algorithm for aggregating agent opinions.

    Args:
        agent_responses (dict): Responses from all agents
        flux_level (float): Current market flux level

    Returns:
        dict: Consensus decision
    """
    # Extract signals and confidence scores
    signals = []
    confidences = []
    reasoning = []

    for agent, response in agent_responses.items():
        if 'signal' in response:
            signals.append(response['signal'])
            confidences.append(response.get('confidence', 0.5))
            reasoning.append(response.get('reasoning', ''))

    # Calculate consensus
    if not signals:
        return {
            'consensus_signal': 'HOLD',
            'confidence': 0.5,
            'agreement_level': 0.0,
            'reasoning': 'No clear signals from agents'
        }

    # Count signal frequencies
    signal_counts = {}
    for signal in signals:
        signal_counts[signal] = signal_counts.get(signal, 0) + 1

    # Find majority signal
    majority_signal = max(signal_counts, key=signal_counts.get)
    agreement_level = signal_counts[majority_signal] / len(signals)

    # Calculate weighted confidence
    total_confidence = sum(confidences)
    avg_confidence = total_confidence / len(confidences) if confidences else 0.5

    # Apply flux adjustment
    flux_adjusted_confidence = avg_confidence * (1 - flux_level * 0.2)

    # Neuro-inspired decision threshold
    decision_threshold = 0.6 - (flux_level * 0.1)  # Lower threshold in high flux

    final_signal = majority_signal if flux_adjusted_confidence > decision_threshold else 'HOLD'

    consensus = {
        'consensus_signal': final_signal,
        'confidence': round(flux_adjusted_confidence, 3),
        'agreement_level': round(agreement_level, 3),
        'flux_level': flux_level,
        'agent_count': len(agent_responses),
        'reasoning': '; '.join(reasoning[:3]),  # Top 3 reasonings
        'individual_responses': agent_responses
    }

    return consensus

def coordinate_swarm_agents(active_agents, task_parameters):
    """
    Coordinate multiple agents in parallel execution.

    Args:
        active_agents (list): List of agent names to coordinate
        task_parameters (dict): Parameters for the coordination task

    Returns:
        dict: Swarm coordination results
    """
    # Create queues for task distribution and result collection
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    # Create worker threads
    workers = []
    for agent_name in active_agents:
        worker = AgentWorker(agent_name, task_queue, result_queue)
        workers.append(worker)
        worker.start()

    # Distribute tasks
    for agent in active_agents:
        task_queue.put(task_parameters)

    # Signal workers to stop after tasks
    for _ in workers:
        task_queue.put(None)

    # Collect results with timeout
    agent_responses = {}
    timeout = 30  # 30 second timeout

    for _ in range(len(active_agents)):
        try:
            agent_name, result = result_queue.get(timeout=timeout)
            agent_responses[agent_name] = result
        except queue.Empty:
            cprint(f"⚠️  Timeout waiting for agent response", "yellow")

    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=5)

    return agent_responses

def execute_swarm_decision(consensus, token):
    """
    Execute the swarm consensus decision.

    Args:
        consensus (dict): Consensus decision
        token (str): Token to trade

    Returns:
        dict: Execution result
    """
    if consensus['consensus_signal'] == 'HOLD':
        return {'action': 'HOLD', 'reason': 'Consensus indicates hold'}

    # Placeholder for actual trade execution
    # In full implementation, this would integrate with trading_agent

    execution = {
        'action': consensus['consensus_signal'],
        'token': token,
        'confidence': consensus['confidence'],
        'agreement_level': consensus['agreement_level'],
        'executed': True,
        'timestamp': datetime.now().isoformat()
    }

    cprint(f"🐝 Swarm executed {execution['action']} for {token} with {execution['confidence']:.2f} confidence", "green")

    return execution

def save_swarm_results(task_params, agent_responses, consensus, execution):
    """
    Save swarm coordination results.

    Args:
        task_params (dict): Original task parameters
        agent_responses (dict): Individual agent responses
        consensus (dict): Consensus decision
        execution (dict): Execution result
    """
    swarm_result = {
        'timestamp': datetime.now().isoformat(),
        'task_parameters': task_params,
        'agent_responses': agent_responses,
        'consensus': consensus,
        'execution': execution,
        'swarm_size': len(agent_responses)
    }

    # Save latest swarm result
    with open(f"{OUTPUT_DIR}/latest_swarm.json", 'w') as f:
        json.dump(swarm_result, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/swarm_history.jsonl", 'a') as f:
        f.write(json.dumps(swarm_result, default=str) + '\n')

def display_swarm_results(consensus, agent_responses):
    """
    Display swarm coordination results in formatted output.

    Args:
        consensus (dict): Consensus decision
        agent_responses (dict): Individual agent responses
    """
    cprint("\\n🐝 SWARM CONSENSUS RESULTS", "cyan", attrs=['bold'])
    cprint(f"Signal: {consensus['consensus_signal']}", "white")
    cprint(f"Confidence: {consensus['confidence']:.3f}", "white")
    cprint(f"Agreement: {consensus['agreement_level']:.3f}", "white")
    cprint(f"Flux Level: {consensus['flux_level']:.3f}", "white")
    cprint(f"Agents: {consensus['agent_count']}", "white")

    cprint("\\n🤖 Individual Agent Responses:", "blue")
    for agent, response in agent_responses.items():
        if 'signal' in response:
            color = 'green' if response['signal'] == 'BUY' else 'red' if response['signal'] == 'SELL' else 'white'
            cprint(f"  {agent}: {response['signal']} ({response.get('confidence', 0):.2f})", color)
        else:
            cprint(f"  {agent}: {response}", "white")

def main():
    """Main swarm coordination loop with neuro-consensus."""
    cprint("🧠 NeuroFlux Swarm Agent starting...", "cyan")
    cprint("Multi-agent coordination with neuro-consensus", "yellow")

    # Define swarm agents (can be configured)
    swarm_agents = ['trading_agent', 'sentiment_agent', 'risk_agent', 'rbi_agent']

    while True:
        try:
            # Get tokens to analyze
            tokens = get_active_tokens()
            if not tokens:
                tokens = ['BTC']  # Default for testing

            cprint(f"🐝 Coordinating {len(swarm_agents)} agents for {len(tokens)} tokens", "blue")

            for token in tokens:
                # Define task parameters
                task_params = {
                    'token': token,
                    'timeframe': '1H',
                    'flux_level': FLUX_SENSITIVITY
                }

                # Coordinate swarm agents
                cprint(f"🚀 Launching swarm for {token}...", "blue")
                agent_responses = coordinate_swarm_agents(swarm_agents, task_params)

                # Apply neuro-consensus algorithm
                flux_level = FLUX_SENSITIVITY  # Placeholder
                consensus = neuro_consensus_algorithm(agent_responses, flux_level)

                # Display results
                display_swarm_results(consensus, agent_responses)

                # Execute decision
                execution = execute_swarm_decision(consensus, token)

                # Save results
                save_swarm_results(task_params, agent_responses, consensus, execution)

                # Brief pause between tokens
                time.sleep(3)

            cprint(f"✅ Swarm cycle complete - sleeping {SLEEP_BETWEEN_RUNS_MINUTES} minutes", "green")
            time.sleep(SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\\n👋 Swarm Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Swarm Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()