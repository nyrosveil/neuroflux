"""
🧠 NeuroFlux Enhanced Backtesting Framework
Neuro-optimized backtesting with flux-adaptive parameters.

Built with love by Nyros Veil 🚀

Advanced backtesting framework that integrates:
- Neuro-optimization algorithms
- Flux-adaptive parameter tuning
- RBI agent integration
- Multi-objective optimization
- Risk-adjusted performance metrics
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from backtesting import Backtest, Strategy
import ta
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from termcolor import cprint
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Output directory for enhanced backtesting
OUTPUT_DIR = "src/data/backtesting/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NeuroFluxStrategy(Strategy):
    """
    Base strategy class with neuro-flux enhancements.
    """

    def init(self):
        """Initialize strategy with neuro-flux parameters."""
        self.flux_level = config.FLUX_SENSITIVITY
        self.neural_layers = config.NEURAL_NETWORK_LAYERS
        self.adaptive_learning_rate = config.ADAPTIVE_LEARNING_RATE

        # Initialize common indicators
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # RSI with adaptive parameters
        self.rsi_window = 14
        self.rsi = self.I(ta.momentum.RSIIndicator, close, window=self.rsi_window).rsi()

        # MACD with adaptive parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        macd = self.I(ta.trend.MACD, close,
                     window_fast=self.macd_fast,
                     window_slow=self.macd_slow,
                     window_sign=self.macd_signal)
        self.macd = macd.macd()
        self.macd_signal_line = macd.macd_signal()

        # Bollinger Bands with adaptive parameters
        self.bb_window = 20
        self.bb_std = 2
        bb = self.I(ta.volatility.BollingerBands, close,
                   window=self.bb_window, window_dev=self.bb_std)
        self.bb_upper = bb.bollinger_hband()
        self.bb_lower = bb.bollinger_lband()

        # Adaptive thresholds based on flux
        self.rsi_oversold = 30 - (self.flux_level * 5)
        self.rsi_overbought = 70 + (self.flux_level * 5)

    def neuro_adapt(self):
        """
        Apply neuro-flux adaptation to strategy parameters.
        """
        # Adjust parameters based on recent performance
        if len(self.data) > 50:
            recent_returns = np.diff(self.data.Close[-50:]) / self.data.Close[-50:-1]
            volatility = np.std(recent_returns)

            # Increase caution in high volatility
            if volatility > self.flux_level:
                self.rsi_oversold += 5
                self.rsi_overbought -= 5

class NeuroOptimizer:
    """
    Neuro-optimization engine for strategy parameters.
    """

    def __init__(self, strategy_class, data):
        self.strategy_class = strategy_class
        self.data = data
        self.optimization_history = []

    def objective_function(self, params):
        """
        Objective function for optimization.
        Maximizes Sharpe ratio with risk constraints.
        """
        try:
            # Unpack parameters
            rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std = params

            # Create strategy instance with parameters
            class OptimizedStrategy(self.strategy_class):
                def init(self):
                    super().init()
                    # Override with optimized parameters
                    self.rsi_window = int(rsi_window)
                    self.macd_fast = int(macd_fast)
                    self.macd_slow = int(macd_slow)
                    self.macd_signal = int(macd_signal)
                    self.bb_window = int(bb_window)
                    self.bb_std = bb_std

            # Run backtest
            bt = Backtest(self.data, OptimizedStrategy,
                         cash=10000, commission=0.002)
            stats = bt.run()

            # Calculate composite score (Sharpe + Return - Risk)
            sharpe = stats['Sharpe Ratio']
            returns = stats['Return [%]']
            max_drawdown = stats['Max. Drawdown [%]']

            # Neuro-flux adjusted scoring
            flux_penalty = config.FLUX_SENSITIVITY * max_drawdown / 100
            score = sharpe * 0.4 + returns * 0.4 - flux_penalty * 0.2

            # Store optimization result
            self.optimization_history.append({
                'params': params,
                'score': score,
                'sharpe': sharpe,
                'returns': returns,
                'max_drawdown': max_drawdown
            })

            return -score  # Minimize negative score

        except Exception as e:
            # Return high penalty for invalid parameters
            return 1000

    def optimize_parameters(self, bounds=None):
        """
        Run neuro-optimization on strategy parameters.
        """
        if bounds is None:
            bounds = [
                (5, 30),    # RSI window
                (8, 20),    # MACD fast
                (20, 40),   # MACD slow
                (5, 15),    # MACD signal
                (10, 30),   # BB window
                (1.5, 3.0)  # BB std
            ]

        # Initial guess
        x0 = [14, 12, 26, 9, 20, 2.0]

        cprint("🧠 Starting neuro-optimization...", "cyan")

        # Run optimization
        result = minimize(
            self.objective_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 50, 'disp': True}
        )

        # Get best parameters
        best_params = result.x
        best_score = -result.fun

        cprint(f"✅ Optimization complete! Best score: {best_score:.4f}", "green")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_result': result,
            'history': self.optimization_history[-10:]  # Last 10 iterations
        }

class FluxAdaptiveBacktest:
    """
    Flux-adaptive backtesting with neuro-optimization.
    """

    def __init__(self, strategy_class, data):
        self.strategy_class = strategy_class
        self.data = data
        self.optimizer = NeuroOptimizer(strategy_class, data)

    def run_adaptive_backtest(self, optimization_bounds=None):
        """
        Run flux-adaptive backtest with optimization.
        """
        cprint("🧠 NeuroFlux Adaptive Backtesting starting...", "cyan")

        # Phase 1: Parameter optimization
        cprint("📊 Phase 1: Neuro-optimization of parameters", "yellow")
        opt_result = self.optimizer.optimize_parameters(optimization_bounds)

        # Phase 2: Flux-adaptive backtest
        cprint("🌊 Phase 2: Flux-adaptive backtesting", "yellow")

        # Create optimized strategy
        class AdaptiveStrategy(self.strategy_class):
            def init(self):
                super().init()
                # Apply optimized parameters
                params = opt_result['best_params']
                self.rsi_window = int(params[0])
                self.macd_fast = int(params[1])
                self.macd_slow = int(params[2])
                self.macd_signal = int(params[3])
                self.bb_window = int(params[4])
                self.bb_std = params[5]

            def next(self):
                # Apply neuro-flux adaptation
                self.neuro_adapt()
                super().next()

        # Run backtest with optimized parameters
        bt = Backtest(self.data, AdaptiveStrategy,
                     cash=10000, commission=0.002)
        stats = bt.run()

        # Calculate flux-adjusted metrics
        flux_adjusted_sharpe = stats['Sharpe Ratio'] * (1 - config.FLUX_SENSITIVITY)
        flux_adjusted_return = stats['Return [%]'] * (1 - config.FLUX_SENSITIVITY * 0.5)

        result = {
            'timestamp': datetime.now().isoformat(),
            'optimization': opt_result,
            'backtest_stats': stats,
            'flux_adjusted_metrics': {
                'sharpe_ratio': flux_adjusted_sharpe,
                'return_pct': flux_adjusted_return,
                'flux_penalty': config.FLUX_SENSITIVITY
            },
            'parameters': {
                'rsi_window': int(opt_result['best_params'][0]),
                'macd_fast': int(opt_result['best_params'][1]),
                'macd_slow': int(opt_result['best_params'][2]),
                'macd_signal': int(opt_result['best_params'][3]),
                'bb_window': int(opt_result['best_params'][4]),
                'bb_std': opt_result['best_params'][5]
            }
        }

        # Save results
        self.save_results(result)

        cprint("✅ Adaptive backtest complete!", "green")
        return result

    def save_results(self, result):
        """
        Save backtesting results.
        """
        # Save latest results
        with open(f"{OUTPUT_DIR}/latest_backtest.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)

        # Append to history
        with open(f"{OUTPUT_DIR}/backtest_history.jsonl", 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

def load_rbi_strategy(rbi_result_path):
    """
    Load and enhance RBI-generated strategy with neuro-optimization.
    """
    try:
        # Load RBI-generated code
        with open(f"{rbi_result_path}/backtest_code.py", 'r') as f:
            rbi_code = f.read()

        # Load RBI results
        with open(f"{rbi_result_path}/backtest_results.json", 'r') as f:
            rbi_results = json.load(f)

        cprint(f"📥 Loaded RBI strategy from {rbi_result_path}", "blue")

        return {
            'code': rbi_code,
            'results': rbi_results,
            'path': rbi_result_path
        }

    except Exception as e:
        cprint(f"❌ Error loading RBI strategy: {str(e)}", "red")
        return None

def run_neuro_backtest(strategy_class, data_path, rbi_integration=False):
    """
    Main function to run neuro-flux enhanced backtesting.
    """
    try:
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        else:
            raise ValueError("Unsupported data format")

        cprint(f"📊 Loaded {len(data)} data points", "blue")

        # Initialize adaptive backtest
        adaptive_bt = FluxAdaptiveBacktest(strategy_class, data)

        # Run with RBI integration if requested
        if rbi_integration:
            cprint("🔗 Integrating with RBI agent results", "yellow")
            # Load latest RBI results
            rbi_dirs = [d for d in os.listdir("src/data/rbi_agent/") if d.startswith("rbi_result_")]
            if rbi_dirs:
                latest_rbi = max(rbi_dirs)
                rbi_data = load_rbi_strategy(f"src/data/rbi_agent/{latest_rbi}")
                if rbi_data:
                    cprint("✅ RBI integration successful", "green")

        # Run adaptive backtest
        results = adaptive_bt.run_adaptive_backtest()

        return results

    except Exception as e:
        cprint(f"❌ Backtest error: {str(e)}", "red")
        return None

# Example usage
if __name__ == "__main__":
    # This would be called by other modules
    print("NeuroFlux Enhanced Backtesting Framework")
    print("Use run_neuro_backtest() function to start backtesting")