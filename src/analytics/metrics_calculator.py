"""
🧠 NeuroFlux Metrics Calculator
Calculates comprehensive performance metrics for trading systems.

Built with love by Nyros Veil 🚀

Features:
- Trading performance metrics (PnL, Sharpe, drawdown, win rate)
- Risk metrics (VaR, volatility, correlation, beta)
- Agent efficiency metrics (success rates, response times)
- Portfolio analytics (diversification, asset allocation)
- Statistical analysis and trend detection
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for trading systems"""
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    risk_adjusted_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    value_at_risk: float = 0.0
    expected_shortfall: float = 0.0

    # Trading metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Portfolio metrics
    diversification_ratio: float = 0.0
    correlation_matrix: Dict[str, Dict[str, float]] = None
    beta: float = 0.0

    # Time-based metrics
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Metadata
    calculation_date: str = ""
    period_days: int = 0
    benchmark_return: float = 0.0

class MetricsCalculator:
    """Calculates comprehensive trading performance metrics"""

    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        cprint("📊 MetricsCalculator initialized", "cyan")

    def calculate_portfolio_metrics(self, portfolio_data: List[Dict[str, Any]],
                                  benchmark_data: Optional[List[float]] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive portfolio performance metrics

        Args:
            portfolio_data: List of portfolio snapshots with timestamps and values
            benchmark_data: Optional benchmark returns for comparison

        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if not portfolio_data:
            return PerformanceMetrics()

        try:
            # Convert to DataFrame for easier calculations
            df = pd.DataFrame(portfolio_data)

            # Ensure we have required columns
            if 'timestamp' not in df.columns or 'equity' not in df.columns:
                cprint("⚠️  Missing required columns (timestamp, equity) in portfolio data", "yellow")
                return PerformanceMetrics()

            # Sort by timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Calculate returns
            df['returns'] = df['equity'].pct_change().fillna(0)

            # Calculate metrics
            metrics = PerformanceMetrics()
            metrics.calculation_date = datetime.now().isoformat()
            metrics.period_days = (df['timestamp'].max() - df['timestamp'].min()).days

            # Basic return metrics
            metrics.total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
            metrics.annualized_return = self._calculate_annualized_return(metrics.total_return, metrics.period_days)

            # Risk metrics
            metrics.volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(df['returns'], metrics.annualized_return)
            metrics.sortino_ratio = self._calculate_sortino_ratio(df['returns'])
            metrics.max_drawdown = self._calculate_max_drawdown(df['equity'])

            # Value at Risk (95% confidence)
            metrics.value_at_risk = self._calculate_var(df['returns'], confidence=0.95)
            metrics.expected_shortfall = self._calculate_expected_shortfall(df['returns'], confidence=0.95)

            # Trading metrics (if available)
            if 'trades' in df.columns:
                trade_metrics = self._calculate_trading_metrics(df)
                metrics.total_trades = trade_metrics['total_trades']
                metrics.win_rate = trade_metrics['win_rate']
                metrics.profit_factor = trade_metrics['profit_factor']
                metrics.average_win = trade_metrics['average_win']
                metrics.average_loss = trade_metrics['average_loss']
                metrics.largest_win = trade_metrics['largest_win']
                metrics.largest_loss = trade_metrics['largest_loss']

            # Advanced ratios
            metrics.calmar_ratio = self._calculate_calmar_ratio(metrics.annualized_return, metrics.max_drawdown)
            metrics.omega_ratio = self._calculate_omega_ratio(df['returns'])

            # Benchmark comparison
            if benchmark_data:
                metrics.benchmark_return = np.mean(benchmark_data) * 252  # Annualized
                metrics.beta = self._calculate_beta(df['returns'], benchmark_data)

            return metrics

        except Exception as e:
            cprint(f"❌ Error calculating portfolio metrics: {e}", "red")
            return PerformanceMetrics()

    def calculate_agent_metrics(self, agent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics for individual agents

        Args:
            agent_data: List of agent execution records

        Returns:
            Dictionary with agent performance metrics
        """
        if not agent_data:
            return {}

        try:
            df = pd.DataFrame(agent_data)

            metrics = {
                'total_executions': len(df),
                'success_rate': 0.0,
                'average_execution_time': 0.0,
                'error_rate': 0.0,
                'last_execution': None,
                'execution_frequency': 0.0
            }

            # Success rate
            if 'success' in df.columns:
                metrics['success_rate'] = df['success'].mean() * 100

            # Execution time
            if 'execution_time' in df.columns:
                metrics['average_execution_time'] = df['execution_time'].mean()

            # Error rate
            if 'error' in df.columns:
                metrics['error_rate'] = df['error'].sum() / len(df) * 100

            # Last execution
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                metrics['last_execution'] = df['timestamp'].max().isoformat()

                # Execution frequency (executions per day)
                date_range = (df['timestamp'].max() - df['timestamp'].min()).days
                if date_range > 0:
                    metrics['execution_frequency'] = len(df) / date_range

            return metrics

        except Exception as e:
            cprint(f"❌ Error calculating agent metrics: {e}", "red")
            return {}

    def calculate_correlation_matrix(self, agent_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between different agents/metrics

        Args:
            agent_data: Dictionary mapping agent names to their data

        Returns:
            Correlation matrix as nested dictionary
        """
        try:
            # Extract time series for each agent
            time_series = {}

            for agent_name, data in agent_data.items():
                if not data:
                    continue

                df = pd.DataFrame(data)
                if 'timestamp' in df.columns and 'value' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp').sort_index()
                    time_series[agent_name] = df['value']

            if len(time_series) < 2:
                return {}

            # Create combined DataFrame
            combined_df = pd.DataFrame(time_series)

            # Calculate correlation matrix
            corr_matrix = combined_df.corr()

            # Convert to nested dictionary
            result = {}
            for col in corr_matrix.columns:
                result[col] = {}
                for idx in corr_matrix.index:
                    result[col][idx] = corr_matrix.loc[idx, col]

            return result

        except Exception as e:
            cprint(f"❌ Error calculating correlation matrix: {e}", "red")
            return {}

    def _calculate_annualized_return(self, total_return: float, days: int) -> float:
        """Calculate annualized return from total return and time period"""
        if days <= 0:
            return 0.0
        years = days / 365.25
        return ((1 + total_return / 100) ** (1 / years) - 1) * 100

    def _calculate_sharpe_ratio(self, returns: pd.Series, annualized_return: float) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        return returns.mean() / downside_returns.std() * np.sqrt(252)

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return drawdown.min() * 100  # Convert to percentage

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100) * 100

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_losses = returns[returns <= var_threshold]
        return tail_losses.mean() * 100 if len(tail_losses) > 0 else 0.0

    def _calculate_trading_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }

        if 'trades' not in df.columns:
            return metrics

        # Flatten all trades
        all_trades = []
        for trades_list in df['trades'].dropna():
            if isinstance(trades_list, list):
                all_trades.extend(trades_list)

        if not all_trades:
            return metrics

        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)

        if 'pnl' not in trades_df.columns:
            return metrics

        metrics['total_trades'] = len(trades_df)

        # Separate wins and losses
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']

        if len(wins) > 0:
            metrics['win_rate'] = len(wins) / len(trades_df) * 100
            metrics['average_win'] = wins.mean()
            metrics['largest_win'] = wins.max()

        if len(losses) > 0:
            metrics['average_loss'] = abs(losses.mean())
            metrics['largest_loss'] = abs(losses.min())

        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0

        if total_losses > 0:
            metrics['profit_factor'] = total_wins / total_losses

        return metrics

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / abs(max_drawdown)

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio (upside potential / downside risk)"""
        upside = returns[returns > threshold].sum()
        downside = abs(returns[returns < threshold].sum())

        if downside == 0:
            return float('inf') if upside > 0 else 0.0

        return upside / downside

    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: List[float]) -> float:
        """Calculate beta relative to benchmark"""
        try:
            # Align the series
            bench_series = pd.Series(benchmark_returns[:len(portfolio_returns)], index=portfolio_returns.index)

            # Calculate covariance and variance
            covariance = portfolio_returns.cov(bench_series)
            benchmark_variance = bench_series.var()

            if benchmark_variance == 0:
                return 1.0

            return covariance / benchmark_variance

        except Exception:
            return 1.0  # Default to market beta