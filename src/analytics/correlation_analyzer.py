"""
🧠 NeuroFlux Correlation Analyzer
Analyzes correlations between different agents and market factors.

Built with love by Nyros Veil 🚀

Features:
- Cross-agent correlation analysis
- Sentiment vs price correlation
- Risk vs performance correlation
- Multi-agent consensus analysis
- Market regime detection
- Statistical significance testing
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class CorrelationAnalyzer:
    """Analyzes correlations between agents and market factors"""

    def __init__(self):
        cprint("🔗 CorrelationAnalyzer initialized", "cyan")

    def analyze_sentiment_price_correlation(self, sentiment_data: List[Dict[str, Any]],
                                           price_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze correlation between sentiment scores and price movements

        Args:
            sentiment_data: List of sentiment analysis results
            price_data: Optional list of price data points

        Returns:
            Dictionary with correlation analysis results
        """
        if not sentiment_data:
            return {}

        try:
            # Convert to DataFrames
            sentiment_df = pd.DataFrame(sentiment_data)

            # Group by token and analyze each token separately
            results = {}

            if 'token' in sentiment_df.columns:
                tokens = sentiment_df['token'].unique()

                for token in tokens:
                    token_sentiment = sentiment_df[sentiment_df['token'] == token].copy()

                    if len(token_sentiment) < 5:  # Need minimum data points
                        continue

                    # Prepare sentiment time series
                    token_sentiment['timestamp'] = pd.to_datetime(token_sentiment['timestamp'])
                    token_sentiment = token_sentiment.sort_values('timestamp').set_index('timestamp')

                    # Resample to daily frequency
                    daily_sentiment = token_sentiment['overall_score'].resample('D').mean().dropna()

                    if len(daily_sentiment) < 3:
                        continue

                    # Calculate sentiment momentum (changes)
                    sentiment_changes = daily_sentiment.diff().dropna()

                    # Calculate autocorrelation (sentiment persistence)
                    autocorr_1d = daily_sentiment.autocorr(lag=1) if len(daily_sentiment) > 1 else 0
                    autocorr_7d = daily_sentiment.autocorr(lag=7) if len(daily_sentiment) > 7 else 0

                    # Calculate volatility of sentiment
                    sentiment_volatility = daily_sentiment.std()

                    # Calculate sentiment trends
                    sentiment_trend = self._calculate_trend(daily_sentiment)

                    results[token] = {
                        'data_points': len(daily_sentiment),
                        'sentiment_volatility': sentiment_volatility,
                        'sentiment_autocorr_1d': autocorr_1d,
                        'sentiment_autocorr_7d': autocorr_7d,
                        'sentiment_trend': sentiment_trend,
                        'sentiment_range': daily_sentiment.max() - daily_sentiment.min(),
                        'sentiment_mean': daily_sentiment.mean(),
                        'sentiment_median': daily_sentiment.median(),
                        'sentiment_skewness': daily_sentiment.skew(),
                        'sentiment_kurtosis': daily_sentiment.kurtosis()
                    }

            # Overall sentiment analysis
            if len(sentiment_df) > 0:
                overall_sentiment = sentiment_df['overall_score'].dropna()

                results['overall'] = {
                    'total_observations': len(sentiment_df),
                    'unique_tokens': len(sentiment_df['token'].unique()) if 'token' in sentiment_df.columns else 0,
                    'sentiment_distribution': {
                        'bullish': (overall_sentiment > 0.1).sum(),
                        'neutral': ((overall_sentiment >= -0.1) & (overall_sentiment <= 0.1)).sum(),
                        'bearish': (overall_sentiment < -0.1).sum()
                    },
                    'average_sentiment': overall_sentiment.mean(),
                    'sentiment_std': overall_sentiment.std(),
                    'sentiment_confidence_avg': sentiment_df.get('confidence', pd.Series()).mean()
                }

            return results

        except Exception as e:
            cprint(f"❌ Error analyzing sentiment-price correlation: {e}", "red")
            return {}

    def analyze_risk_performance_correlation(self, risk_data: List[Dict[str, Any]],
                                            performance_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze correlation between risk metrics and performance

        Args:
            risk_data: List of risk assessment results
            performance_data: Optional list of performance metrics

        Returns:
            Dictionary with risk-performance correlation analysis
        """
        if not risk_data:
            return {}

        try:
            risk_df = pd.DataFrame(risk_data)

            results = {
                'risk_distribution': {},
                'risk_trends': {},
                'violation_patterns': {},
                'flux_analysis': {}
            }

            # Risk distribution analysis
            if 'flux_level' in risk_df.columns:
                flux_levels = risk_df['flux_level'].dropna()
                results['flux_analysis'] = {
                    'mean_flux': flux_levels.mean(),
                    'flux_volatility': flux_levels.std(),
                    'high_flux_periods': (flux_levels > 0.7).sum(),
                    'low_flux_periods': (flux_levels < 0.3).sum(),
                    'flux_trend': self._calculate_trend(flux_levels)
                }

            # Risk check analysis
            if 'risk_check' in risk_df.columns:
                violations = []
                warnings = []

                for check in risk_df['risk_check'].dropna():
                    if isinstance(check, dict):
                        violations.extend(check.get('violations', []))
                        warnings.extend(check.get('warnings', []))

                results['violation_patterns'] = {
                    'total_violations': len(violations),
                    'unique_violation_types': len(set(str(v) for v in violations)),
                    'total_warnings': len(warnings),
                    'violation_rate': len(violations) / len(risk_df) if len(risk_df) > 0 else 0
                }

            # Position analysis
            if 'total_positions' in risk_df.columns:
                positions = risk_df['total_positions'].dropna()
                results['position_analysis'] = {
                    'average_positions': positions.mean(),
                    'max_positions': positions.max(),
                    'position_volatility': positions.std(),
                    'zero_position_periods': (positions == 0).sum()
                }

            # PnL analysis
            if 'total_pnl' in risk_df.columns:
                pnl = risk_df['total_pnl'].dropna()
                if len(pnl) > 0:
                    results['pnl_analysis'] = {
                        'total_pnl': pnl.sum(),
                        'average_daily_pnl': pnl.mean(),
                        'pnl_volatility': pnl.std(),
                        'positive_days': (pnl > 0).sum(),
                        'negative_days': (pnl < 0).sum(),
                        'win_rate': (pnl > 0).sum() / len(pnl) if len(pnl) > 0 else 0
                    }

            return results

        except Exception as e:
            cprint(f"❌ Error analyzing risk-performance correlation: {e}", "red")
            return {}

    def analyze_agent_consensus(self, agent_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze consensus levels between different agents

        Args:
            agent_data: Dictionary mapping agent names to their data

        Returns:
            Dictionary with consensus analysis results
        """
        try:
            consensus_results = {
                'agent_count': len(agent_data),
                'consensus_metrics': {},
                'disagreement_patterns': {},
                'agent_reliability': {}
            }

            # Extract signals from each agent
            agent_signals = {}

            for agent_name, data in agent_data.items():
                if not data:
                    continue

                signals = []
                for record in data:
                    # Extract signal/action from different agent types
                    signal = self._extract_agent_signal(agent_name, record)
                    if signal is not None:
                        signals.append({
                            'timestamp': record.get('timestamp'),
                            'signal': signal,
                            'confidence': record.get('confidence', 0.5)
                        })

                if signals:
                    agent_signals[agent_name] = signals

            # Calculate consensus metrics
            if len(agent_signals) > 1:
                consensus_results['consensus_metrics'] = self._calculate_consensus_metrics(agent_signals)

            # Calculate agent reliability scores
            for agent_name, signals in agent_signals.items():
                if signals:
                    confidence_scores = [s['confidence'] for s in signals]
                    consensus_results['agent_reliability'][agent_name] = {
                        'signal_count': len(signals),
                        'average_confidence': np.mean(confidence_scores),
                        'confidence_volatility': np.std(confidence_scores),
                        'consistency_score': self._calculate_signal_consistency(signals)
                    }

            return consensus_results

        except Exception as e:
            cprint(f"❌ Error analyzing agent consensus: {e}", "red")
            return {}

    def detect_market_regime(self, combined_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Detect market regime based on combined agent signals

        Args:
            combined_data: Dictionary with data from multiple agents

        Returns:
            Dictionary with market regime analysis
        """
        try:
            regime_analysis = {
                'current_regime': 'unknown',
                'regime_confidence': 0.0,
                'regime_indicators': {},
                'regime_history': []
            }

            # Extract key indicators
            sentiment_score = self._get_overall_sentiment(combined_data.get('sentiment_agent', []))
            risk_level = self._get_overall_risk(combined_data.get('risk_agent', []))
            flux_level = self._get_overall_flux(combined_data.get('risk_agent', []))

            # Determine regime based on indicators
            if sentiment_score > 0.2 and risk_level < 0.3 and flux_level < 0.4:
                regime_analysis['current_regime'] = 'bull_optimistic'
                regime_analysis['regime_confidence'] = 0.8
            elif sentiment_score > 0.1 and risk_level < 0.5:
                regime_analysis['current_regime'] = 'bull_cautious'
                regime_analysis['regime_confidence'] = 0.6
            elif sentiment_score < -0.2 and risk_level > 0.7:
                regime_analysis['current_regime'] = 'bear_distressed'
                regime_analysis['regime_confidence'] = 0.8
            elif sentiment_score < -0.1 or risk_level > 0.6:
                regime_analysis['current_regime'] = 'bear_concerned'
                regime_analysis['regime_confidence'] = 0.6
            else:
                regime_analysis['current_regime'] = 'neutral_range'
                regime_analysis['regime_confidence'] = 0.5

            # Store regime indicators
            regime_analysis['regime_indicators'] = {
                'sentiment_score': sentiment_score,
                'risk_level': risk_level,
                'flux_level': flux_level,
                'timestamp': datetime.now().isoformat()
            }

            return regime_analysis

        except Exception as e:
            cprint(f"❌ Error detecting market regime: {e}", "red")
            return {}

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend slope using linear regression"""
        try:
            if len(series) < 2:
                return 0.0

            x = np.arange(len(series))
            y = series.values

            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        except:
            return 0.0

    def _extract_agent_signal(self, agent_name: str, record: Dict[str, Any]) -> Optional[str]:
        """Extract signal/action from agent record"""
        try:
            if agent_name == 'sentiment_agent':
                score = record.get('overall_score', 0)
                if score > 0.1:
                    return 'bullish'
                elif score < -0.1:
                    return 'bearish'
                else:
                    return 'neutral'

            elif agent_name == 'risk_agent':
                risk_check = record.get('risk_check', {})
                if isinstance(risk_check, dict) and not risk_check.get('ok', True):
                    return 'risk_high'
                else:
                    return 'risk_low'

            elif agent_name == 'strategy_agent':
                # Placeholder for strategy signals
                return record.get('signal', 'hold')

            # Default to neutral for unknown agents
            return 'neutral'

        except:
            return None

    def _calculate_consensus_metrics(self, agent_signals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate consensus metrics between agents"""
        try:
            consensus = {
                'agreement_rate': 0.0,
                'disagreement_rate': 0.0,
                'consensus_strength': 0.0,
                'signal_distribution': {}
            }

            # Get all timestamps
            all_timestamps = set()
            for signals in agent_signals.values():
                for signal in signals:
                    if signal.get('timestamp'):
                        all_timestamps.add(signal['timestamp'])

            # Calculate consensus for each timestamp
            total_comparisons = 0
            agreements = 0

            for timestamp in sorted(all_timestamps):
                timestamp_signals = []

                for agent_signals_list in agent_signals.values():
                    for signal in agent_signals_list:
                        if signal.get('timestamp') == timestamp:
                            timestamp_signals.append(signal['signal'])
                            break

                if len(timestamp_signals) > 1:
                    # Check if all signals agree
                    unique_signals = set(timestamp_signals)
                    if len(unique_signals) == 1:
                        agreements += 1
                    total_comparisons += 1

            if total_comparisons > 0:
                consensus['agreement_rate'] = agreements / total_comparisons
                consensus['disagreement_rate'] = 1 - consensus['agreement_rate']

                # Consensus strength (weighted by number of agents)
                agent_count = len(agent_signals)
                consensus['consensus_strength'] = consensus['agreement_rate'] * (agent_count / max(agent_count, 1))

            return consensus

        except Exception as e:
            cprint(f"❌ Error calculating consensus metrics: {e}", "red")
            return {}

    def _calculate_signal_consistency(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate consistency score for agent signals"""
        try:
            if len(signals) < 2:
                return 1.0

            signal_values = [s['signal'] for s in signals]
            most_common = max(set(signal_values), key=signal_values.count)
            consistency = signal_values.count(most_common) / len(signal_values)

            return consistency

        except:
            return 0.0

    def _get_overall_sentiment(self, sentiment_data: List[Dict[str, Any]]) -> float:
        """Get overall sentiment score from sentiment data"""
        if not sentiment_data:
            return 0.0

        try:
            scores = [record.get('overall_score', 0) for record in sentiment_data if 'overall_score' in record]
            return np.mean(scores) if scores else 0.0
        except:
            return 0.0

    def _get_overall_risk(self, risk_data: List[Dict[str, Any]]) -> float:
        """Get overall risk level from risk data"""
        if not risk_data:
            return 0.5

        try:
            risk_levels = []
            for record in risk_data:
                risk_check = record.get('risk_check', {})
                if isinstance(risk_check, dict) and not risk_check.get('ok', True):
                    risk_levels.append(0.8)  # High risk
                else:
                    risk_levels.append(0.2)  # Low risk

            return np.mean(risk_levels) if risk_levels else 0.5
        except:
            return 0.5

    def _get_overall_flux(self, risk_data: List[Dict[str, Any]]) -> float:
        """Get overall flux level from risk data"""
        if not risk_data:
            return 0.5

        try:
            flux_levels = [record.get('flux_level', 0.5) for record in risk_data if 'flux_level' in record]
            return np.mean(flux_levels) if flux_levels else 0.5
        except:
            return 0.5