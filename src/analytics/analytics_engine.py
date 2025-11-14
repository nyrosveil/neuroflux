"""
🧠 NeuroFlux Analytics Engine
Central orchestrator for all analytics and reporting functionality.

Built with love by Nyros Veil 🚀

Features:
- Unified analytics processing pipeline
- Real-time and historical analysis
- Performance dashboard generation
- Automated insights and recommendations
- Cross-agent correlation analysis
- Export and reporting capabilities
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

from .data_aggregator import DataAggregator
from .metrics_calculator import MetricsCalculator, PerformanceMetrics
from .correlation_analyzer import CorrelationAnalyzer

class AnalyticsEngine:
    """Main analytics engine orchestrating all analysis components"""

    def __init__(self, data_dir: str = "src/data", enable_cache: bool = True):
        self.data_dir = Path(data_dir)
        self.enable_cache = enable_cache

        # Initialize components
        self.data_aggregator = DataAggregator(str(data_dir))
        self.metrics_calculator = MetricsCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()

        # Cache for computed results
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5 minutes default TTL

        cprint("🧠 AnalyticsEngine initialized with all components", "cyan")

    def generate_system_overview(self) -> Dict[str, Any]:
        """Generate comprehensive system overview with all key metrics"""
        try:
            overview = {
                'timestamp': datetime.now().isoformat(),
                'system_health': {},
                'performance_summary': {},
                'agent_status': {},
                'market_analysis': {},
                'recommendations': []
            }

            # System health metrics
            system_summary = self.data_aggregator.get_system_summary()
            overview['system_health'] = {
                'total_agents': system_summary['total_agents'],
                'active_agents': system_summary['agents_with_data'],
                'data_coverage': system_summary['agents_with_data'] / max(system_summary['total_agents'], 1),
                'total_data_points': system_summary['total_history_records']
            }

            # Get latest agent data
            latest_data = self.data_aggregator.get_all_agents_latest()

            # Performance summary
            if latest_data:
                portfolio_data = self._extract_portfolio_data(latest_data)
                if portfolio_data:
                    performance = self.metrics_calculator.calculate_portfolio_metrics(portfolio_data)
                    overview['performance_summary'] = {
                        'total_return_pct': performance.total_return,
                        'sharpe_ratio': performance.sharpe_ratio,
                        'max_drawdown_pct': performance.max_drawdown,
                        'win_rate': performance.win_rate,
                        'current_positions': len([d for d in portfolio_data if d.get('total_positions', 0) > 0])
                    }

            # Agent status
            overview['agent_status'] = {}
            for agent_name, agent_data in latest_data.items():
                status = self._assess_agent_health(agent_name, agent_data)
                overview['agent_status'][agent_name] = status

            # Market analysis
            sentiment_data = latest_data.get('sentiment_agent', [])
            risk_data = latest_data.get('risk_agent', [])

            if sentiment_data:
                sentiment_analysis = self.correlation_analyzer.analyze_sentiment_price_correlation(sentiment_data)
                overview['market_analysis']['sentiment'] = sentiment_analysis.get('overall', {})

            if risk_data:
                risk_analysis = self.correlation_analyzer.analyze_risk_performance_correlation(risk_data)
                overview['market_analysis']['risk'] = risk_analysis

            # Market regime detection
            combined_data = {
                'sentiment_agent': sentiment_data,
                'risk_agent': risk_data
            }
            regime = self.correlation_analyzer.detect_market_regime(combined_data)
            overview['market_analysis']['regime'] = regime

            # Generate recommendations
            overview['recommendations'] = self._generate_system_recommendations(overview)

            return overview

        except Exception as e:
            cprint(f"❌ Error generating system overview: {e}", "red")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def generate_agent_report(self, agent_name: str, hours_back: int = 24) -> Dict[str, Any]:
        """Generate detailed report for a specific agent"""
        try:
            report = {
                'agent_name': agent_name,
                'timestamp': datetime.now().isoformat(),
                'period_hours': hours_back,
                'latest_data': {},
                'historical_analysis': {},
                'performance_metrics': {},
                'health_score': 0.0,
                'recommendations': []
            }

            # Get latest data
            latest = self.data_aggregator.get_latest_reports(agent_name)
            if latest:
                report['latest_data'] = latest

            # Get historical data
            historical = self.data_aggregator.get_historical_data(agent_name, hours_back)
            if historical:
                report['historical_analysis'] = self._analyze_historical_trends(historical)

            # Calculate agent-specific metrics
            if agent_name == 'risk_agent':
                report['performance_metrics'] = self._analyze_risk_metrics(historical)
            elif agent_name == 'sentiment_agent':
                report['performance_metrics'] = self._analyze_sentiment_metrics(historical)
            elif agent_name == 'trading_agent':
                report['performance_metrics'] = self._analyze_trading_metrics(historical)

            # Health assessment
            report['health_score'] = self._calculate_agent_health_score(agent_name, latest, historical)

            # Recommendations
            report['recommendations'] = self._generate_agent_recommendations(agent_name, report)

            return report

        except Exception as e:
            cprint(f"❌ Error generating agent report for {agent_name}: {e}", "red")
            return {'error': str(e), 'agent_name': agent_name, 'timestamp': datetime.now().isoformat()}

    def generate_performance_dashboard(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard"""
        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'period_days': days_back,
                'portfolio_performance': {},
                'agent_performance': {},
                'market_analysis': {},
                'risk_metrics': {},
                'correlation_analysis': {},
                'alerts': []
            }

            hours_back = days_back * 24

            # Portfolio performance
            all_historical = self.data_aggregator.get_all_agents_history(hours_back)
            portfolio_data = self._extract_portfolio_data_from_history(all_historical)

            if portfolio_data:
                performance = self.metrics_calculator.calculate_portfolio_metrics(portfolio_data)
                dashboard['portfolio_performance'] = self._format_performance_metrics(performance)

            # Agent performance
            dashboard['agent_performance'] = {}
            for agent_name in self.data_aggregator.agent_directories.keys():
                agent_report = self.generate_agent_report(agent_name, hours_back)
                dashboard['agent_performance'][agent_name] = {
                    'health_score': agent_report.get('health_score', 0),
                    'data_points': len(agent_report.get('historical_analysis', {})),
                    'key_metrics': agent_report.get('performance_metrics', {})
                }

            # Market analysis
            sentiment_data = all_historical.get('sentiment_agent', [])
            risk_data = all_historical.get('risk_agent', [])

            if sentiment_data:
                sentiment_analysis = self.correlation_analyzer.analyze_sentiment_price_correlation(sentiment_data)
                dashboard['market_analysis']['sentiment_trends'] = sentiment_analysis

            if risk_data:
                risk_analysis = self.correlation_analyzer.analyze_risk_performance_correlation(risk_data)
                dashboard['market_analysis']['risk_analysis'] = risk_analysis

            # Correlation analysis
            agent_data = {name: data for name, data in all_historical.items() if data}
            if len(agent_data) > 1:
                correlations = self.correlation_analyzer.analyze_agent_consensus(agent_data)
                dashboard['correlation_analysis'] = correlations

            # Risk metrics
            if portfolio_data:
                dashboard['risk_metrics'] = self._extract_risk_metrics(performance)

            # Generate alerts
            dashboard['alerts'] = self._generate_system_alerts(dashboard)

            return dashboard

        except Exception as e:
            cprint(f"❌ Error generating performance dashboard: {e}", "red")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def export_report(self, report_data: Dict[str, Any], format: str = 'json',
                     output_path: Optional[str] = None) -> str:
        """Export report in specified format"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = str(self.data_dir / "analytics" / f"report_{timestamp}.{format}")

            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)

            elif format == 'csv':
                # Convert to CSV format (simplified)
                self._export_as_csv(report_data, output_path)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            cprint(f"✅ Report exported to {output_path}", "green")
            return output_path

        except Exception as e:
            cprint(f"❌ Error exporting report: {e}", "red")
            return ""

    def get_cached_result(self, key: str, compute_func, ttl_seconds: Optional[int] = None) -> Any:
        """Get cached result or compute if expired"""
        if not self.enable_cache:
            return compute_func()

        ttl = ttl_seconds or self.cache_ttl
        now = time.time()

        if key in self.cache and key in self.cache_timestamps:
            if now - self.cache_timestamps[key] < ttl:
                return self.cache[key]

        # Compute fresh result
        result = compute_func()
        self.cache[key] = result
        self.cache_timestamps[key] = now

        return result

    def clear_cache(self):
        """Clear all cached results"""
        self.cache = {}
        self.cache_timestamps = {}
        cprint("🧹 Analytics cache cleared", "blue")

    def _extract_portfolio_data(self, latest_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract portfolio data from latest agent data"""
        portfolio_data = []

        # Get risk agent data (primary source for portfolio info)
        risk_data = latest_data.get('risk_agent')
        if risk_data and isinstance(risk_data, dict):
            portfolio_data.append({
                'timestamp': risk_data.get('timestamp', datetime.now().isoformat()),
                'equity': risk_data.get('balance', {}).get('equity', 1000),
                'total_positions': risk_data.get('total_positions', 0),
                'total_pnl': risk_data.get('total_pnl', 0),
                'flux_level': risk_data.get('flux_level', 0.5)
            })

        return portfolio_data

    def _extract_portfolio_data_from_history(self, historical_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract portfolio data from historical agent data"""
        portfolio_data = []

        risk_history = historical_data.get('risk_agent', [])
        for record in risk_history:
            if isinstance(record, dict):
                portfolio_data.append({
                    'timestamp': record.get('timestamp', datetime.now().isoformat()),
                    'equity': record.get('balance', {}).get('equity', 1000),
                    'total_positions': record.get('total_positions', 0),
                    'total_pnl': record.get('total_pnl', 0),
                    'flux_level': record.get('flux_level', 0.5)
                })

        return portfolio_data

    def _assess_agent_health(self, agent_name: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess health status of an agent"""
        health = {
            'status': 'unknown',
            'score': 0.0,
            'last_update': None,
            'data_quality': 'unknown'
        }

        if not agent_data:
            health['status'] = 'no_data'
            return health

        # Check timestamp freshness
        timestamp_str = agent_data.get('timestamp') or agent_data.get('_metadata', {}).get('timestamp')
        if timestamp_str:
            try:
                last_update = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                health['last_update'] = last_update.isoformat()
                hours_old = (datetime.now() - last_update).total_seconds() / 3600

                if hours_old < 1:
                    health['status'] = 'active'
                    health['score'] = 1.0
                elif hours_old < 6:
                    health['status'] = 'recent'
                    health['score'] = 0.8
                elif hours_old < 24:
                    health['status'] = 'stale'
                    health['score'] = 0.5
                else:
                    health['status'] = 'inactive'
                    health['score'] = 0.2
            except:
                pass

        # Assess data quality
        if agent_name == 'sentiment_agent':
            confidence = agent_data.get('confidence', 0)
            if confidence > 0.8:
                health['data_quality'] = 'high'
            elif confidence > 0.6:
                health['data_quality'] = 'medium'
            else:
                health['data_quality'] = 'low'
        elif agent_name == 'risk_agent':
            violations = len(agent_data.get('risk_check', {}).get('violations', []))
            if violations == 0:
                health['data_quality'] = 'good'
            else:
                health['data_quality'] = 'concerning'

        return health

    def _analyze_historical_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical trends in agent data"""
        if not historical_data:
            return {}

        try:
            df = pd.DataFrame(historical_data)
            if 'timestamp' not in df.columns:
                return {}

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            trends = {
                'data_points': len(df),
                'time_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'frequency': self._calculate_data_frequency(df),
                'volatility': {},
                'trends': {}
            }

            # Calculate trends for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'timestamp':
                    values = df[col].dropna()
                    if len(values) > 1:
                        trends['volatility'][col] = values.std()
                        trends['trends'][col] = self.correlation_analyzer._calculate_trend(values)

            return trends

        except Exception as e:
            cprint(f"❌ Error analyzing historical trends: {e}", "red")
            return {}

    def _calculate_data_frequency(self, df: pd.DataFrame) -> str:
        """Calculate data collection frequency"""
        if len(df) < 2:
            return 'unknown'

        time_diffs = df['timestamp'].diff().dropna()
        avg_interval = time_diffs.mean()

        if avg_interval < pd.Timedelta(minutes=5):
            return 'high_frequency'
        elif avg_interval < pd.Timedelta(hours=1):
            return 'hourly'
        elif avg_interval < pd.Timedelta(days=1):
            return 'daily'
        else:
            return 'irregular'

    def _analyze_risk_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk-specific metrics"""
        if not historical_data:
            return {}

        violations = []
        flux_levels = []

        for record in historical_data:
            if isinstance(record, dict):
                risk_check = record.get('risk_check', {})
                if isinstance(risk_check, dict):
                    violations.extend(risk_check.get('violations', []))

                flux_levels.append(record.get('flux_level', 0.5))

        return {
            'total_violations': len(violations),
            'unique_violation_types': len(set(str(v) for v in violations)),
            'average_flux_level': np.mean(flux_levels) if flux_levels else 0,
            'flux_volatility': np.std(flux_levels) if flux_levels else 0
        }

    def _analyze_sentiment_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment-specific metrics"""
        if not historical_data:
            return {}

        scores = []
        confidences = []

        for record in historical_data:
            if isinstance(record, dict):
                scores.append(record.get('overall_score', 0))
                confidences.append(record.get('confidence', 0.5))

        return {
            'average_sentiment': np.mean(scores) if scores else 0,
            'sentiment_volatility': np.std(scores) if scores else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'sentiment_distribution': self._calculate_sentiment_distribution(scores)
        }

    def _analyze_trading_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading-specific metrics"""
        # Placeholder for trading metrics analysis
        return {
            'total_signals': len(historical_data),
            'active_tokens': len(set(r.get('token') for r in historical_data if isinstance(r, dict)))
        }

    def _calculate_sentiment_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate sentiment distribution"""
        if not scores:
            return {'bullish': 0, 'neutral': 0, 'bearish': 0}

        return {
            'bullish': sum(1 for s in scores if s > 0.1),
            'neutral': sum(1 for s in scores if -0.1 <= s <= 0.1),
            'bearish': sum(1 for s in scores if s < -0.1)
        }

    def _calculate_agent_health_score(self, agent_name: str, latest_data: Optional[Dict[str, Any]],
                                    historical_data: List[Dict[str, Any]]) -> float:
        """Calculate overall health score for an agent"""
        score = 0.0

        # Data availability (30%)
        if latest_data:
            score += 0.3
        if historical_data:
            score += 0.1 * min(len(historical_data) / 100, 1)  # Up to 10% based on data volume

        # Data freshness (20%)
        if latest_data and latest_data.get('timestamp'):
            try:
                last_update = datetime.fromisoformat(latest_data['timestamp'].replace('Z', '+00:00'))
                hours_old = (datetime.now() - last_update).total_seconds() / 3600
                freshness_score = max(0, 1 - (hours_old / 24))  # Linear decay over 24 hours
                score += 0.2 * freshness_score
            except:
                pass

        # Data quality (50%)
        if agent_name == 'sentiment_agent' and latest_data:
            confidence = latest_data.get('confidence', 0)
            score += 0.5 * confidence
        elif agent_name == 'risk_agent' and latest_data:
            risk_check = latest_data.get('risk_check', {})
            if isinstance(risk_check, dict) and risk_check.get('ok', True):
                score += 0.5
            else:
                score += 0.25  # Partial credit for having risk data
        else:
            score += 0.25  # Default quality score

        return min(score, 1.0)

    def _generate_system_recommendations(self, overview: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations"""
        recommendations = []

        # Check agent health
        agent_status = overview.get('agent_status', {})
        unhealthy_agents = [name for name, status in agent_status.items()
                          if status.get('health_score', 0) < 0.5]

        if unhealthy_agents:
            recommendations.append(f"Investigate health issues with agents: {', '.join(unhealthy_agents)}")

        # Check performance
        performance = overview.get('performance_summary', {})
        if performance.get('max_drawdown_pct', 0) > 20:
            recommendations.append("High drawdown detected - consider risk management review")

        if performance.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy performance")

        # Check market regime
        regime = overview.get('market_analysis', {}).get('regime', {})
        regime_type = regime.get('current_regime', 'unknown')

        if regime_type in ['bear_distressed', 'bear_concerned']:
            recommendations.append("Bearish market conditions detected - consider defensive strategies")
        elif regime_type in ['bull_optimistic']:
            recommendations.append("Bullish market conditions - monitor for entry opportunities")

        # Data coverage
        health = overview.get('system_health', {})
        coverage = health.get('data_coverage', 0)
        if coverage < 0.7:
            recommendations.append("Low agent data coverage - ensure all agents are functioning")

        return recommendations

    def _generate_agent_recommendations(self, agent_name: str, report: Dict[str, Any]) -> List[str]:
        """Generate agent-specific recommendations"""
        recommendations = []

        health_score = report.get('health_score', 0)
        if health_score < 0.5:
            recommendations.append(f"Agent {agent_name} health is poor - investigate data collection issues")

        # Agent-specific recommendations
        if agent_name == 'risk_agent':
            metrics = report.get('performance_metrics', {})
            if metrics.get('total_violations', 0) > 10:
                recommendations.append("High number of risk violations - review risk parameters")

        elif agent_name == 'sentiment_agent':
            metrics = report.get('performance_metrics', {})
            if metrics.get('average_confidence', 0) < 0.6:
                recommendations.append("Low sentiment confidence - consider improving data sources")

        return recommendations

    def _format_performance_metrics(self, performance: PerformanceMetrics) -> Dict[str, Any]:
        """Format performance metrics for dashboard display"""
        return {
            'total_return_pct': round(performance.total_return, 2),
            'annualized_return_pct': round(performance.annualized_return, 2),
            'sharpe_ratio': round(performance.sharpe_ratio, 2),
            'sortino_ratio': round(performance.sortino_ratio, 2),
            'max_drawdown_pct': round(performance.max_drawdown, 2),
            'win_rate_pct': round(performance.win_rate, 2),
            'profit_factor': round(performance.profit_factor, 2),
            'total_trades': performance.total_trades,
            'value_at_risk_pct': round(performance.value_at_risk, 2),
            'expected_shortfall_pct': round(performance.expected_shortfall, 2)
        }

    def _extract_risk_metrics(self, performance: PerformanceMetrics) -> Dict[str, Any]:
        """Extract risk-focused metrics"""
        return {
            'volatility': round(performance.volatility * 100, 2),  # Convert to percentage
            'max_drawdown_pct': round(performance.max_drawdown, 2),
            'value_at_risk_pct': round(performance.value_at_risk, 2),
            'expected_shortfall_pct': round(performance.expected_shortfall, 2),
            'sharpe_ratio': round(performance.sharpe_ratio, 2),
            'sortino_ratio': round(performance.sortino_ratio, 2)
        }

    def _generate_system_alerts(self, dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system alerts based on dashboard data"""
        alerts = []

        # Performance alerts
        performance = dashboard.get('portfolio_performance', {})
        if performance.get('max_drawdown_pct', 0) > 25:
            alerts.append({
                'type': 'critical',
                'message': f"Critical drawdown: {performance['max_drawdown_pct']:.1f}%",
                'metric': 'max_drawdown'
            })

        if performance.get('sharpe_ratio', 0) < 0:
            alerts.append({
                'type': 'warning',
                'message': f"Negative Sharpe ratio: {performance['sharpe_ratio']:.2f}",
                'metric': 'sharpe_ratio'
            })

        # Agent health alerts
        agent_performance = dashboard.get('agent_performance', {})
        for agent_name, perf in agent_performance.items():
            if perf.get('health_score', 1) < 0.3:
                alerts.append({
                    'type': 'error',
                    'message': f"Agent {agent_name} is unhealthy (score: {perf['health_score']:.2f})",
                    'metric': 'agent_health',
                    'agent': agent_name
                })

        return alerts

    def _export_as_csv(self, data: Dict[str, Any], output_path: str):
        """Export data as CSV (simplified implementation)"""
        # This is a simplified CSV export - in production, you'd want more sophisticated handling
        import csv

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Key', 'Value'])

            # Flatten nested dictionary
            def flatten_dict(d, prefix=''):
                for key, value in d.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        flatten_dict(value, full_key)
                    elif isinstance(value, list):
                        writer.writerow([full_key, str(value)])
                    else:
                        writer.writerow([full_key, value])

            flatten_dict(data)