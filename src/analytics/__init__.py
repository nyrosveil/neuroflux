"""
🧠 NeuroFlux Analytics & Reporting Module
Comprehensive analytics engine for multi-agent trading system performance.

Built with love by Nyros Veil 🚀

Features:
- Centralized data aggregation from all agents
- Performance metrics calculation and analysis
- Real-time and historical analytics dashboard
- Cross-agent correlation analysis
- Automated report generation and insights
- Alert and notification system
"""

from .analytics_engine import AnalyticsEngine
from .metrics_calculator import MetricsCalculator
from .data_aggregator import DataAggregator
from .correlation_analyzer import CorrelationAnalyzer

__all__ = [
    'AnalyticsEngine',
    'MetricsCalculator',
    'DataAggregator',
    'CorrelationAnalyzer'
]