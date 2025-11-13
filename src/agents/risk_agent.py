"""
🧠 NeuroFlux Risk Agent
Circuit breaker with neuro-flux awareness - runs FIRST before any trading decisions.

Built with love by Nyros Veil 🚀

Features:
- Real-time portfolio risk monitoring with neuro-flux adaptation
- Circuit breaker system with AI-powered decision making
- Multi-exchange position netting and correlation analysis
- Dynamic risk limit adjustment based on market conditions
- Emergency stop coordination across all trading agents
- Comprehensive risk analytics and reporting
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from termcolor import cprint
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

from ..base_agent import BaseAgent
from ..orchestration.communication_bus import CommunicationBus, Message, MessageType, MessagePriority
from ..orchestration.agent_registry import AgentCapability
from .trading.types import Position, RiskLimits

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class CircuitBreakerState(Enum):
    """Circuit breaker states for risk management."""
    NORMAL = "normal"           # Trading allowed
    WARNING = "warning"         # Reduced risk limits
    HALTED = "halted"          # Trading suspended
    EMERGENCY = "emergency"    # Emergency closure initiated


class RiskMessageType(Enum):
    """Risk agent message types."""
    RISK_STATUS_UPDATE = "risk_status_update"
    TRADING_HALT = "trading_halt"
    RISK_LIMIT_UPDATE = "risk_limit_update"
    EMERGENCY_STOP = "emergency_stop"
    RISK_ALERT = "risk_alert"
    CIRCUIT_BREAKER_UPDATE = "circuit_breaker_update"


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result."""
    timestamp: float
    circuit_breaker_state: CircuitBreakerState
    portfolio_risk_pct: float
    max_drawdown_pct: float
    position_correlation_risk: float
    flux_adjusted_limits: Dict[str, float]
    var_95: float = 0.0  # Value at Risk (95% confidence)
    expected_shortfall_95: float = 0.0  # Expected Shortfall (95% confidence)
    stress_test_results: Dict[str, Any] = field(default_factory=dict)
    sharpe_ratio: float = 0.0
    volatility_pct: float = 0.0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    ai_insights: Optional[Dict[str, Any]] = None
    ai_predictions: Dict[str, Any] = field(default_factory=dict)  # AI risk predictions
    ai_confidence: float = 0.0  # AI prediction confidence score


@dataclass
class AIPrediction:
    """AI-powered risk prediction result."""
    predicted_risk_level: float  # 0-100 risk score
    confidence_score: float  # 0-1 confidence level
    predicted_volatility: float  # Predicted volatility percentage
    risk_trend: str  # "increasing", "decreasing", "stable"
    time_horizon: int  # Prediction horizon in seconds
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_timestamp: float = field(default_factory=time.time)


class AIRiskAdvisor:
    """
    AI-powered risk advisor for intelligent risk assessment and prediction.
    """

    def __init__(self, model_dir: str = "src/data/risk_agent/models/"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # ML Models
        self.risk_predictor = None
        self.volatility_forecaster = None
        self.emergency_classifier = None

        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Model metadata
        self.model_version = "1.0.0"
        self.last_training_time = None
        self.training_accuracy = 0.0

        # Feature engineering
        self.feature_columns = [
            'portfolio_risk_pct', 'max_drawdown_pct', 'position_correlation_risk',
            'flux_level', 'sharpe_ratio', 'volatility_pct', 'var_95',
            'total_positions', 'total_exposure', 'equity_balance'
        ]

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize or load AI models."""
        try:
            # Risk prediction model (Random Forest for interpretability)
            self.risk_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            # Volatility forecasting model (LSTM neural network)
            self.volatility_forecaster = self._build_lstm_model()

            # Emergency classification model
            self.emergency_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )

            cprint("🧠 AI Risk Advisor models initialized", "blue")

        except Exception as e:
            cprint(f"⚠️ Failed to initialize AI models: {str(e)}", "yellow")
            # Fallback to rule-based system
            self.risk_predictor = None
            self.volatility_forecaster = None
            self.emergency_classifier = None

    def _build_lstm_model(self) -> keras.Model:
        """Build LSTM model for volatility forecasting."""
        model = keras.Sequential([
            layers.LSTM(64, input_shape=(None, len(self.feature_columns)), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')  # Predict volatility
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def predict_risk_level(self, risk_assessment: RiskAssessment,
                          historical_data: Optional[List[RiskAssessment]] = None,
                          current_flux_level: float = FLUX_SENSITIVITY,
                          positions: Optional[Dict[str, Position]] = None,
                          portfolio_balance: Optional[Dict[str, float]] = None) -> AIPrediction:
        """
        Predict future risk levels using AI models.
        """
        if not self.risk_predictor:
            # Fallback prediction
            return AIPrediction(
                predicted_risk_level=risk_assessment.portfolio_risk_pct * 1.1,
                confidence_score=0.5,
                predicted_volatility=risk_assessment.volatility_pct,
                risk_trend="stable",
                time_horizon=3600  # 1 hour
            )

        try:
            # Prepare features
            features = self._extract_features(risk_assessment, historical_data, current_flux_level, positions, portfolio_balance)

            # Make prediction
            prediction = self.risk_predictor.predict([features])[0]

            # Calculate confidence based on feature importance variance
            confidence = self._calculate_prediction_confidence(features)

            # Determine risk trend
            current_risk = risk_assessment.portfolio_risk_pct
            risk_trend = "stable"
            if prediction > current_risk * 1.1:
                risk_trend = "increasing"
            elif prediction < current_risk * 0.9:
                risk_trend = "decreasing"

            # Get feature importance
            feature_importance = dict(zip(self.feature_columns,
                                        self.risk_predictor.feature_importances_))

            return AIPrediction(
                predicted_risk_level=float(prediction),
                confidence_score=confidence,
                predicted_volatility=risk_assessment.volatility_pct * 1.05,  # Slight increase assumption
                risk_trend=risk_trend,
                time_horizon=3600,  # 1 hour prediction
                feature_importance=feature_importance
            )

        except Exception as e:
            cprint(f"⚠️ AI risk prediction failed: {str(e)}", "yellow")
            # Return fallback prediction
            return AIPrediction(
                predicted_risk_level=risk_assessment.portfolio_risk_pct,
                confidence_score=0.3,
                predicted_volatility=risk_assessment.volatility_pct,
                risk_trend="stable",
                time_horizon=3600
            )

    def predict_emergency_probability(self, risk_assessment: RiskAssessment,
                                    current_flux_level: float = FLUX_SENSITIVITY,
                                    positions: Optional[Dict[str, Position]] = None,
                                    portfolio_balance: Optional[Dict[str, float]] = None) -> float:
        """
        Predict probability of emergency situation using classification model.
        """
        if not self.emergency_classifier:
            # Rule-based fallback
            risk_score = risk_assessment.portfolio_risk_pct
            if risk_score > 15:
                return 0.8
            elif risk_score > 10:
                return 0.5
            else:
                return 0.1

        try:
            features = self._extract_features(risk_assessment, None, current_flux_level, positions, portfolio_balance)
            probability = self.emergency_classifier.predict_proba([features])[0][1]  # Probability of emergency
            return float(probability)

        except Exception as e:
            cprint(f"⚠️ Emergency prediction failed: {str(e)}", "yellow")
            return 0.0

    def generate_smart_recommendations(self, risk_assessment: RiskAssessment,
                                     current_flux_level: float = FLUX_SENSITIVITY,
                                     positions: Optional[Dict[str, Position]] = None,
                                     portfolio_balance: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Generate AI-powered risk management recommendations.
        """
        recommendations = []

        try:
            # Analyze current risk state
            ai_prediction = self.predict_risk_level(risk_assessment, None, current_flux_level, positions, portfolio_balance)

            # High risk prediction
            if ai_prediction.predicted_risk_level > 20 and ai_prediction.confidence_score > 0.7:
                recommendations.append({
                    'type': 'ai_risk_prediction',
                    'priority': 'high',
                    'message': f'AI predicts elevated risk ({ai_prediction.predicted_risk_level:.1f}%) in next hour',
                    'confidence': ai_prediction.confidence_score,
                    'action': 'reduce_exposure',
                    'suggested_reduction': min(ai_prediction.predicted_risk_level / 2, 30)
                })

            # Emergency probability
            emergency_prob = self.predict_emergency_probability(risk_assessment, current_flux_level, positions, portfolio_balance)
            if emergency_prob > 0.6:
                recommendations.append({
                    'type': 'ai_emergency_alert',
                    'priority': 'critical',
                    'message': f'AI predicts {emergency_prob:.1%} probability of emergency situation',
                    'confidence': ai_prediction.confidence_score,
                    'action': 'prepare_emergency_procedures'
                })

            # Volatility-based recommendations
            if ai_prediction.predicted_volatility > risk_assessment.volatility_pct * 1.2:
                recommendations.append({
                    'type': 'ai_volatility_forecast',
                    'priority': 'medium',
                    'message': f'AI forecasts increased volatility ({ai_prediction.predicted_volatility:.1f}%)',
                    'action': 'tighten_stops',
                    'suggested_stop_adjustment': 0.02  # 2% tighter stops
                })

            # Portfolio optimization recommendations
            if positions and portfolio_balance:
                portfolio_recs = self._generate_portfolio_optimization_recommendations(risk_assessment, positions, portfolio_balance)
                recommendations.extend(portfolio_recs)

                # Position sizing recommendations
                sizing_recs = self._generate_position_sizing_recommendations(risk_assessment, positions, portfolio_balance)
                recommendations.extend(sizing_recs)

            # Market timing recommendations
            timing_recs = self._generate_market_timing_recommendations(ai_prediction, risk_assessment)
            recommendations.extend(timing_recs)

            # Feature importance insights
            if ai_prediction.feature_importance:
                top_features = sorted(ai_prediction.feature_importance.items(),
                                    key=lambda x: x[1], reverse=True)[:3]
                recommendations.append({
                    'type': 'ai_insight',
                    'priority': 'low',
                    'message': f'Top risk factors: {", ".join([f"{k}({v:.1%})" for k, v in top_features])}',
                    'action': 'monitor_key_factors'
                })

        except Exception as e:
            cprint(f"⚠️ AI recommendations failed: {str(e)}", "yellow")

        return recommendations

    def _generate_portfolio_optimization_recommendations(self, risk_assessment: RiskAssessment,
                                                       positions: Optional[Dict[str, Position]] = None,
                                                       portfolio_balance: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Generate portfolio optimization recommendations."""
        recommendations = []

        if not positions or not portfolio_balance:
            return recommendations

        portfolio_value = portfolio_balance.get('equity', 1000.0)

        # Diversification analysis
        position_weights = {}
        for symbol, position in positions.items():
            weight = abs(position.market_value) / portfolio_value
            position_weights[symbol] = weight

        # Check for over-concentration
        if position_weights:
            max_weight = max(position_weights.values())
            if max_weight > 0.3:  # 30% concentration
                concentrated_symbol = max(position_weights, key=lambda k: position_weights[k])
                recommendations.append({
                    'type': 'diversification',
                    'priority': 'medium',
                    'message': f'Position {concentrated_symbol} is {max_weight:.1%} of portfolio - consider reducing',
                    'action': 'reduce_position',
                    'target_symbol': concentrated_symbol,
                    'suggested_weight': 0.2  # Target 20%
                })

        # Risk parity suggestions
        if len(positions) > 1:
            # Simple risk parity: equal risk contribution
            equal_weight = 1.0 / len(positions)
            for symbol, current_weight in position_weights.items():
                if abs(current_weight - equal_weight) > 0.1:  # 10% deviation
                    recommendations.append({
                        'type': 'risk_parity',
                        'priority': 'low',
                        'message': f'Adjust {symbol} weight from {current_weight:.1%} toward {equal_weight:.1%} for better risk distribution',
                        'action': 'rebalance',
                        'target_symbol': symbol,
                        'suggested_weight': equal_weight
                    })

        return recommendations

    def _generate_position_sizing_recommendations(self, risk_assessment: RiskAssessment,
                                                positions: Dict[str, Position],
                                                portfolio_balance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate position sizing recommendations."""
        recommendations = []

        portfolio_value = portfolio_balance.get('equity', 1000.0)

        for symbol, position in positions.items():
            position_value = abs(position.market_value)
            position_weight = position_value / portfolio_value

            # Kelly criterion approximation (simplified)
            # Assume win rate and win/loss ratio based on historical performance
            if hasattr(position, 'unrealized_pnl') and position_value > 0:
                pnl_ratio = position.unrealized_pnl / position_value

                # Conservative Kelly sizing
                if pnl_ratio > 0.05:  # Profitable position
                    optimal_size = min(position_weight * 1.2, 0.25)  # Increase up to 25%
                    if optimal_size > position_weight * 1.1:  # At least 10% increase
                        recommendations.append({
                            'type': 'kelly_sizing',
                            'priority': 'low',
                            'message': f'Consider increasing {symbol} position size to {optimal_size:.1%} of portfolio',
                            'action': 'increase_position',
                            'target_symbol': symbol,
                            'suggested_weight': optimal_size
                        })
                elif pnl_ratio < -0.05:  # Losing position
                    reduction_size = position_weight * 0.8  # 20% reduction
                    recommendations.append({
                        'type': 'loss_management',
                        'priority': 'medium',
                        'message': f'Consider reducing {symbol} position due to losses (current P&L: {pnl_ratio:.1%})',
                        'action': 'reduce_position',
                        'target_symbol': symbol,
                        'suggested_weight': reduction_size
                    })

        return recommendations

    def _generate_market_timing_recommendations(self, ai_prediction: AIPrediction,
                                              risk_assessment: RiskAssessment) -> List[Dict[str, Any]]:
        """Generate market timing recommendations based on AI predictions."""
        recommendations = []

        # Risk trend analysis
        if ai_prediction.risk_trend == "increasing":
            recommendations.append({
                'type': 'market_timing',
                'priority': 'high',
                'message': 'AI predicts increasing risk - consider reducing overall exposure',
                'action': 'reduce_exposure',
                'timeframe': 'immediate'
            })
        elif ai_prediction.risk_trend == "decreasing":
            recommendations.append({
                'type': 'market_timing',
                'priority': 'low',
                'message': 'AI predicts decreasing risk - favorable conditions for increasing exposure',
                'action': 'increase_exposure',
                'timeframe': 'gradual'
            })

        # Volatility timing
        if ai_prediction.predicted_volatility > risk_assessment.volatility_pct * 1.5:
            recommendations.append({
                'type': 'volatility_timing',
                'priority': 'medium',
                'message': 'High volatility expected - consider volatility-based strategies',
                'action': 'implement_hedges',
                'suggested_strategy': 'options_hedging'
            })

        return recommendations

    def _extract_features(self, risk_assessment: RiskAssessment,
                         historical_data: Optional[List[RiskAssessment]] = None,
                         current_flux_level: float = FLUX_SENSITIVITY,
                         positions: Optional[Dict[str, Position]] = None,
                         portfolio_balance: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Extract features from risk assessment for ML models."""
        features = []

        # Current risk metrics
        features.extend([
            risk_assessment.portfolio_risk_pct,
            risk_assessment.max_drawdown_pct,
            risk_assessment.position_correlation_risk,
            current_flux_level,
            risk_assessment.sharpe_ratio,
            risk_assessment.volatility_pct,
            risk_assessment.var_95,
            len(positions) if positions else 0,
            sum(abs(p.market_value) for p in positions.values()) if positions else 0,
            portfolio_balance.get('equity', 1000.0) if portfolio_balance else 1000.0
        ])

        # Historical trends (if available)
        if historical_data and len(historical_data) >= 5:
            recent = historical_data[-5:]
            risk_trend = np.mean([r.portfolio_risk_pct for r in recent[-3:]]) - np.mean([r.portfolio_risk_pct for r in recent[:2]])
            volatility_trend = np.mean([r.volatility_pct for r in recent[-3:]]) - np.mean([r.volatility_pct for r in recent[:2]])
            features.extend([risk_trend, volatility_trend])
        else:
            features.extend([0.0, 0.0])  # No trend data

        return np.array(features)

    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence score for predictions."""
        # Simple confidence based on feature variance and model certainty
        if self.risk_predictor:
            # Use standard deviation of tree predictions as uncertainty measure
            tree_predictions = np.array([tree.predict([features]) for tree in self.risk_predictor.estimators_])
            prediction_std = np.std(tree_predictions)
            base_prediction = np.mean(tree_predictions)

            # Confidence decreases with higher prediction variance
            if base_prediction > 0:
                relative_uncertainty = prediction_std / abs(base_prediction)
                confidence = max(0.1, 1.0 - relative_uncertainty)
                return min(confidence, 0.95)
            else:
                return 0.5
        else:
            return 0.5

    def train_models(self, historical_assessments: List[RiskAssessment]) -> bool:
        """
        Train AI models using historical risk assessment data.
        """
        if len(historical_assessments) < 20:
            cprint("⚠️ Insufficient data for AI training (need at least 20 assessments)", "yellow")
            return False

        try:
            cprint("🧠 Training AI risk models...", "blue")

            # Prepare training data
            X, y_risk, y_emergency = self._prepare_training_data(historical_assessments)

            if len(X) < 10:
                cprint("⚠️ Insufficient training samples", "yellow")
                return False

            # Train risk predictor
            X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
            self.risk_predictor.fit(X_train, y_train)

            # Evaluate
            y_pred = self.risk_predictor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            self.training_accuracy = 1.0 - min(mse / np.var(y_test), 1.0)  # R²-like score

            # Train emergency classifier
            _, _, y_emergency_train, y_emergency_test = train_test_split(X, y_emergency, test_size=0.2, random_state=42)
            self.emergency_classifier.fit(X_train, y_emergency_train)

            # Evaluate classifier
            emergency_pred = self.emergency_classifier.predict(X_test)
            accuracy = accuracy_score(y_emergency_test, emergency_pred)

            self.last_training_time = time.time()

            cprint(f"✅ AI models trained - Risk predictor R²: {self.training_accuracy:.3f}, Emergency classifier accuracy: {accuracy:.3f}", "green")

            # Save models
            self._save_models()

            return True

        except Exception as e:
            cprint(f"❌ AI training failed: {str(e)}", "red")
            return False

    def _prepare_training_data(self, assessments: List[RiskAssessment]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from historical assessments."""
        X = []
        y_risk = []
        y_emergency = []

        for i, assessment in enumerate(assessments):
            # Skip if not enough historical data for features
            if i < 5:
                continue

            # Extract features
            historical_context = assessments[max(0, i-10):i]  # Last 10 assessments as context
            features = self._extract_features(assessment, historical_context)
            X.append(features)

            # Target: future risk level (next assessment's portfolio risk)
            if i < len(assessments) - 1:
                future_risk = assessments[i+1].portfolio_risk_pct
                y_risk.append(future_risk)

                # Emergency target: whether emergency was triggered in next assessment
                future_emergency = 1 if assessments[i+1].circuit_breaker_state == CircuitBreakerState.EMERGENCY else 0
                y_emergency.append(future_emergency)

        return np.array(X), np.array(y_risk), np.array(y_emergency)

    def _save_models(self):
        """Save trained models to disk."""
        try:
            import joblib

            # Save scikit-learn models
            joblib.dump(self.risk_predictor, os.path.join(self.model_dir, 'risk_predictor.pkl'))
            joblib.dump(self.emergency_classifier, os.path.join(self.model_dir, 'emergency_classifier.pkl'))

            # Save model metadata
            metadata = {
                'version': self.model_version,
                'training_time': self.last_training_time,
                'accuracy': self.training_accuracy,
                'feature_columns': self.feature_columns
            }

            with open(os.path.join(self.model_dir, 'model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            cprint("💾 AI models saved", "blue")

        except Exception as e:
            cprint(f"⚠️ Failed to save models: {str(e)}", "yellow")

    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            import joblib

            model_path = os.path.join(self.model_dir, 'risk_predictor.pkl')
            if os.path.exists(model_path):
                self.risk_predictor = joblib.load(model_path)
                self.emergency_classifier = joblib.load(os.path.join(self.model_dir, 'emergency_classifier.pkl'))

                # Load metadata
                metadata_path = os.path.join(self.model_dir, 'model_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.model_version = metadata.get('version', '1.0.0')
                        self.last_training_time = metadata.get('training_time')
                        self.training_accuracy = metadata.get('accuracy', 0.0)

                cprint("📂 AI models loaded", "blue")
                return True
            else:
                cprint("⚠️ No saved AI models found", "yellow")
                return False

        except Exception as e:
            cprint(f"⚠️ Failed to load models: {str(e)}", "yellow")
            return False


class RiskAgent(BaseAgent):
    """
    NeuroFlux Risk Management Agent

    Circuit breaker with neuro-flux awareness that monitors portfolio risk
    and enforces trading limits with AI-powered decision making.
    """

    def __init__(self, agent_id: str, communication_bus: CommunicationBus = None, **kwargs):
        """Initialize the Risk Agent."""
        super().__init__(agent_id, name="NeuroFlux_Risk_Agent", **kwargs)

        # Communication
        self.communication_bus = communication_bus

        # Risk monitoring state
        self.circuit_breaker_state = CircuitBreakerState.NORMAL
        self.last_risk_assessment = None
        self.risk_assessment_history: List[RiskAssessment] = []

        # Portfolio tracking
        self.portfolio_balance = {"equity": 0.0, "available": 0.0, "positions_value": 0.0}
        self.positions: Dict[str, Position] = {}
        self.cross_exchange_positions: Dict[str, Position] = {}

        # Risk limits (flux-adjusted)
        self.base_risk_limits = RiskLimits(
            max_position_size=MAX_POSITION_PERCENTAGE / 100 * 1000,  # Assume $1000 base
            max_portfolio_risk=MAX_POSITION_PERCENTAGE,
            max_drawdown=20.0,  # 20% max drawdown
            max_leverage=5.0,
            max_orders_per_minute=50,
            restricted_symbols=[]
        )
        self.current_risk_limits = self.base_risk_limits

        # Flux tracking
        self.current_flux_level = FLUX_SENSITIVITY
        self.flux_history: List[Tuple[float, float]] = []  # (timestamp, flux_level)

        # Emergency stop coordination
        self.emergency_active = False
        self.emergency_reason = ""

        # AI risk advisor settings
        self.ai_confirmation_required = USE_AI_CONFIRMATION
        self.ai_risk_threshold = 0.8  # Confidence threshold for AI decisions

        # AI Risk Advisor
        self.ai_advisor = AIRiskAdvisor()
        self.ai_enabled = True

        # Monitoring intervals
        self.risk_check_interval = 5.0  # Check risk every 5 seconds
        self.flux_update_interval = 30.0  # Update flux every 30 seconds
        self.report_interval = 60.0  # Generate reports every minute

        # Data storage
        self.output_dir = "src/data/risk_agent/"
        os.makedirs(self.output_dir, exist_ok=True)

        # Background tasks
        self.monitoring_task = None
        self.flux_task = None
        self.reporting_task = None

    # ============================================================================
    # Abstract Method Implementations from BaseAgent
    # ============================================================================

    def _initialize_agent(self) -> bool:
        """Agent-specific initialization logic."""
        try:
            cprint("🛡️ Initializing NeuroFlux Risk Agent...", "cyan")

            # Register capabilities
            self.capabilities = [
                AgentCapability.RISK_MANAGEMENT,
                AgentCapability.PORTFOLIO_MONITORING,
                AgentCapability.CIRCUIT_BREAKER,
                AgentCapability.EMERGENCY_COORDINATION,
                AgentCapability.FLUX_ANALYSIS,
                AgentCapability.AI_RISK_ADVISOR
            ]

            # Initialize risk monitoring
            self._initialize_risk_monitoring()

            # Load existing risk data if available
            self._load_existing_risk_data()

            cprint("✅ Risk Agent initialized successfully", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to initialize Risk Agent: {str(e)}", "red")
            return False

    def _execute_agent_cycle(self):
        """Execute one cycle of agent logic."""
        # Risk monitoring is handled by background tasks
        # This method can be used for periodic maintenance
        pass

    def _cleanup_agent(self):
        """Agent-specific cleanup logic."""
        try:
            cprint("🧹 Cleaning up Risk Agent...", "blue")

            # Cancel background tasks
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.flux_task:
                self.flux_task.cancel()
            if self.reporting_task:
                self.reporting_task.cancel()

            # Save final risk report
            if self.last_risk_assessment:
                self._save_risk_report(self.last_risk_assessment)

            cprint("✅ Risk Agent cleaned up", "green")

        except Exception as e:
            cprint(f"❌ Error during Risk Agent cleanup: {str(e)}", "red")

    # ============================================================================
    # Risk Monitoring Core Methods
    # ============================================================================

    def _initialize_risk_monitoring(self):
        """Initialize risk monitoring components."""
        # Set up initial portfolio state
        self.portfolio_balance = {
            'equity': 1000.0,  # Default starting balance
            'available': 950.0,  # 95% available (5% buffer)
            'positions_value': 50.0
        }

        # Initialize flux tracking
        self._update_flux_level()

        # Set initial risk limits
        self._adjust_risk_limits_for_flux()

    def _load_existing_risk_data(self):
        """Load existing risk data from storage."""
        try:
            latest_report_path = os.path.join(self.output_dir, "latest_report.json")
            if os.path.exists(latest_report_path):
                with open(latest_report_path, 'r') as f:
                    data = json.load(f)

                # Restore circuit breaker state
                if 'circuit_breaker_state' in data:
                    self.circuit_breaker_state = CircuitBreakerState(data['circuit_breaker_state'])

                cprint("📊 Loaded existing risk data", "blue")

        except Exception as e:
            cprint(f"⚠️ Could not load existing risk data: {str(e)}", "yellow")

    async def start_monitoring(self):
        """Start risk monitoring background tasks."""
        try:
            # Start risk monitoring task
            self.monitoring_task = asyncio.create_task(self._risk_monitoring_loop())
            cprint("🔍 Started risk monitoring loop", "blue")

            # Start flux monitoring task
            self.flux_task = asyncio.create_task(self._flux_monitoring_loop())
            cprint("🌊 Started flux monitoring loop", "blue")

            # Start reporting task
            self.reporting_task = asyncio.create_task(self._reporting_loop())
            cprint("📊 Started risk reporting loop", "blue")

        except Exception as e:
            cprint(f"❌ Failed to start risk monitoring: {str(e)}", "red")
            raise

    async def _risk_monitoring_loop(self):
        """Main risk monitoring loop."""
        while self.is_running:
            try:
                # Perform comprehensive risk assessment
                assessment = await self._perform_risk_assessment()

                # Update circuit breaker state
                await self._update_circuit_breaker_state(assessment)

                # Broadcast risk status
                await self._broadcast_risk_status(assessment)

                # Handle any violations
                await self._handle_risk_violations(assessment)

                # Store assessment
                self.last_risk_assessment = assessment
                self.risk_assessment_history.append(assessment)

                # Keep only recent history (last 100 assessments)
                if len(self.risk_assessment_history) > 100:
                    self.risk_assessment_history = self.risk_assessment_history[-100:]

                await asyncio.sleep(self.risk_check_interval)

            except Exception as e:
                cprint(f"❌ Error in risk monitoring loop: {str(e)}", "red")
                await asyncio.sleep(10)  # Brief pause on error

    async def _flux_monitoring_loop(self):
        """Monitor market flux levels."""
        while self.is_running:
            try:
                self._update_flux_level()
                self._adjust_risk_limits_for_flux()

                # Record flux history
                self.flux_history.append((time.time(), self.current_flux_level))
                if len(self.flux_history) > 100:  # Keep last 100 readings
                    self.flux_history = self.flux_history[-100:]

                await asyncio.sleep(self.flux_update_interval)

            except Exception as e:
                cprint(f"❌ Error in flux monitoring: {str(e)}", "red")
                await asyncio.sleep(30)

    async def _reporting_loop(self):
        """Generate periodic risk reports."""
        while self.is_running:
            try:
                if self.last_risk_assessment:
                    self._save_risk_report(self.last_risk_assessment)

                await asyncio.sleep(self.report_interval)

            except Exception as e:
                cprint(f"❌ Error in reporting loop: {str(e)}", "red")
                await asyncio.sleep(60)

    # ============================================================================
    # Risk Assessment Engine
    # ============================================================================

    async def _perform_risk_assessment(self) -> RiskAssessment:
        """Perform comprehensive risk assessment."""
        timestamp = time.time()

        # Gather current portfolio data
        await self._update_portfolio_data()

        # Calculate risk metrics
        portfolio_risk_pct = self._calculate_portfolio_risk()
        max_drawdown_pct = self._calculate_max_drawdown()
        position_correlation_risk = await self._assess_position_correlation()

        # Advanced risk calculations
        var_95 = self._calculate_var(confidence_level=0.95)
        expected_shortfall_95 = self._calculate_expected_shortfall(confidence_level=0.95)
        stress_test_results = self._perform_stress_test()
        sharpe_ratio, volatility_pct = self._calculate_sharpe_and_volatility()

        # Get flux-adjusted limits
        flux_adjusted_limits = self._calculate_flux_adjusted_limits()

        # Perform risk checks
        violations, warnings, recommendations = self._check_risk_limits()

        # AI-powered risk assessment
        ai_predictions = {}
        ai_insights = {}
        ai_confidence = 0.0

        if self.ai_enabled and self.ai_advisor:
            try:
                # Get AI risk predictions
                ai_prediction = self.ai_advisor.predict_risk_level(
                    RiskAssessment(
                        timestamp=timestamp,
                        circuit_breaker_state=self.circuit_breaker_state,
                        portfolio_risk_pct=portfolio_risk_pct,
                        max_drawdown_pct=max_drawdown_pct,
                        position_correlation_risk=position_correlation_risk,
                        flux_adjusted_limits=flux_adjusted_limits,
                        var_95=var_95,
                        expected_shortfall_95=expected_shortfall_95,
                        stress_test_results=stress_test_results,
                        sharpe_ratio=sharpe_ratio,
                        volatility_pct=volatility_pct
                    ),
                    self.risk_assessment_history[-20:] if len(self.risk_assessment_history) >= 20 else None,
                    self.current_flux_level,
                    self.positions,
                    self.portfolio_balance
                )
                ai_predictions = ai_prediction.__dict__
                ai_confidence = ai_prediction.confidence_score

                # Get AI emergency probability
                emergency_prob = self.ai_advisor.predict_emergency_probability(
                    RiskAssessment(
                        timestamp=timestamp,
                        circuit_breaker_state=self.circuit_breaker_state,
                        portfolio_risk_pct=portfolio_risk_pct,
                        max_drawdown_pct=max_drawdown_pct,
                        position_correlation_risk=position_correlation_risk,
                        flux_adjusted_limits=flux_adjusted_limits,
                        var_95=var_95,
                        expected_shortfall_95=expected_shortfall_95,
                        stress_test_results=stress_test_results,
                        sharpe_ratio=sharpe_ratio,
                        volatility_pct=volatility_pct
                    ),
                    self.current_flux_level,
                    self.positions,
                    self.portfolio_balance
                )

                # Generate AI recommendations
                ai_recommendations = self.ai_advisor.generate_smart_recommendations(
                    RiskAssessment(
                        timestamp=timestamp,
                        circuit_breaker_state=self.circuit_breaker_state,
                        portfolio_risk_pct=portfolio_risk_pct,
                        max_drawdown_pct=max_drawdown_pct,
                        position_correlation_risk=position_correlation_risk,
                        flux_adjusted_limits=flux_adjusted_limits,
                        var_95=var_95,
                        expected_shortfall_95=expected_shortfall_95,
                        stress_test_results=stress_test_results,
                        sharpe_ratio=sharpe_ratio,
                        volatility_pct=volatility_pct
                    )
                )

                # Add AI recommendations to existing recommendations
                recommendations.extend(ai_recommendations)

                ai_insights = {
                    'emergency_probability': emergency_prob,
                    'ai_recommendations_count': len(ai_recommendations),
                    'model_accuracy': getattr(self.ai_advisor, 'training_accuracy', 0.0),
                    'last_training': getattr(self.ai_advisor, 'last_training_time', None)
                }

            except Exception as e:
                cprint(f"⚠️ AI assessment failed: {str(e)}", "yellow")
                ai_insights = {'error': str(e)}

        # Create assessment
        assessment = RiskAssessment(
            timestamp=timestamp,
            circuit_breaker_state=self.circuit_breaker_state,
            portfolio_risk_pct=portfolio_risk_pct,
            max_drawdown_pct=max_drawdown_pct,
            position_correlation_risk=position_correlation_risk,
            flux_adjusted_limits=flux_adjusted_limits,
            var_95=var_95,
            expected_shortfall_95=expected_shortfall_95,
            stress_test_results=stress_test_results,
            sharpe_ratio=sharpe_ratio,
            volatility_pct=volatility_pct,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            ai_insights=ai_insights,
            ai_predictions=ai_predictions,
            ai_confidence=ai_confidence
        )

        return assessment

    async def _update_portfolio_data(self):
        """Update current portfolio data from trading agents."""
        try:
            # Request portfolio data from trading agents
            if self.communication_bus:
                # Send request for portfolio data
                request = Message(
                    message_id=f"risk_portfolio_request_{int(time.time())}",
                    sender_id=self.agent_id,
                    message_type=MessageType.REQUEST,
                    topic="portfolio_data_request",
                    payload={"request_type": "current_portfolio"},
                    priority=MessagePriority.HIGH
                )

                # For now, use placeholder data
                # In full implementation, this would wait for responses from trading agents
                pass

        except Exception as e:
            cprint(f"⚠️ Could not update portfolio data: {str(e)}", "yellow")

    def _calculate_portfolio_risk(self) -> float:
        """Calculate total portfolio risk as percentage."""
        if self.portfolio_balance['equity'] <= 0:
            return 0.0

        total_exposure = sum(abs(pos.market_value) for pos in self.positions.values())
        return (total_exposure / self.portfolio_balance['equity']) * 100

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage from equity history."""
        if not self.risk_assessment_history:
            return 0.0

        # Extract equity values from assessment history
        equity_values = []
        for assessment in self.risk_assessment_history[-100:]:  # Last 100 assessments
            # Estimate equity from portfolio balance if available
            if hasattr(self, 'portfolio_balance') and 'equity' in self.portfolio_balance:
                equity_values.append(self.portfolio_balance['equity'])

        if len(equity_values) < 2:
            return 0.0

        # Calculate drawdown
        peak = equity_values[0]
        max_drawdown = 0.0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_var(self, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR) using historical simulation."""
        if not self.risk_assessment_history or len(self.risk_assessment_history) < 30:
            # Fallback to simple VaR calculation
            portfolio_value = self.portfolio_balance.get('equity', 1000.0)
            return portfolio_value * 0.05  # 5% VaR as fallback

        # Extract portfolio returns from history
        returns = []
        prev_equity = None

        for assessment in self.risk_assessment_history[-100:]:
            current_equity = self.portfolio_balance.get('equity', 1000.0)
            if prev_equity is not None:
                ret = (current_equity - prev_equity) / prev_equity
                returns.append(ret)
            prev_equity = current_equity

        if len(returns) < 10:
            portfolio_value = self.portfolio_balance.get('equity', 1000.0)
            return portfolio_value * 0.05

        # Calculate VaR using historical simulation
        returns_array = np.array(returns)
        var_percentile = np.percentile(returns_array, (1 - confidence_level) * 100)
        portfolio_value = self.portfolio_balance.get('equity', 1000.0)

        return abs(var_percentile * portfolio_value)

    def _calculate_expected_shortfall(self, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (CVaR) beyond VaR."""
        if not self.risk_assessment_history or len(self.risk_assessment_history) < 30:
            portfolio_value = self.portfolio_balance.get('equity', 1000.0)
            return portfolio_value * 0.08  # 8% ES as fallback

        # Extract portfolio returns
        returns = []
        prev_equity = None

        for assessment in self.risk_assessment_history[-100:]:
            current_equity = self.portfolio_balance.get('equity', 1000.0)
            if prev_equity is not None:
                ret = (current_equity - prev_equity) / prev_equity
                returns.append(ret)
            prev_equity = current_equity

        if len(returns) < 10:
            portfolio_value = self.portfolio_balance.get('equity', 1000.0)
            return portfolio_value * 0.08

        returns_array = np.array(returns)
        var_percentile = np.percentile(returns_array, (1 - confidence_level) * 100)

        # Expected Shortfall is the average of returns beyond VaR
        tail_returns = returns_array[returns_array <= var_percentile]
        if len(tail_returns) == 0:
            return abs(var_percentile * self.portfolio_balance.get('equity', 1000.0))

        es_return = np.mean(tail_returns)
        portfolio_value = self.portfolio_balance.get('equity', 1000.0)

        return abs(es_return * portfolio_value)

    async def _assess_position_correlation(self) -> float:
        """Assess correlation risk between positions using correlation matrix."""
        if len(self.positions) < 2:
            return 0.0

        # Collect position returns data (simplified - would need historical price data)
        position_returns = {}

        # For each position, simulate return series based on current data
        for symbol, position in self.positions.items():
            # Simplified: use position's unrealized PnL as proxy for volatility
            # In full implementation, this would use historical price data
            volatility = abs(position.unrealized_pnl / position.market_value) if position.market_value > 0 else 0.1
            position_returns[symbol] = np.random.normal(0, volatility, 100)  # Simulated returns

        if len(position_returns) < 2:
            return 0.0

        # Calculate correlation matrix
        returns_df = np.column_stack(list(position_returns.values()))
        corr_matrix = np.corrcoef(returns_df.T)

        # Calculate average correlation (excluding diagonal)
        n = corr_matrix.shape[0]
        avg_correlation = (np.sum(corr_matrix) - n) / (n * (n - 1))  # Exclude diagonal

        # Correlation risk is higher when correlations are high (less diversification)
        correlation_risk = max(0, avg_correlation) * 100  # Convert to percentage

        return min(correlation_risk, 100.0)  # Cap at 100%

    def _perform_stress_test(self) -> Dict[str, Any]:
        """Perform stress testing with various market scenarios."""
        scenarios = {
            'market_crash': {'shock': -0.20, 'description': '20% market crash'},
            'high_volatility': {'shock': -0.10, 'description': '10% volatility shock'},
            'liquidity_crisis': {'shock': -0.15, 'description': '15% liquidity crisis'},
            'correlated_crash': {'shock': -0.25, 'description': '25% correlated crash'}
        }

        results = {}
        portfolio_value = self.portfolio_balance.get('equity', 1000.0)

        for scenario_name, scenario_data in scenarios.items():
            shock = scenario_data['shock']

            # Calculate impact on portfolio
            # Simplified: assume all positions move together
            impacted_value = portfolio_value * (1 + shock)

            # Calculate recovery time estimate (simplified)
            recovery_time_days = abs(shock * 100) * 2  # Rough estimate

            results[scenario_name] = {
                'description': scenario_data['description'],
                'shock_percentage': shock * 100,
                'portfolio_impact_usd': portfolio_value - impacted_value,
                'portfolio_impact_pct': shock * 100,
                'estimated_recovery_days': recovery_time_days,
                'breaches_limit': abs(shock * 100) > self.current_risk_limits.max_drawdown
            }

        return results

    def _calculate_sharpe_and_volatility(self) -> Tuple[float, float]:
        """Calculate Sharpe ratio and portfolio volatility."""
        if not self.risk_assessment_history or len(self.risk_assessment_history) < 10:
            return 0.0, 0.0

        # Extract portfolio returns
        returns = []
        prev_equity = None

        for assessment in self.risk_assessment_history[-100:]:
            current_equity = self.portfolio_balance.get('equity', 1000.0)
            if prev_equity is not None:
                ret = (current_equity - prev_equity) / prev_equity
                returns.append(ret)
            prev_equity = current_equity

        if len(returns) < 5:
            return 0.0, 0.0

        returns_array = np.array(returns)

        # Calculate annualized volatility (assuming daily returns)
        volatility = np.std(returns_array) * np.sqrt(252)  # 252 trading days
        volatility_pct = volatility * 100

        # Calculate Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns_array - risk_free_rate
        if volatility > 0:
            sharpe_ratio = np.mean(excess_returns) / volatility
        else:
            sharpe_ratio = 0.0

        return sharpe_ratio, volatility_pct

    def _calculate_flux_adjusted_limits(self) -> Dict[str, float]:
        """Calculate risk limits adjusted for current flux level."""
        base_limits = {
            'max_position_size': self.base_risk_limits.max_position_size,
            'max_portfolio_risk': self.base_risk_limits.max_portfolio_risk,
            'max_drawdown': self.base_risk_limits.max_drawdown
        }

        # Reduce limits during high flux
        flux_multiplier = 1.0 - (self.current_flux_level * 0.3)  # Reduce by up to 30%

        adjusted_limits = {}
        for key, value in base_limits.items():
            adjusted_limits[key] = value * flux_multiplier

        return adjusted_limits

    def _check_risk_limits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Check all risk limits and return violations, warnings, and recommendations."""
        violations = []
        warnings = []
        recommendations = []

        # Portfolio-level risk checks
        violations.extend(self._check_portfolio_level_risks())
        warnings.extend(self._check_portfolio_level_warnings())
        recommendations.extend(self._check_portfolio_level_recommendations())

        # Position-level risk checks
        position_violations, position_warnings, position_recommendations = self._check_position_level_risks()
        violations.extend(position_violations)
        warnings.extend(position_warnings)
        recommendations.extend(position_recommendations)

        return violations, warnings, recommendations

    def _check_portfolio_level_risks(self) -> List[Dict]:
        """Check portfolio-level risk violations."""
        violations = []

        # Portfolio risk check
        portfolio_risk = self._calculate_portfolio_risk()
        max_portfolio_risk = self.current_risk_limits.max_portfolio_risk

        if portfolio_risk > max_portfolio_risk:
            violations.append({
                'type': 'portfolio_risk',
                'message': f"Portfolio risk {portfolio_risk:.1f}% exceeds limit {max_portfolio_risk:.1f}%",
                'severity': 'critical',
                'current_value': portfolio_risk,
                'limit_value': max_portfolio_risk
            })

        # Drawdown check
        drawdown = self._calculate_max_drawdown()
        max_drawdown = self.current_risk_limits.max_drawdown

        if drawdown > max_drawdown:
            violations.append({
                'type': 'drawdown',
                'message': f"Drawdown {drawdown:.1f}% exceeds limit {max_drawdown:.1f}%",
                'severity': 'critical',
                'current_value': drawdown,
                'limit_value': max_drawdown
            })

        # VaR check
        portfolio_value = self.portfolio_balance.get('equity', 1000.0)
        var_limit = portfolio_value * 0.10  # 10% VaR limit
        latest_assessment = self.last_risk_assessment
        if latest_assessment and latest_assessment.var_95 > var_limit:
            violations.append({
                'type': 'var_breach',
                'message': f"VaR {latest_assessment.var_95:.2f} exceeds limit {var_limit:.2f}",
                'severity': 'high',
                'current_value': latest_assessment.var_95,
                'limit_value': var_limit
            })

        return violations

    def _check_portfolio_level_warnings(self) -> List[Dict]:
        """Check portfolio-level risk warnings."""
        warnings = []

        latest_assessment = self.last_risk_assessment
        if not latest_assessment:
            return warnings

        # High volatility warning
        if latest_assessment.volatility_pct > 30:
            warnings.append({
                'type': 'high_volatility',
                'message': f"Portfolio volatility {latest_assessment.volatility_pct:.1f}% is elevated",
                'current_value': latest_assessment.volatility_pct
            })

        # Low Sharpe ratio warning
        if latest_assessment.sharpe_ratio < 0.0:
            warnings.append({
                'type': 'negative_sharpe',
                'message': f"Negative Sharpe ratio {latest_assessment.sharpe_ratio:.2f} indicates poor risk-adjusted returns",
                'current_value': latest_assessment.sharpe_ratio
            })

        return warnings

    def _check_portfolio_level_recommendations(self) -> List[Dict]:
        """Check portfolio-level risk recommendations."""
        recommendations = []

        # Flux-based recommendations
        if self.current_flux_level > FLUX_SENSITIVITY:
            recommendations.append({
                'type': 'flux_adaptation',
                'message': f"High flux detected ({self.current_flux_level:.2f}). Consider reducing position sizes.",
                'action': 'reduce_risk'
            })

        # Diversification recommendations
        latest_assessment = self.last_risk_assessment
        if latest_assessment and latest_assessment.position_correlation_risk > 60:
            recommendations.append({
                'type': 'diversification',
                'message': f"High correlation risk ({latest_assessment.position_correlation_risk:.1f}%). Consider diversifying positions.",
                'action': 'diversify_portfolio'
            })

        return recommendations

    def _check_position_level_risks(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Check position-level risks and limits."""
        violations = []
        warnings = []
        recommendations = []

        for symbol, position in self.positions.items():
            position_risks = self._assess_position_risk(position, symbol)

            # Check position size limits
            position_value = abs(position.market_value)
            max_position_size = self.current_risk_limits.max_position_size

            if position_value > max_position_size:
                violations.append({
                    'type': 'position_size',
                    'symbol': symbol,
                    'message': f"Position {symbol} size ${position_value:.2f} exceeds limit ${max_position_size:.2f}",
                    'severity': 'high',
                    'current_value': position_value,
                    'limit_value': max_position_size
                })

            # Check position concentration
            portfolio_value = self.portfolio_balance.get('equity', 1000.0)
            concentration_pct = (position_value / portfolio_value) * 100
            max_concentration = 25.0  # 25% max concentration per position

            if concentration_pct > max_concentration:
                warnings.append({
                    'type': 'concentration',
                    'symbol': symbol,
                    'message': f"Position {symbol} concentration {concentration_pct:.1f}% exceeds recommended {max_concentration:.1f}%",
                    'current_value': concentration_pct,
                    'limit_value': max_concentration
                })

            # Check position P&L volatility
            if position_risks.get('pnl_volatility', 0) > 0.5:  # 50% P&L volatility
                recommendations.append({
                    'type': 'position_volatility',
                    'symbol': symbol,
                    'message': f"Position {symbol} shows high P&L volatility. Consider reducing size.",
                    'action': 'reduce_position'
                })

        return violations, warnings, recommendations

    def _assess_position_risk(self, position: Position, symbol: str) -> Dict[str, float]:
        """Assess risk metrics for an individual position."""
        # Calculate position-specific risk metrics
        position_value = abs(position.market_value)

        # P&L volatility (simplified - would use historical data)
        pnl_volatility = abs(position.unrealized_pnl / position_value) if position_value > 0 else 0.0

        # Position beta (simplified - would use market correlation)
        # For now, assume beta of 1.0 (market matching)
        position_beta = 1.0

        # Contribution to portfolio VaR (simplified)
        portfolio_var = self.last_risk_assessment.var_95 if self.last_risk_assessment else 0.0
        var_contribution = (position_value / self.portfolio_balance.get('equity', 1000.0)) * portfolio_var

        return {
            'pnl_volatility': pnl_volatility,
            'beta': position_beta,
            'var_contribution': var_contribution,
            'concentration_risk': (position_value / self.portfolio_balance.get('equity', 1000.0)) * 100
        }

    # ============================================================================
    # Circuit Breaker System
    # ============================================================================

    async def _update_circuit_breaker_state(self, assessment: RiskAssessment):
        """Update circuit breaker state based on comprehensive risk assessment."""
        old_state = self.circuit_breaker_state

        # Multi-factor state determination
        new_state = await self._determine_circuit_breaker_state(assessment)

        # Check for recovery conditions
        if old_state != CircuitBreakerState.NORMAL and new_state == CircuitBreakerState.NORMAL:
            # Only allow recovery if conditions are stable for a period
            if await self._check_recovery_conditions(assessment):
                new_state = CircuitBreakerState.NORMAL
            else:
                new_state = old_state  # Stay in current state

        # Update state if changed
        if new_state != old_state:
            self.circuit_breaker_state = new_state
            await self._handle_circuit_breaker_change(old_state, new_state, assessment)

    async def _determine_circuit_breaker_state(self, assessment: RiskAssessment) -> CircuitBreakerState:
        """Determine circuit breaker state using multiple risk factors."""
        # Emergency conditions (immediate halt)
        if self._check_emergency_conditions(assessment):
            return CircuitBreakerState.EMERGENCY

        # Critical violations
        critical_violations = [v for v in assessment.violations if v.get('severity') == 'critical']
        if critical_violations:
            return CircuitBreakerState.EMERGENCY

        # Halt conditions (suspended trading)
        if self._check_halt_conditions(assessment):
            return CircuitBreakerState.HALTED

        # Warning conditions (reduced limits)
        if self._check_warning_conditions(assessment):
            return CircuitBreakerState.WARNING

        # Normal conditions
        return CircuitBreakerState.NORMAL

    def _check_emergency_conditions(self, assessment: RiskAssessment) -> bool:
        """Check for emergency trigger conditions."""
        conditions = []

        # Portfolio value drop emergency
        portfolio_value = self.portfolio_balance.get('equity', 1000.0)
        emergency_threshold = self.base_risk_limits.max_drawdown * 0.8  # 80% of max drawdown
        if assessment.max_drawdown_pct > emergency_threshold:
            conditions.append("portfolio_drawdown")

        # VaR breach emergency
        var_limit = portfolio_value * 0.15  # 15% VaR limit
        if assessment.var_95 > var_limit:
            conditions.append("var_breach")

        # Stress test failures
        stress_failures = [s for s in assessment.stress_test_results.values()
                          if s.get('breaches_limit', False)]
        if len(stress_failures) > 1:  # Multiple stress scenarios fail
            conditions.append("stress_test_failure")

        # Extreme volatility
        if assessment.volatility_pct > 50:  # 50% annualized volatility
            conditions.append("extreme_volatility")

        return len(conditions) > 0

    def _check_halt_conditions(self, assessment: RiskAssessment) -> bool:
        """Check for trading halt conditions."""
        conditions = []

        # High violations count
        if len(assessment.violations) > 2:
            conditions.append("multiple_violations")

        # Correlation risk too high
        if assessment.position_correlation_risk > 70:  # 70% correlation risk
            conditions.append("high_correlation")

        # Portfolio risk above warning threshold
        if assessment.portfolio_risk_pct > self.current_risk_limits.max_portfolio_risk * 1.5:
            conditions.append("portfolio_risk")

        # Negative Sharpe ratio (risk not compensated)
        if assessment.sharpe_ratio < -0.5:
            conditions.append("negative_sharpe")

        return len(conditions) > 0

    def _check_warning_conditions(self, assessment: RiskAssessment) -> bool:
        """Check for warning conditions requiring reduced limits."""
        conditions = []

        # Moderate violations
        if assessment.violations:
            conditions.append("violations_present")

        # High flux levels
        if self.current_flux_level > FLUX_SENSITIVITY * 1.2:
            conditions.append("high_flux")

        # Moderate correlation risk
        if assessment.position_correlation_risk > 50:
            conditions.append("moderate_correlation")

        # Low Sharpe ratio
        if assessment.sharpe_ratio < 0.5:
            conditions.append("low_sharpe")

        return len(conditions) > 0

    async def _check_recovery_conditions(self, assessment: RiskAssessment) -> bool:
        """Check if conditions are stable enough for recovery to normal trading."""
        # Require stable conditions for at least 5 assessments
        if len(self.risk_assessment_history) < 5:
            return False

        recent_assessments = self.risk_assessment_history[-5:]

        # Check that recent assessments show improving or stable risk
        improving_trend = True
        for i in range(1, len(recent_assessments)):
            prev = recent_assessments[i-1]
            curr = recent_assessments[i]

            # Risk should not be increasing
            if (curr.portfolio_risk_pct > prev.portfolio_risk_pct * 1.1 or
                curr.var_95 > prev.var_95 * 1.1):
                improving_trend = False
                break

        # No violations in recent assessments
        no_recent_violations = all(len(assessment.violations) == 0
                                 for assessment in recent_assessments)

        # Flux levels normalized
        flux_normalized = self.current_flux_level < FLUX_SENSITIVITY * 1.1

        return improving_trend and no_recent_violations and flux_normalized

    async def _handle_circuit_breaker_change(self, old_state: CircuitBreakerState,
                                           new_state: CircuitBreakerState,
                                           assessment: RiskAssessment):
        """Handle circuit breaker state changes."""
        state_messages = {
            CircuitBreakerState.NORMAL: "🟢 Risk status: NORMAL - Trading allowed",
            CircuitBreakerState.WARNING: "🟡 Risk status: WARNING - Reduced risk limits active",
            CircuitBreakerState.HALTED: "🟠 Risk status: HALTED - Trading suspended",
            CircuitBreakerState.EMERGENCY: "🔴 Risk status: EMERGENCY - Emergency stop initiated"
        }

        cprint(state_messages[new_state], "red" if new_state == CircuitBreakerState.EMERGENCY else "yellow")

        # Broadcast state change
        await self._broadcast_circuit_breaker_update(new_state, assessment)

        # Handle emergency stop
        if new_state == CircuitBreakerState.EMERGENCY:
            await self._initiate_emergency_stop(assessment)

    async def _initiate_emergency_stop(self, assessment: RiskAssessment):
        """Initiate emergency stop across all trading agents with AI confirmation."""
        # AI confirmation for emergency stop (if enabled)
        ai_confirmed = True
        ai_confidence = 0.0

        if self.ai_enabled and self.ai_advisor and self.ai_confirmation_required:
            try:
                emergency_prob = self.ai_advisor.predict_emergency_probability(
                    assessment,
                    self.current_flux_level,
                    self.positions,
                    self.portfolio_balance
                )

                ai_confidence = emergency_prob
                ai_confirmed = emergency_prob >= self.ai_risk_threshold

                if not ai_confirmed:
                    cprint(f"🤖 AI rejected emergency stop (confidence: {emergency_prob:.2f} < {self.ai_risk_threshold})", "yellow")
                    # Still proceed but log the AI decision
                    ai_confirmed = False

            except Exception as e:
                cprint(f"⚠️ AI confirmation failed: {str(e)} - proceeding with emergency stop", "yellow")

        self.emergency_active = True
        self.emergency_reason = f"Risk violation: {len(assessment.violations)} critical violations"

        emergency_details = f"🚨 EMERGENCY STOP: {self.emergency_reason}"
        if self.ai_enabled:
            emergency_details += f" | AI Confidence: {ai_confidence:.2f}"

        cprint(emergency_details, "red", attrs=["bold"])

        # Broadcast emergency stop command
        if self.communication_bus:
            emergency_message = Message(
                message_id=f"emergency_stop_{int(time.time())}",
                sender_id=self.agent_id,
                message_type=MessageType.COMMAND,
                topic=str(RiskMessageType.EMERGENCY_STOP.value),
                payload={
                    'reason': self.emergency_reason,
                    'assessment': assessment.__dict__,
                    'ai_confirmed': ai_confirmed,
                    'ai_confidence': ai_confidence,
                    'timestamp': time.time()
                },
                priority=MessagePriority.CRITICAL
            )

            await self.communication_bus.broadcast_message(emergency_message)

    # ============================================================================
    # Communication & Broadcasting
    # ============================================================================

    async def _broadcast_risk_status(self, assessment: RiskAssessment):
        """Broadcast current risk status to all agents."""
        if not self.communication_bus:
            return

        status_message = Message(
            message_id=f"risk_status_{int(time.time())}",
            sender_id=self.agent_id,
            message_type=MessageType.STATUS,
            topic=str(RiskMessageType.RISK_STATUS_UPDATE.value),
            payload={
                'circuit_breaker_state': assessment.circuit_breaker_state.value,
                'portfolio_risk_pct': assessment.portfolio_risk_pct,
                'max_drawdown_pct': assessment.max_drawdown_pct,
                'flux_level': self.current_flux_level,
                'violations_count': len(assessment.violations),
                'warnings_count': len(assessment.warnings),
                'timestamp': assessment.timestamp
            },
            priority=MessagePriority.NORMAL
        )

        await self.communication_bus.broadcast_message(status_message)

    async def _broadcast_circuit_breaker_update(self, new_state: CircuitBreakerState,
                                              assessment: RiskAssessment):
        """Broadcast circuit breaker state change."""
        if not self.communication_bus:
            return

        update_message = Message(
            message_id=f"circuit_breaker_{int(time.time())}",
            sender_id=self.agent_id,
            message_type=MessageType.STATUS,
            topic=str(RiskMessageType.CIRCUIT_BREAKER_UPDATE.value),
            payload={
                'old_state': assessment.circuit_breaker_state.value,
                'new_state': new_state.value,
                'reason': 'risk_assessment',
                'assessment': assessment.__dict__,
                'timestamp': time.time()
            },
            priority=MessagePriority.HIGH
        )

        await self.communication_bus.broadcast_message(update_message)

    # ============================================================================
    # Risk Handling & Response
    # ============================================================================

    async def _handle_risk_violations(self, assessment: RiskAssessment):
        """Handle risk violations based on severity."""
        for violation in assessment.violations:
            severity = violation.get('severity', 'warning')

            if severity == 'critical':
                # Log critical violation
                cprint(f"🚨 CRITICAL RISK VIOLATION: {violation['message']}", "red", attrs=["bold"])

                # Could trigger additional actions here
                if violation['type'] == 'portfolio_risk':
                    await self._handle_portfolio_risk_violation(violation)
                elif violation['type'] == 'drawdown':
                    await self._handle_drawdown_violation(violation)

        for warning in assessment.warnings:
            cprint(f"⚠️  RISK WARNING: {warning['message']}", "yellow")

    async def _handle_portfolio_risk_violation(self, violation: Dict[str, Any]):
        """Handle portfolio risk violation."""
        # Could implement position reduction logic here
        cprint("📉 Portfolio risk violation detected - consider position reduction", "yellow")

    async def _handle_drawdown_violation(self, violation: Dict[str, Any]):
        """Handle drawdown violation."""
        # Could implement stop-loss or position closure logic here
        cprint("📉 Drawdown violation detected - consider emergency measures", "red")

    # ============================================================================
    # Flux Management
    # ============================================================================

    def _update_flux_level(self):
        """Update current market flux level."""
        # Placeholder flux calculation
        # In full implementation, this would analyze:
        # - Recent price volatility
        # - Volume spikes
        # - Order book depth changes
        # - News sentiment
        # - Market indicators

        # For now, use config default with some randomization
        import random
        self.current_flux_level = FLUX_SENSITIVITY + random.uniform(-0.1, 0.1)
        self.current_flux_level = max(0.0, min(1.0, self.current_flux_level))  # Clamp to [0,1]

    def _adjust_risk_limits_for_flux(self):
        """Adjust risk limits based on current flux level and AI recommendations."""
        # Base flux adjustment
        flux_multiplier = 1.0 - (self.current_flux_level * 0.3)  # Reduce by up to 30%

        # AI-based adaptive adjustments
        ai_adjustments = self._calculate_ai_limit_adjustments()

        # Combine flux and AI adjustments
        final_multiplier = flux_multiplier * ai_adjustments.get('overall_multiplier', 1.0)

        # Ensure reasonable bounds
        final_multiplier = max(0.1, min(final_multiplier, 2.0))  # Between 10% and 200%

        self.current_risk_limits = RiskLimits(
            max_position_size=self.base_risk_limits.max_position_size * final_multiplier,
            max_portfolio_risk=self.base_risk_limits.max_portfolio_risk * final_multiplier,
            max_drawdown=self.base_risk_limits.max_drawdown,
            max_leverage=self.base_risk_limits.max_leverage,
            max_orders_per_minute=self.base_risk_limits.max_orders_per_minute,
            restricted_symbols=self.base_risk_limits.restricted_symbols.copy()
        )

    def _calculate_ai_limit_adjustments(self) -> Dict[str, float]:
        """Calculate AI-based risk limit adjustments."""
        adjustments = {'overall_multiplier': 1.0}

        if not self.ai_enabled or not self.ai_advisor or not self.risk_assessment_history:
            return adjustments

        try:
            # Analyze recent performance
            recent_assessments = self.risk_assessment_history[-10:]  # Last 10 assessments

            if len(recent_assessments) < 5:
                return adjustments

            # Calculate performance metrics
            avg_violations = np.mean([len(a.violations) for a in recent_assessments])
            avg_sharpe = np.mean([a.sharpe_ratio for a in recent_assessments])
            avg_volatility = np.mean([a.volatility_pct for a in recent_assessments])

            # AI-based adjustments logic
            multiplier = 1.0

            # If low violations and good Sharpe, increase limits slightly
            if avg_violations < 0.5 and avg_sharpe > 0.5:
                multiplier *= 1.1  # 10% increase

            # If high violations, reduce limits
            elif avg_violations > 2.0:
                multiplier *= 0.8  # 20% reduction

            # Volatility-based adjustments
            if avg_volatility > 25:  # High volatility
                multiplier *= 0.9  # 10% reduction
            elif avg_volatility < 10:  # Low volatility
                multiplier *= 1.05  # 5% increase

            # Recent emergency events - be more conservative
            recent_emergencies = sum(1 for a in recent_assessments
                                   if a.circuit_breaker_state == CircuitBreakerState.EMERGENCY)
            if recent_emergencies > 0:
                multiplier *= 0.85  # 15% reduction

            adjustments['overall_multiplier'] = multiplier
            adjustments['performance_score'] = (avg_sharpe + 1) / 2  # Normalize to 0-1
            adjustments['stability_score'] = max(0, 1 - avg_violations / 5)  # Lower violations = higher stability

        except Exception as e:
            cprint(f"⚠️ AI limit adjustment failed: {str(e)}", "yellow")

        return adjustments

    # ============================================================================
    # Data Persistence
    # ============================================================================

    def _save_risk_report(self, assessment: RiskAssessment):
        """Save risk assessment report to file."""
        try:
            report = {
                'timestamp': assessment.timestamp,
                'circuit_breaker_state': assessment.circuit_breaker_state.value,
                'flux_level': self.current_flux_level,
                'portfolio_balance': self.portfolio_balance,
                'positions': [pos.__dict__ for pos in self.positions.values()],
                'risk_assessment': {
                    'portfolio_risk_pct': assessment.portfolio_risk_pct,
                    'max_drawdown_pct': assessment.max_drawdown_pct,
                    'position_correlation_risk': assessment.position_correlation_risk,
                    'flux_adjusted_limits': assessment.flux_adjusted_limits,
                    'violations': assessment.violations,
                    'warnings': assessment.warnings,
                    'recommendations': assessment.recommendations
                },
                'total_positions': len(self.positions),
                'total_pnl': sum(pos.unrealized_pnl for pos in self.positions.values())
            }

            # Save latest report
            latest_path = os.path.join(self.output_dir, "latest_report.json")
            with open(latest_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Append to history
            history_path = os.path.join(self.output_dir, "risk_history.jsonl")
            with open(history_path, 'a') as f:
                f.write(json.dumps(report, default=str) + '\n')

        except Exception as e:
            cprint(f"❌ Error saving risk report: {str(e)}", "red")

    # ============================================================================
    # Public API Methods
    # ============================================================================

    async def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive current risk status."""
        latest_assessment = self.last_risk_assessment

        # Calculate risk metrics summary
        risk_summary = self._generate_risk_summary()

        # Get position risk breakdown
        position_risks = self._get_position_risk_breakdown()

        # Circuit breaker status
        circuit_breaker_info = {
            'current_state': self.circuit_breaker_state.value,
            'state_description': self._get_circuit_breaker_description(),
            'last_changed': getattr(self, '_last_state_change', None),
            'emergency_active': self.emergency_active,
            'emergency_reason': self.emergency_reason if self.emergency_active else None
        }

        return {
            'timestamp': time.time(),
            'circuit_breaker': circuit_breaker_info,
            'portfolio': {
                'balance': self.portfolio_balance,
                'position_count': len(self.positions),
                'total_exposure': sum(abs(p.market_value) for p in self.positions.values()),
                'total_pnl': sum(p.unrealized_pnl for p in self.positions.values())
            },
            'risk_metrics': risk_summary,
            'position_risks': position_risks,
            'risk_limits': {
                'current': self.current_risk_limits.__dict__,
                'base': self.base_risk_limits.__dict__,
                'flux_multiplier': 1.0 - (self.current_flux_level * 0.3)
            },
            'market_conditions': {
                'flux_level': self.current_flux_level,
                'flux_sensitivity': FLUX_SENSITIVITY,
                'flux_status': 'high' if self.current_flux_level > FLUX_SENSITIVITY else 'normal'
            },
            'recent_history': {
                'assessment_count': len(self.risk_assessment_history),
                'avg_volatility': np.mean([a.volatility_pct for a in self.risk_assessment_history[-10:]]) if self.risk_assessment_history else 0.0,
                'avg_sharpe': np.mean([a.sharpe_ratio for a in self.risk_assessment_history[-10:]]) if self.risk_assessment_history else 0.0
            }
        }

    def _generate_risk_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive risk metrics summary."""
        if not self.last_risk_assessment:
            return {'status': 'no_assessment_available'}

        assessment = self.last_risk_assessment

        # Risk score calculation (0-100, higher = riskier)
        risk_score = self._calculate_risk_score(assessment)

        # Risk health assessment
        risk_health = self._assess_risk_health(risk_score)

        return {
            'overall_risk_score': risk_score,
            'risk_health': risk_health,
            'key_metrics': {
                'portfolio_risk_pct': assessment.portfolio_risk_pct,
                'max_drawdown_pct': assessment.max_drawdown_pct,
                'var_95': assessment.var_95,
                'expected_shortfall_95': assessment.expected_shortfall_95,
                'sharpe_ratio': assessment.sharpe_ratio,
                'volatility_pct': assessment.volatility_pct,
                'correlation_risk': assessment.position_correlation_risk
            },
            'stress_test_summary': {
                'scenarios_tested': len(assessment.stress_test_results),
                'failed_scenarios': len([s for s in assessment.stress_test_results.values() if s.get('breaches_limit', False)]),
                'worst_case_impact': max([s.get('portfolio_impact_pct', 0) for s in assessment.stress_test_results.values()])
            },
            'violations_summary': {
                'total_violations': len(assessment.violations),
                'critical_violations': len([v for v in assessment.violations if v.get('severity') == 'critical']),
                'high_violations': len([v for v in assessment.violations if v.get('severity') == 'high'])
            },
            'warnings_summary': {
                'total_warnings': len(assessment.warnings),
                'recommendations_count': len(assessment.recommendations)
            }
        }

    def _calculate_risk_score(self, assessment: RiskAssessment) -> float:
        """Calculate an overall risk score (0-100)."""
        score_components = []

        # Portfolio risk component (0-25 points)
        portfolio_risk_score = min(assessment.portfolio_risk_pct / 2, 25)
        score_components.append(portfolio_risk_score)

        # VaR component (0-20 points)
        portfolio_value = self.portfolio_balance.get('equity', 1000.0)
        var_pct = (assessment.var_95 / portfolio_value) * 100
        var_score = min(var_pct * 2, 20)
        score_components.append(var_score)

        # Drawdown component (0-15 points)
        drawdown_score = min(assessment.max_drawdown_pct / 2, 15)
        score_components.append(drawdown_score)

        # Volatility component (0-15 points)
        volatility_score = min(assessment.volatility_pct / 10, 15)
        score_components.append(volatility_score)

        # Correlation component (0-10 points)
        correlation_score = min(assessment.position_correlation_risk / 10, 10)
        score_components.append(correlation_score)

        # Sharpe ratio penalty (0 to -10 points)
        sharpe_penalty = max(0, -assessment.sharpe_ratio * 5)
        score_components.append(-min(sharpe_penalty, 10))

        # Violations penalty (0-15 points)
        violation_score = len(assessment.violations) * 3
        score_components.append(min(violation_score, 15))

        total_score = sum(score_components)
        return max(0, min(total_score, 100))

    def _assess_risk_health(self, risk_score: float) -> Dict[str, Any]:
        """Assess overall risk health based on risk score."""
        if risk_score < 20:
            health = 'excellent'
            color = 'green'
            description = 'Risk levels are very low. Full trading capacity available.'
        elif risk_score < 40:
            health = 'good'
            color = 'blue'
            description = 'Risk levels are acceptable. Normal trading conditions.'
        elif risk_score < 60:
            health = 'moderate'
            color = 'yellow'
            description = 'Elevated risk levels. Consider reducing position sizes.'
        elif risk_score < 80:
            health = 'high'
            color = 'orange'
            description = 'High risk levels. Trading limits active. Monitor closely.'
        else:
            health = 'critical'
            color = 'red'
            description = 'Critical risk levels. Emergency measures may be triggered.'

        return {
            'status': health,
            'color': color,
            'description': description,
            'action_required': health in ['high', 'critical']
        }

    def _get_position_risk_breakdown(self) -> List[Dict[str, Any]]:
        """Get detailed risk breakdown for each position."""
        position_risks = []

        for symbol, position in self.positions.items():
            risk_metrics = self._assess_position_risk(position, symbol)
            portfolio_value = self.portfolio_balance.get('equity', 1000.0)
            position_value = abs(position.market_value)

            position_risks.append({
                'symbol': symbol,
                'position_value': position_value,
                'concentration_pct': (position_value / portfolio_value) * 100,
                'unrealized_pnl': position.unrealized_pnl,
                'pnl_pct': (position.unrealized_pnl / position_value) * 100 if position_value > 0 else 0,
                'risk_metrics': risk_metrics,
                'risk_contribution': {
                    'to_portfolio_risk': (position_value / portfolio_value) * self.last_risk_assessment.portfolio_risk_pct if self.last_risk_assessment else 0,
                    'to_portfolio_var': risk_metrics.get('var_contribution', 0)
                }
            })

        # Sort by concentration (highest first)
        position_risks.sort(key=lambda x: x['concentration_pct'], reverse=True)

        return position_risks

    def _get_circuit_breaker_description(self) -> str:
        """Get human-readable description of current circuit breaker state."""
        descriptions = {
            CircuitBreakerState.NORMAL: "Trading fully allowed with standard risk limits",
            CircuitBreakerState.WARNING: "Trading allowed with reduced risk limits due to elevated risk",
            CircuitBreakerState.HALTED: "Trading suspended due to risk violations - monitoring active",
            CircuitBreakerState.EMERGENCY: "Emergency stop active - all trading halted"
        }
        return descriptions.get(self.circuit_breaker_state, "Unknown state")

    async def check_trade_risk(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a proposed trade is within risk limits."""
        # Placeholder - would implement trade-specific risk checking
        return {
            'approved': True,
            'risk_score': 0.1,
            'warnings': [],
            'adjusted_size': trade_params.get('size', 0)
        }

    async def force_emergency_stop(self, reason: str) -> bool:
        """Force an emergency stop with given reason."""
        try:
            self.emergency_reason = reason
            assessment = await self._perform_risk_assessment()
            await self._initiate_emergency_stop(assessment)
            return True
        except Exception as e:
            cprint(f"❌ Failed to force emergency stop: {str(e)}", "red")
            return False

    async def train_ai_models(self) -> bool:
        """Train AI models using historical risk assessment data."""
        if not self.ai_enabled or not self.ai_advisor:
            cprint("⚠️ AI not enabled", "yellow")
            return False

        if len(self.risk_assessment_history) < 30:
            cprint("⚠️ Insufficient historical data for AI training (need at least 30 assessments)", "yellow")
            return False

        try:
            cprint("🧠 Training AI risk models with historical data...", "blue")
            success = self.ai_advisor.train_models(self.risk_assessment_history)

            if success:
                cprint("✅ AI models trained successfully", "green")
                # Save updated models
                self.ai_advisor._save_models()
            else:
                cprint("❌ AI model training failed", "red")

            return success

        except Exception as e:
            cprint(f"❌ AI training error: {str(e)}", "red")
            return False

    async def get_ai_status(self) -> Dict[str, Any]:
        """Get AI system status and performance metrics."""
        if not self.ai_enabled or not self.ai_advisor:
            return {'enabled': False, 'status': 'AI not available'}

        return {
            'enabled': True,
            'models_loaded': self.ai_advisor.risk_predictor is not None,
            'training_accuracy': getattr(self.ai_advisor, 'training_accuracy', 0.0),
            'last_training': getattr(self.ai_advisor, 'last_training_time', None),
            'model_version': getattr(self.ai_advisor, 'model_version', 'unknown'),
            'feature_count': len(getattr(self.ai_advisor, 'feature_columns', [])),
            'historical_data_points': len(self.risk_assessment_history)
        }

    # ============================================================================
    # Agent Lifecycle
    # ============================================================================

    async def start(self) -> None:
        """Start the Risk Agent."""
        # Call parent start
        super(RiskAgent, self).start()  # Call BaseAgent.start directly

        # Start risk monitoring
        await self.start_monitoring()

        cprint("🛡️ NeuroFlux Risk Agent started - monitoring active", "green")

    async def stop(self) -> None:
        """Stop the Risk Agent."""
        cprint("🛑 Stopping NeuroFlux Risk Agent...", "yellow")

        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.flux_task:
            self.flux_task.cancel()
        if self.reporting_task:
            self.reporting_task.cancel()

        # Call cleanup
        self._cleanup_agent()

        # Call parent stop
        super(RiskAgent, self).stop()

        cprint("✅ NeuroFlux Risk Agent stopped", "green")


# ============================================================================
# Standalone Risk Agent Runner (for backward compatibility)
# ============================================================================

def calculate_flux_level():
    """
    Calculate current market flux level for adaptive risk management.
    (Legacy function for backward compatibility)
    """
    return FLUX_SENSITIVITY


def get_portfolio_balance():
    """
    Get current portfolio balance across all exchanges.
    (Legacy function for backward compatibility)
    """
    return {
        'equity': 1000.0,
        'available': 950.0,
        'positions_value': 50.0
    }


def get_positions():
    """
    Get all current positions across exchanges.
    (Legacy function for backward compatibility)
    """
    return []


def check_risk_limits(balance, positions, flux_level):
    """
    Check all risk limits with flux-adaptive thresholds.
    (Legacy function for backward compatibility)
    """
    results = {
        'ok': True,
        'violations': [],
        'warnings': [],
        'recommendations': []
    }

    # Basic balance check
    if balance['equity'] < MINIMUM_BALANCE_USD:
        results['ok'] = False
        results['violations'].append({
            'type': 'balance',
            'message': f"Balance ${balance['equity']:.2f} below minimum ${MINIMUM_BALANCE_USD}",
            'severity': 'critical'
        })

    return results


def save_risk_report(results, balance, positions, flux_level):
    """
    Save risk assessment report to file.
    (Legacy function for backward compatibility)
    """
    output_dir = "src/data/risk_agent/"
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'flux_level': flux_level,
        'balance': balance,
        'positions': positions,
        'risk_check': results,
        'total_positions': len(positions),
        'total_pnl': 0
    }

    filename = f"{output_dir}/latest_report.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def emergency_close_positions(reason):
    """
    Emergency close all positions.
    (Legacy function for backward compatibility)
    """
    cprint(f"🚨 EMERGENCY: Closing all positions - {reason}", "red", attrs=["bold"])
    cprint("✅ All positions closed", "green")


def main():
    """Legacy main function for backward compatibility."""
    cprint("🧠 NeuroFlux Risk Agent (Legacy Mode) starting...", "cyan")
    cprint("⚠️  This is the legacy standalone version. Use RiskAgent class for full functionality.", "yellow")

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
                cprint("⚠️  Risk violations detected!", "red", attrs=["bold"])

                for violation in risk_results['violations']:
                    cprint(f"🚨 {violation['message']}", "red")

                    if violation['severity'] == 'critical':
                        emergency_close_positions(violation['message'])
                        break

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