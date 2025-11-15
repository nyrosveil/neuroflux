"""
🧠 NeuroFlux ML Prediction Agent
Machine learning-based price and volume prediction agent for NeuroFlux trading system.

Built with love by Nyros Veil 🚀

Features:
- Time series price prediction using multiple ML models
- Volume prediction and analysis
- Confidence scoring and risk assessment
- Integration with NeuroFlux ML API
"""

import asyncio
import time
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from termcolor import cprint

class MLPredictionAgent:
    """
    ML Prediction Agent for NeuroFlux trading system.

    This agent handles machine learning-based predictions for:
    - Price forecasting using time series models
    - Volume prediction and analysis
    - Confidence interval calculation
    - Risk assessment based on prediction uncertainty
    """

    def __init__(self, agent_id: str, communication_bus=None):
        self.agent_id = agent_id
        self.communication_bus = communication_bus
        self.start_time = time.time()

        # Agent capabilities
        self.capabilities = ['ML_PREDICTION', 'TIME_SERIES_ANALYSIS']

        # ML API configuration
        self.ml_api_base = "http://localhost:8000/api/ml"  # Dashboard API endpoint
        self.prediction_timeout = 30  # seconds

        # Model configurations
        self.supported_models = ['arima', 'exponential_smoothing', 'simple_moving_average']
        self.default_confidence_threshold = 0.7

        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_confidence': 0.0,
            'model_performance': {}
        }

        # Message handlers
        self.message_handlers = {}

    def register_handler(self, message_type: str, handler):
        """Register a message handler."""
        self.message_handlers[message_type] = handler

    def initialize(self) -> bool:
        """Initialize the ML prediction agent."""
        try:
            cprint(f"🧠 Initializing ML Prediction Agent {self.agent_id}", "cyan", attrs=['bold'])

            # Register message handlers
            self.register_handler("execute_task", self._handle_execute_task)

            # Initialize model performance tracking
            for model in self.supported_models:
                self.prediction_stats['model_performance'][model] = {
                    'predictions': 0,
                    'accuracy': 0.0,
                    'avg_confidence': 0.0
                }

            cprint("✅ ML Prediction Agent initialized", "green")
            return True
        except Exception as e:
            cprint(f"❌ ML Prediction Agent initialization failed: {e}", "red")
            return False

    async def _handle_execute_task(self, message: Dict[str, Any]) -> None:
        """Handle task execution requests."""
        task_data = message.get('payload', {}).get('task', {})
        task_type = task_data.get('task_type', '')

        if task_type == 'ml_prediction':
            await self._execute_ml_prediction_task(task_data)
        else:
            cprint(f"⚠️ Unknown task type: {task_type}", "yellow")

    async def _execute_ml_prediction_task(self, task: Dict[str, Any]) -> None:
        """Execute an ML prediction task."""
        task_id = task.get('task_id', 'unknown')
        payload = task.get('payload', {})

        cprint(f"🔮 Executing ML prediction task {task_id}", "blue")

        try:
            # Determine prediction type
            if 'price' in task.get('name', '').lower():
                result = await self._predict_prices(payload)
            elif 'volume' in task.get('name', '').lower():
                result = await self._predict_volumes(payload)
            else:
                result = await self._predict_prices(payload)  # Default to price prediction

            # Send completion message
            await self._send_task_result(task_id, result, status='completed')

        except Exception as e:
            cprint(f"❌ ML prediction task {task_id} failed: {e}", "red")
            await self._send_task_result(task_id, {'error': str(e)}, status='failed')

    def _get_historical_data(self, token: str) -> List[Dict[str, Any]]:
        """Get historical price data for ML model training/prediction."""
        # In a real implementation, this would fetch from a database or API
        # For now, generate synthetic historical data
        import random
        from datetime import datetime, timedelta

        data_points = []
        base_price = 45000 if token == 'BTC' else 3000 if token == 'ETH' else 150
        current_time = datetime.now()

        for i in range(100):  # 100 data points for training
            timestamp = current_time - timedelta(hours=100-i)
            price_change = random.uniform(-0.05, 0.05)  # ±5% daily change
            price = base_price * (1 + price_change * i * 0.01)  # Slight trend

            # Add some noise
            noise = random.uniform(-0.02, 0.02)
            final_price = price * (1 + noise)

            data_points.append({
                'timestamp': timestamp.isoformat(),
                'open': final_price * random.uniform(0.98, 1.02),
                'high': final_price * random.uniform(1.00, 1.05),
                'low': final_price * random.uniform(0.95, 1.00),
                'close': final_price,
                'volume': random.uniform(1000000, 10000000)
            })

        return data_points

    async def _predict_prices(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate price predictions using ML models."""
        tokens = payload.get('tokens', ['BTC'])
        prediction_horizon = payload.get('prediction_horizon', 60)  # minutes
        models = payload.get('models', self.supported_models)
        confidence_threshold = payload.get('confidence_threshold', self.default_confidence_threshold)

        predictions = {}

        for token in tokens:
            token_predictions = {}

            for model in models:
                if model not in self.supported_models:
                    continue

                try:
                    # Call ML API for prediction
                    prediction_result = await self._call_ml_api(
                        endpoint=f"/api/ml/predict/{model.lower()}",
                        payload={
                            'data': self._get_historical_data(token),  # Historical price data
                            'target_column': 'close',
                            'config': {
                                'prediction_horizon': prediction_horizon,
                                'confidence_threshold': confidence_threshold
                            }
                        }
                    )

                    if prediction_result:
                        token_predictions[model] = prediction_result
                        self._update_model_stats(model, prediction_result)

                except Exception as e:
                    cprint(f"⚠️ Failed to get {model} prediction for {token}: {e}", "yellow")
                    continue

            if token_predictions:
                # Calculate ensemble prediction
                ensemble_prediction = self._calculate_ensemble_prediction(token_predictions, confidence_threshold)
                predictions[token] = {
                    'individual_models': token_predictions,
                    'ensemble_prediction': ensemble_prediction,
                    'timestamp': datetime.now().isoformat()
                }

        return {
            'prediction_type': 'price',
            'predictions': predictions,
            'model_performance': self.prediction_stats['model_performance'],
            'timestamp': datetime.now().isoformat()
        }

    async def _predict_volumes(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate volume predictions using ML models."""
        tokens = payload.get('tokens', ['BTC'])
        prediction_horizon = payload.get('prediction_horizon', 30)  # minutes
        models = payload.get('models', ['arima', 'exponential_smoothing'])
        volume_types = payload.get('volume_types', ['total_volume'])

        predictions = {}

        for token in tokens:
            token_predictions = {}

            for volume_type in volume_types:
                volume_predictions = {}

                for model in models:
                    if model not in self.supported_models:
                        continue

                    try:
                        # Call ML API for volume prediction
                        prediction_result = await self._call_ml_api(
                            endpoint=f"/predict/{model}",
                            payload={
                                'token': token,
                                'data_type': volume_type,
                                'horizon': prediction_horizon,
                                'include_confidence': True
                            }
                        )

                        if prediction_result:
                            volume_predictions[model] = prediction_result
                            self._update_model_stats(model, prediction_result)

                    except Exception as e:
                        cprint(f"⚠️ Failed to get {model} volume prediction for {token}: {e}", "yellow")
                        continue

                if volume_predictions:
                    token_predictions[volume_type] = volume_predictions

            if token_predictions:
                predictions[token] = {
                    'volume_predictions': token_predictions,
                    'timestamp': datetime.now().isoformat()
                }

        return {
            'prediction_type': 'volume',
            'predictions': predictions,
            'metadata': {
                'models_used': models,
                'volume_types': volume_types,
                'prediction_horizon_minutes': prediction_horizon
            }
        }

    async def _call_ml_api(self, endpoint: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call the ML API endpoint."""
        try:
            url = f"{self.ml_api_base}{endpoint}"
            response = requests.post(url, json=payload, timeout=self.prediction_timeout)

            if response.status_code == 200:
                return response.json()
            else:
                cprint(f"⚠️ ML API call failed: {response.status_code} - {response.text}", "yellow")
                return None

        except requests.exceptions.RequestException as e:
            cprint(f"⚠️ ML API request error: {e}", "yellow")
            return None

    def _calculate_ensemble_prediction(self, model_predictions: Dict[str, Any],
                                     confidence_threshold: float) -> Dict[str, Any]:
        """Calculate ensemble prediction from multiple models."""
        # Filter predictions by confidence threshold
        valid_predictions = {}
        weights = {}

        for model, prediction in model_predictions.items():
            confidence = prediction.get('confidence', 0.0)
            if confidence >= confidence_threshold:
                valid_predictions[model] = prediction
                weights[model] = confidence

        if not valid_predictions:
            # Fallback to all predictions if none meet threshold
            valid_predictions = model_predictions
            weights = {model: pred.get('confidence', 0.5) for model, pred in model_predictions.items()}

        # Calculate weighted ensemble
        total_weight = sum(weights.values())

        if total_weight == 0:
            # Equal weighting if all weights are 0
            ensemble_value = sum(pred.get('prediction', 0) for pred in valid_predictions.values()) / len(valid_predictions)
            ensemble_confidence = sum(pred.get('confidence', 0) for pred in valid_predictions.values()) / len(valid_predictions)
        else:
            # Weighted average
            ensemble_value = sum(pred.get('prediction', 0) * weights[model] for model, pred in valid_predictions.items()) / total_weight
            ensemble_confidence = sum(pred.get('confidence', 0) * weights[model] for model, pred in valid_predictions.items()) / total_weight

        # Calculate confidence intervals
        predictions_list = [pred.get('prediction', 0) for pred in valid_predictions.values()]
        if len(predictions_list) > 1:
            variance = sum((p - ensemble_value) ** 2 for p in predictions_list) / len(predictions_list)
            std_dev = variance ** 0.5
            confidence_interval = {
                'lower': ensemble_value - 1.96 * std_dev,  # 95% confidence interval
                'upper': ensemble_value + 1.96 * std_dev
            }
        else:
            confidence_interval = {'lower': ensemble_value * 0.95, 'upper': ensemble_value * 1.05}

        return {
            'prediction': ensemble_value,
            'confidence': ensemble_confidence,
            'confidence_interval': confidence_interval,
            'models_contributed': list(valid_predictions.keys()),
            'ensemble_method': 'weighted_average'
        }

    def _update_model_stats(self, model: str, prediction_result: Dict[str, Any]) -> None:
        """Update model performance statistics."""
        self.prediction_stats['total_predictions'] += 1

        model_stats = self.prediction_stats['model_performance'][model]
        model_stats['predictions'] += 1

        confidence = prediction_result.get('confidence', 0.0)
        model_stats['avg_confidence'] = (
            (model_stats['avg_confidence'] * (model_stats['predictions'] - 1)) + confidence
        ) / model_stats['predictions']

    async def _send_task_result(self, task_id: str, result: Dict[str, Any], status: str) -> None:
        """Send task completion result."""
        message = {
            'message_id': f"result_{task_id}_{int(time.time())}",
            'sender_id': self.agent_id,
            'recipient_id': 'orchestrator',
            'message_type': 'task_result',
            'topic': 'task_result',
            'payload': {
                'task_id': task_id,
                'status': status,
                'result': result,
                'timestamp': datetime.now().isoformat()
            },
            'timestamp': time.time()
        }

        if self.communication_bus and hasattr(self.communication_bus, 'send_message'):
            await self.communication_bus.send_message(message)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'agent_id': self.agent_id,
            'capabilities': self.capabilities,
            'prediction_stats': self.prediction_stats,
            'supported_models': self.supported_models,
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }

    def _initialize_agent(self) -> bool:
        """Initialize the ML prediction agent."""
        try:
            cprint(f"🧠 Initializing ML Prediction Agent {self.agent_id}", "cyan", attrs=['bold'])

            # Register message handlers
            self.register_handler("execute_task", self._handle_execute_task)

            # Initialize model performance tracking
            for model in self.supported_models:
                self.prediction_stats['model_performance'][model] = {
                    'predictions': 0,
                    'accuracy': 0.0,
                    'avg_confidence': 0.0
                }

            cprint("✅ ML Prediction Agent initialized", "green")
            return True
        except Exception as e:
            cprint(f"❌ ML Prediction Agent initialization failed: {e}", "red")
            return False

    def _execute_agent_cycle(self):
        """Execute one cycle of the ML prediction agent."""
        # This agent primarily responds to tasks, so no continuous cycle needed
        # Minimal implementation for the abstract method requirement
        pass

    def _cleanup_agent(self):
        """Clean up the ML prediction agent resources."""
        # No specific cleanup needed for this agent
        pass