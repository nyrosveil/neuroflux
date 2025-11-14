"""
Base predictor classes for NeuroFlux ML predictions.

This module provides the foundation for all prediction models used in NeuroFlux,
including time series forecasting, market prediction, and risk assessment.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os

logger = logging.getLogger(__name__)


class PredictionResult:
    """Container for prediction results with confidence scores."""

    def __init__(self,
                 predictions: np.ndarray,
                 confidence_scores: Optional[np.ndarray] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'predictions': self.predictions.tolist() if isinstance(self.predictions, np.ndarray) else self.predictions,
            'confidence_scores': self.confidence_scores.tolist() if self.confidence_scores is not None and isinstance(self.confidence_scores, np.ndarray) else self.confidence_scores,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class BasePredictor(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.is_trained = False
        self.model = None
        self.feature_names: List[str] = []
        self.target_name: Optional[str] = None
        self.training_metrics: Dict[str, float] = {}

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{model_name}")

    def _save_model_data(self) -> Dict[str, Any]:
        """Save model-specific data. Override in subclasses."""
        return {}

    def _load_model_data(self, model_data: Dict[str, Any]) -> None:
        """Load model-specific data. Override in subclasses."""
        pass

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess input data for model training/prediction.

        Args:
            data: Input dataframe with features and target

        Returns:
            Tuple of (X, y) where X is features array and y is target array
        """
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the prediction model.

        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """Make predictions using the trained model.

        Args:
            X: Feature matrix for prediction
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult object with predictions and metadata
        """
        pass

    def fit(self, data: pd.DataFrame, target_column: str, **kwargs) -> Dict[str, float]:
        """Fit the model on training data.

        Args:
            data: Training dataframe
            target_column: Name of target column
            **kwargs: Additional training parameters

        Returns:
            Training metrics
        """
        self.target_name = target_column
        X, y = self.preprocess_data(data)
        metrics = self.train(X, y, **kwargs)
        self.is_trained = True
        self.training_metrics = metrics
        return metrics

    def predict_from_data(self, data: pd.DataFrame, **kwargs) -> PredictionResult:
        """Make predictions from dataframe input.

        Args:
            data: Input dataframe
            **kwargs: Additional prediction parameters

        Returns:
            PredictionResult object
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained yet")

        X, _ = self.preprocess_data(data)
        return self.predict(X, **kwargs)

    def save_model(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model_name': self.model_name,
            'config': self.config,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Save model-specific data
        if hasattr(self, '_save_model_data'):
            model_data['model_data'] = self._save_model_data()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)

        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.target_name = model_data['target_name']
        self.training_metrics = model_data['training_metrics']

        # Load model-specific data
        if 'model_data' in model_data and hasattr(self, '_load_model_data'):
            self._load_model_data(model_data['model_data'])

        self.logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_metrics': self.training_metrics,
            'config': self.config
        }

    def validate_data(self, data: pd.DataFrame) -> List[str]:
        """Validate input data for prediction.

        Args:
            data: Input dataframe to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if data.empty:
            errors.append("Input data is empty")

        if self.feature_names and not all(col in data.columns for col in self.feature_names):
            missing = [col for col in self.feature_names if col not in data.columns]
            errors.append(f"Missing required features: {missing}")

        return errors


class TimeSeriesPredictor(BasePredictor):
    """Base class for time series prediction models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.sequence_length = self.config.get('sequence_length', 60)  # Default 60 time steps
        self.prediction_horizon = self.config.get('prediction_horizon', 1)  # Default 1 step ahead

    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for time series prediction.

        Args:
            data: Time series data array
            sequence_length: Length of each sequence

        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])

        return np.array(X), np.array(y)

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess time series data."""
        # Assume data has a time index and target column
        if self.target_name not in data.columns:
            raise ValueError(f"Target column '{self.target_name}' not found in data")

        # Extract target series
        target_data = data[self.target_name].values

        # Create sequences
        X, y = self.create_sequences(target_data, self.sequence_length)

        return X, y

    def predict_future(self, data: pd.DataFrame, steps: int = 1) -> PredictionResult:
        """Predict future values beyond the available data.

        Args:
            data: Historical data
            steps: Number of future steps to predict

        Returns:
            PredictionResult with future predictions
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained yet")

        # Start with the last sequence from historical data
        last_sequence = data[self.target_name].values[-self.sequence_length:]

        predictions = []
        confidence_scores = []

        for _ in range(steps):
            # Reshape for prediction
            X = last_sequence.reshape(1, -1, 1) if len(last_sequence.shape) == 1 else last_sequence.reshape(1, *last_sequence.shape)

            # Make prediction
            result = self.predict(X)
            pred = result.predictions[0]

            predictions.append(pred)
            if result.confidence_scores is not None:
                confidence_scores.append(result.confidence_scores[0])

            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred

        return PredictionResult(
            predictions=np.array(predictions),
            confidence_scores=np.array(confidence_scores) if confidence_scores else None,
            metadata={'prediction_type': 'future', 'steps': steps}
        )