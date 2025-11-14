"""
Simple time series predictor implementations for NeuroFlux.

This module provides concrete implementations of time series prediction models
including statistical models and basic neural network approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

from .base_predictor import TimeSeriesPredictor, PredictionResult


class ARIMAPredictor(TimeSeriesPredictor):
    """ARIMA-based time series predictor."""

    def __init__(self, model_name: str = "arima_predictor", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.order = self.config.get('order', (1, 1, 1))  # (p, d, q)
        self.arima_model = None

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for ARIMA model."""
        if self.target_name not in data.columns:
            raise ValueError(f"Target column '{self.target_name}' not found in data")

        # ARIMA works with 1D time series
        target_data = data[self.target_name].values

        # For ARIMA, we don't need sequences, just the target series
        # But we'll return in the expected format
        X = target_data[:-1].reshape(-1, 1)  # All but last value
        y = target_data[1:]  # Next values

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train ARIMA model."""
        # For ARIMA, we need the full time series
        # X and y are derived from the same series, so reconstruct
        full_series = np.concatenate([X.flatten(), [y[-1]]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.arima_model = ARIMA(full_series, order=self.order)
                fitted_model = self.arima_model.fit()

                # Calculate basic metrics
                aic = fitted_model.aic if hasattr(fitted_model, 'aic') else 0
                bic = fitted_model.bic if hasattr(fitted_model, 'bic') else 0

                return {
                    'aic': aic,
                    'bic': bic,
                    'converged': True
                }
            except Exception as e:
                self.logger.warning(f"ARIMA training failed: {e}")
                return {
                    'aic': float('inf'),
                    'bic': float('inf'),
                    'converged': False,
                    'error': str(e)
                }

    def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """Make predictions using trained ARIMA model."""
        if self.arima_model is None:
            raise ValueError("Model not trained")

        steps = kwargs.get('steps', 1)

        try:
            fitted_model = self.arima_model.fit()
            forecast = fitted_model.forecast(steps=steps)

            # Calculate confidence intervals
            conf_int = fitted_model.get_forecast(steps=steps).conf_int()

            return PredictionResult(
                predictions=forecast.values,
                confidence_scores=np.abs(conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).values / 2,  # Half-width as confidence score
                metadata={'model_type': 'ARIMA', 'steps': steps}
            )
        except Exception as e:
            self.logger.error(f"ARIMA prediction failed: {e}")
            return PredictionResult(
                predictions=np.array([np.nan] * steps),
                metadata={'error': str(e)}
            )

    def _save_model_data(self) -> Dict[str, Any]:
        """Save ARIMA model data."""
        return {
            'order': self.order,
            'arima_model': self.arima_model.to_dict() if self.arima_model else None
        }

    def _load_model_data(self, model_data: Dict[str, Any]) -> None:
        """Load ARIMA model data."""
        self.order = model_data.get('order', (1, 1, 1))
        # Note: ARIMA model serialization is complex, would need custom implementation


class ExponentialSmoothingPredictor(TimeSeriesPredictor):
    """Exponential Smoothing predictor for time series."""

    def __init__(self, model_name: str = "exp_smoothing_predictor", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.trend = self.config.get('trend', 'add')
        self.seasonal = self.config.get('seasonal', None)
        self.seasonal_periods = self.config.get('seasonal_periods', None)
        self.exp_model = None

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for exponential smoothing."""
        if self.target_name not in data.columns:
            raise ValueError(f"Target column '{self.target_name}' not found in data")

        target_data = data[self.target_name].values

        # Similar to ARIMA, work with full series
        X = target_data[:-1].reshape(-1, 1)
        y = target_data[1:]

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train exponential smoothing model."""
        full_series = np.concatenate([X.flatten(), [y[-1]]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.exp_model = ExponentialSmoothing(
                    full_series,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods
                )
                fitted_model = self.exp_model.fit()

                # Calculate basic metrics
                aic = fitted_model.aic if hasattr(fitted_model, 'aic') else 0
                bic = fitted_model.bic if hasattr(fitted_model, 'bic') else 0

                return {
                    'aic': aic,
                    'bic': bic,
                    'converged': True
                }
            except Exception as e:
                self.logger.warning(f"Exponential smoothing training failed: {e}")
                return {
                    'aic': float('inf'),
                    'bic': float('inf'),
                    'converged': False,
                    'error': str(e)
                }

    def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """Make predictions using exponential smoothing."""
        if self.exp_model is None:
            raise ValueError("Model not trained")

        steps = kwargs.get('steps', 1)

        try:
            fitted_model = self.exp_model.fit()
            forecast = fitted_model.forecast(steps=steps)

            return PredictionResult(
                predictions=forecast.values,
                metadata={'model_type': 'ExponentialSmoothing', 'steps': steps}
            )
        except Exception as e:
            self.logger.error(f"Exponential smoothing prediction failed: {e}")
            return PredictionResult(
                predictions=np.array([np.nan] * steps),
                metadata={'error': str(e)}
            )


class SimpleMovingAveragePredictor(TimeSeriesPredictor):
    """Simple moving average predictor for baseline comparisons."""

    def __init__(self, model_name: str = "sma_predictor", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.window_size = self.config.get('window_size', 10)

    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for moving average."""
        if self.target_name not in data.columns:
            raise ValueError(f"Target column '{self.target_name}' not found in data")

        target_data = data[self.target_name].values

        # Create sequences for moving average
        X, y = [], []
        for i in range(len(target_data) - self.window_size):
            X.append(target_data[i:i + self.window_size])
            y.append(target_data[i + self.window_size])

        return np.array(X), np.array(y)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train moving average model (no actual training needed)."""
        # Moving average doesn't need training, just calculate baseline metrics
        predictions = np.mean(X, axis=1)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))

        return {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }

    def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """Make predictions using moving average."""
        steps = kwargs.get('steps', 1)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Use the last window for prediction
        last_window = X[-1]

        predictions = []
        for _ in range(steps):
            pred = np.mean(last_window)
            predictions.append(pred)
            # Update window for next prediction
            last_window = np.roll(last_window, -1)
            last_window[-1] = pred

        return PredictionResult(
            predictions=np.array(predictions),
            metadata={'model_type': 'SimpleMovingAverage', 'window_size': self.window_size, 'steps': steps}
        )