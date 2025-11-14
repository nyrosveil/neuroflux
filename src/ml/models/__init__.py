# ML Models Module
# Contains base predictor classes and time series models for NeuroFlux

from .base_predictor import BasePredictor, TimeSeriesPredictor, PredictionResult

# Import time series models
try:
    from .time_series_models import (
        ARIMAPredictor,
        ExponentialSmoothingPredictor,
        SimpleMovingAveragePredictor
    )
    TIME_SERIES_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Time series models not available: {e}")
    TIME_SERIES_MODELS_AVAILABLE = False
    ARIMAPredictor = None
    ExponentialSmoothingPredictor = None
    SimpleMovingAveragePredictor = None

__all__ = [
    'BasePredictor',
    'TimeSeriesPredictor',
    'PredictionResult',
    'ARIMAPredictor',
    'ExponentialSmoothingPredictor',
    'SimpleMovingAveragePredictor',
    'TIME_SERIES_MODELS_AVAILABLE'
]