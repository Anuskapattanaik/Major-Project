"""UI Prediction Service for CarbonIQ Vehicle Emission Predictor.

This module provides a reusable prediction service that loads the trained model
and makes predictions using the exact same preprocessing pipeline used during training.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


class PredictionService:
    """Service for loading model and making predictions."""
    
    def __init__(self):
        self._model = None
        self._metadata = None
        self._feature_names = None
        self._numerical_features = None
        self._categorical_features = None
        self._target_column = None
        self._model_name = None
        self._train_config = None
        self._test_metrics = None
        self._categorical_categories = None
        self._numerical_ranges = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and metadata."""
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            self._model = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
            
            if METADATA_PATH.exists():
                import json
                with open(METADATA_PATH, 'r') as f:
                    self._metadata = json.load(f)
                
                self._feature_names = self._metadata.get('feature_names', [])
                self._numerical_features = self._metadata.get('numerical_features', [])
                self._categorical_features = self._metadata.get('categorical_features', [])
                self._target_column = self._metadata.get('target_column', 'CO2 Emissions')
                self._model_name = self._metadata.get('model_name', 'Unknown')
                self._train_config = self._metadata.get('train_config', {})
                self._test_metrics = self._metadata.get('test_metrics', {})
                logger.info(f"Metadata loaded: {self._model_name}")
            else:
                logger.warning("Metadata file not found, using defaults")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def model(self):
        """Get the loaded model."""
        return self._model
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return self._metadata or {}
    
    @property
    def feature_names(self) -> List[str]:
        """Get expected feature names in order."""
        return self._feature_names or []
    
    @property
    def numerical_features(self) -> List[str]:
        """Get numerical feature names."""
        return self._numerical_features or []
    
    @property
    def categorical_features(self) -> List[str]:
        """Get categorical feature names."""
        return self._categorical_features or []
    
    @property
    def target_column(self) -> str:
        """Get target column name."""
        return self._target_column or 'CO2 Emissions'
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name or 'Unknown'
    
    @property
    def train_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self._train_config or {}
    
    @property
    def test_metrics(self) -> Dict[str, float]:
        """Get test metrics."""
        return self._test_metrics or {}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded successfully."""
        return self._model is not None
    
    def get_categorical_options(self) -> Dict[str, List[str]]:
        """Get valid categories for categorical features from the dataset."""
        if self._categorical_categories is not None:
            return self._categorical_categories
        
        # Default categories from the dataset
        self._categorical_categories = {
            'Vehicle Type': ['Car', 'Truck', 'Bus', 'Motorcycle'],
            'Fuel Type': ['Petrol', 'Diesel', 'Electric', 'Hybrid'],
            'Road Type': ['Highway', 'City', 'Rural'],
            'Traffic Conditions': ['Free flow', 'Moderate', 'Heavy']
        }
        return self._categorical_categories
    
    def get_numerical_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get observed ranges for numerical features from the dataset."""
        if self._numerical_ranges is not None:
            return self._numerical_ranges
        
        # Default ranges from the dataset
        self._numerical_ranges = {
            'Engine Size': {'min': 0.8, 'max': 6.0, 'mean': 3.37, 'std': 1.50},
            'Age of Vehicle': {'min': 0, 'max': 29, 'mean': 14.49, 'std': 8.58},
            'Mileage': {'min': 56, 'max': 299863, 'mean': 148995.71, 'std': 85000},
            'Speed': {'min': 0, 'max': 120, 'mean': 59.45, 'std': 30},
            'Acceleration': {'min': 0, 'max': 5.0, 'mean': 2.49, 'std': 1.2},
            'Temperature': {'min': -10, 'max': 40, 'mean': 14.94, 'std': 15},
            'Humidity': {'min': 0, 'max': 100, 'mean': 50.14, 'std': 25},
            'Wind Speed': {'min': 0, 'max': 20, 'mean': 10.03, 'std': 5.7},
            'Air Pressure': {'min': 950, 'max': 1050, 'mean': 1000.20, 'std': 28}
        }
        return self._numerical_ranges
    
    def validate_inputs(self, input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate input data before prediction.
        
        Args:
            input_data: Dictionary of input features
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required features
        for feature in self.feature_names:
            if feature not in input_data:
                errors.append(f"Missing required feature: {feature}")
        
        if errors:
            return False, errors
        
        # Validate numerical features
        ranges = self.get_numerical_ranges()
        for feature in self.numerical_features:
            if feature in input_data:
                value = input_data[feature]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    errors.append(f"{feature}: value is required")
                    continue
                
                try:
                    val = float(value)
                    if not np.isfinite(val):
                        errors.append(f"{feature}: value must be a finite number")
                    elif feature in ranges:
                        r = ranges[feature]
                        if val < r['min'] or val > r['max']:
                            errors.append(
                                f"{feature}: value {val} is outside observed range "
                                f"[{r['min']}, {r['max']}] (dataset units)"
                            )
                except (ValueError, TypeError):
                    errors.append(f"{feature}: must be a valid number")
        
        # Validate categorical features
        cat_options = self.get_categorical_options()
        for feature in self.categorical_features:
            if feature in input_data:
                value = input_data[feature]
                if value is None or (isinstance(value, str) and value.strip() == ''):
                    errors.append(f"{feature}: value is required")
                elif feature in cat_options and value not in cat_options[feature]:
                    errors.append(
                        f"{feature}: '{value}' is not a valid category. "
                        f"Valid options: {', '.join(cat_options[feature])}"
                    )
        
        return len(errors) == 0, errors
    
    def prepare_input_dataframe(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data as DataFrame with correct feature order.
        
        Args:
            input_data: Dictionary of input features
            
        Returns:
            DataFrame with features in the correct order
        """
        # Create DataFrame with single row
        df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = np.nan
        
        # Reorder columns to match training order
        df = df[self.feature_names]
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Make a prediction using the trained model.
        
        Args:
            input_data: Dictionary of input features
            
        Returns:
            Tuple of (prediction, prediction_info)
            
        Raises:
            ValueError: If inputs are invalid or prediction fails
        """
        if not self.is_loaded():
            raise ValueError("Model not loaded. Please check model artifacts.")
        
        # Validate inputs
        is_valid, errors = self.validate_inputs(input_data)
        if not is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(errors)}")
        
        # Prepare DataFrame
        X = self.prepare_input_dataframe(input_data)
        
        # Make prediction
        try:
            prediction = self._model.predict(X)[0]
            
            if not np.isfinite(prediction):
                raise ValueError("Model returned non-finite prediction")
            
            prediction_info = {
                'prediction': float(prediction),
                'model_name': self.model_name,
                'target_column': self.target_column,
                'input_features': input_data,
                'feature_names': self.feature_names
            }
            
            return float(prediction), prediction_info
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")


# Streamlit cached instance
@st.cache_resource
def get_prediction_service() -> PredictionService:
    """Get cached prediction service instance."""
    return PredictionService()


def get_service() -> PredictionService:
    """Get prediction service (non-cached for testing)."""
    return PredictionService()