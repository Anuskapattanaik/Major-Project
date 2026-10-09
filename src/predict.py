"""Prediction module for making predictions on new data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import json
from .config import config
from .preprocessing import validate_input_features
from .train import load_model, load_model_metadata

from .config import config, get_model_path, get_output_path
from .preprocessing import validate_input_features
from .train import load_model, load_model_metadata

logger = logging.getLogger(__name__)


def load_pipeline_and_metadata(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model pipeline and metadata.
    
    Args:
        model_path: Optional custom model path
        metadata_path: Optional custom metadata path
        
    Returns:
        Tuple of (pipeline, metadata)
    """
    pipeline = load_model(model_path)
    metadata = load_model_metadata(metadata_path)
    return pipeline, metadata


def prepare_input_data(
    input_data: Union[Dict, List[Dict], str, Path, pd.DataFrame],
    expected_features: List[str]
) -> pd.DataFrame:
    """
    Prepare input data for prediction.
    
    Args:
        input_data: Input as dict, list of dicts, CSV path, or DataFrame
        expected_features: List of expected feature names
        
    Returns:
        Prepared DataFrame
    """
    # Handle different input types
    if isinstance(input_data, (str, Path)):
        # CSV file
        df = pd.read_csv(input_data)
    elif isinstance(input_data, dict):
        # Single record as dict
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        # List of dicts
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Select only expected features in correct order
    # Add missing columns with NaN (will be imputed by pipeline)
    for feat in expected_features:
        if feat not in df.columns:
            df[feat] = np.nan
            logger.warning(f"Feature '{feat}' not in input, will be imputed")
    
    # Reorder columns
    # Only select features that are in expected_features and present in df
    available_features = [f for f in expected_features if f in df.columns]
    df = df[available_features]
    
    return df


def predict(
    input_data: Union[Dict, List[Dict], str, Path, pd.DataFrame],
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Make predictions on new data.
    
    Args:
        input_data: Input data (dict, list of dicts, CSV path, or DataFrame)
        model_path: Optional custom model path
        metadata_path: Optional custom metadata path
        output_path: Optional path to save predictions CSV
        
    Returns:
        DataFrame with predictions
    """
    # Load model and metadata
    pipeline, metadata = load_pipeline_and_metadata(model_path, metadata_path)
    
    # Get expected features
    expected_features = metadata.get("feature_names", [])
    target_name = metadata.get("target_column", config.data.target_column)
    
    if not expected_features:
        raise ValueError("No feature names found in model metadata")
    
    # Prepare input
    X = prepare_input_data(input_data, expected_features)
    
    # Predict
    logger.info(f"Making predictions for {len(X)} records")
    predictions = pipeline.predict(X)
    
    # Create output DataFrame
    output_df = X.copy()
    output_df[f"predicted_{target_name}"] = predictions
    
    # Save if requested
    if output_path is None:
        output_path = get_output_path("predictions.csv")
    
    output_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Log summary
    logger.info(f"Prediction stats: mean={np.mean(predictions):.2f}, "
                f"std={np.std(predictions):.2f}, "
                f"min={np.min(predictions):.2f}, max={np.max(predictions):.2f}")
    
    return output_df


def predict_single(
    vehicle_data: Dict[str, Any],
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None
) -> float:
    """
    Predict emissions for a single vehicle.
    
    Args:
        vehicle_data: Dictionary with vehicle features
        model_path: Optional custom model path
        metadata_path: Optional custom metadata path
        
    Returns:
        Predicted emission value
    """
    pipeline, metadata = load_pipeline_and_metadata(model_path, metadata_path)
    expected_features = metadata.get("feature_names", [])
    
    # Prepare input
    X = prepare_input_data(vehicle_data, expected_features)
    
    # Predict
    prediction = pipeline.predict(X)[0]
    
    logger.info(f"Single prediction: {prediction:.4f}")
    return float(prediction)


def create_sample_input(output_path: Optional[Path] = None) -> Path:
    """
    Create a sample input file with plausible values.
    
    Args:
        output_path: Optional custom path
        
    Returns:
        Path to created sample file
    """
    if output_path is None:
        output_path = get_output_path("sample_input.csv")
    
    # Sample data based on typical ranges in the dataset
    sample_data = {
        "Vehicle Type": ["Car", "Truck", "Motorcycle", "Bus", "Car"],
        "Fuel Type": ["Petrol", "Diesel", "Electric", "Hybrid", "Petrol"],
        "Engine Size": [2.0, 5.5, 1.5, 4.0, 1.8],
        "Age of Vehicle": [5, 10, 3, 8, 12],
        "Mileage": [50000, 150000, 20000, 80000, 120000],
        "Speed": [60, 55, 70, 45, 65],
        "Acceleration": [2.5, 1.8, 3.0, 2.0, 2.2],
        "Road Type": ["Highway", "City", "Rural", "Highway", "City"],
        "Traffic Conditions": ["Moderate", "Heavy", "Free flow", "Moderate", "Free flow"],
        "Temperature": [20, 15, 25, 10, 22],
        "Humidity": [60, 70, 50, 80, 55],
        "Wind Speed": [5, 10, 3, 8, 6],
        "Air Pressure": [1013, 1015, 1010, 1012, 1014]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Write clean CSV - disclaimer documented in function docstring
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample input created at {output_path}")
    return output_path


def validate_prediction_input(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Validate input data for prediction and provide detailed report.
    
    Args:
        df: Input DataFrame
        numerical_features: Expected numerical features
        categorical_features: Expected categorical features
        
    Returns:
        Validation report
    """
    return validate_input_features(df, numerical_features, categorical_features)