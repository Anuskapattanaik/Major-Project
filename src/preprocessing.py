"""Preprocessing pipeline module using scikit-learn."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import joblib
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted

from .config import config, get_model_path

logger = logging.getLogger(__name__)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Select columns from DataFrame and return as numpy array."""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.columns].values


class FeatureNameExtractor(BaseEstimator, TransformerMixin):
    """Extract feature names after preprocessing for interpretation."""
    
    def __init__(self, numerical_features: List[str], categorical_features: List[str]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_names_out_ = None
    
    def fit(self, X: pd.DataFrame, y=None):
        # Build expected output feature names
        cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
        cat_encoder.fit(X[self.categorical_features])
        cat_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        self.feature_names_out_ = self.numerical_features + cat_names
        return self
    
    def transform(self, X):
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def create_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    scale_numerical: bool = True
) -> ColumnTransformer:
    """
    Create a ColumnTransformer for preprocessing.
    
    Args:
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        scale_numerical: Whether to scale numerical features
        
    Returns:
        Fitted ColumnTransformer
    """
    # Numerical pipeline
    if scale_numerical:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    return preprocessor


def create_full_pipeline(
    model: Any,
    numerical_features: List[str],
    categorical_features: List[str],
    scale_numerical: bool = True
) -> Pipeline:
    """
    Create a full pipeline with preprocessing and model.
    
    Args:
        model: Scikit-learn estimator
        numerical_features: List of numerical column names
        categorical_features: List of categorical column names
        scale_numerical: Whether to scale numerical features
        
    Returns:
        Complete Pipeline
    """
    preprocessor = create_preprocessing_pipeline(
        numerical_features, categorical_features, scale_numerical
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return pipeline


def get_model_specific_scaling(model_name: str) -> bool:
    """
    Determine if a model needs feature scaling.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if scaling is recommended
    """
    # Tree-based models don't need scaling
    tree_models = [
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "HistGradientBoostingRegressor",
        "GradientBoostingRegressor",
        "DecisionTreeRegressor",
        "XGBRegressor"
    ]
    
    return model_name not in tree_models


def fit_preprocessing_pipeline(
    X_train: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    model_name: str = "default"
) -> ColumnTransformer:
    """
    Fit preprocessing pipeline on training data only.
    
    Args:
        X_train: Training features
        numerical_features: Numerical column names
        categorical_features: Categorical column names
        model_name: Model name for logging
        
    Returns:
        Fitted ColumnTransformer
    """
    scale = get_model_specific_scaling(model_name)
    preprocessor = create_preprocessing_pipeline(
        numerical_features, categorical_features, scale
    )
    
    logger.info(f"Fitting preprocessor for {model_name} (scale={scale})")
    logger.info(f"  Numerical features: {len(numerical_features)}")
    logger.info(f"  Categorical features: {len(categorical_features)}")
    
    preprocessor.fit(X_train)
    
    # Log feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"  Output features: {len(feature_names)}")
    except Exception:
        pass
    
    return preprocessor


def transform_features(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Transform features using fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        X: Features to transform
        
    Returns:
        Transformed feature array
    """
    check_is_fitted(preprocessor)
    return preprocessor.transform(X)


def save_preprocessor(
    preprocessor: ColumnTransformer,
    filepath: Optional[Path] = None
) -> Path:
    """
    Save fitted preprocessor to disk.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        filepath: Optional custom path
        
    Returns:
        Path where preprocessor was saved
    """
    if filepath is None:
        filepath = get_model_path("preprocessor.joblib")
    
    joblib.dump(preprocessor, filepath)
    logger.info(f"Preprocessor saved to {filepath}")
    return filepath


def load_preprocessor(filepath: Optional[Path] = None) -> ColumnTransformer:
    """
    Load fitted preprocessor from disk.
    
    Args:
        filepath: Optional custom path
        
    Returns:
        Loaded ColumnTransformer
    """
    if filepath is None:
        filepath = get_model_path("preprocessor.joblib")
    
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessor not found at {filepath}")
    
    preprocessor = joblib.load(filepath)
    logger.info(f"Preprocessor loaded from {filepath}")
    return preprocessor


def validate_input_features(
    X: pd.DataFrame,
    expected_numerical: List[str],
    expected_categorical: List[str]
) -> Dict[str, Any]:
    """
    Validate input features for prediction.
    
    Args:
        X: Input features
        expected_numerical: Expected numerical columns
        expected_categorical: Expected categorical columns
        
    Returns:
        Validation report
    """
    report = {
        "valid": True,
        "missing_numerical": [],
        "missing_categorical": [],
        "extra_columns": [],
        "warnings": []
    }
    
    all_expected = set(expected_numerical + expected_categorical)
    all_input = set(X.columns)
    
    # Check missing
    report["missing_numerical"] = [c for c in expected_numerical if c not in all_input]
    report["missing_categorical"] = [c for c in expected_categorical if c not in all_input]
    
    # Check extra
    report["extra_columns"] = list(all_input - all_expected)
    
    if report["missing_numerical"] or report["missing_categorical"]:
        report["valid"] = False
        missing = report["missing_numerical"] + report["missing_categorical"]
        report["warnings"].append(f"Missing required features: {missing}")
    
    if report["extra_columns"]:
        report["warnings"].append(f"Extra columns will be ignored: {report['extra_columns']}")
    
    # Check for unseen categories in categorical columns
    for col in expected_categorical:
        if col in X.columns:
            # This would need the fitted encoder to check properly
            pass
    
    return report