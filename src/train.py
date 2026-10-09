"""Model training module."""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from .config import config, get_model_path, get_output_path
from .preprocessing import (
    create_full_pipeline,
    get_model_specific_scaling,
    fit_preprocessing_pipeline,
    save_preprocessor
)

logger = logging.getLogger(__name__)


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.
    
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {}
    
    for name, params in config.model.models.items():
        if name == "DummyRegressor":
            models[name] = DummyRegressor(**params)
        elif name == "LinearRegression":
            models[name] = LinearRegression(**params)
        elif name == "Ridge":
            models[name] = Ridge(**params)
        elif name == "RandomForestRegressor":
            models[name] = RandomForestRegressor(**params)
        elif name == "ExtraTreesRegressor":
            models[name] = ExtraTreesRegressor(**params)
        elif name == "HistGradientBoostingRegressor":
            models[name] = HistGradientBoostingRegressor(**params)
        elif name == "GradientBoostingRegressor":
            models[name] = GradientBoostingRegressor(**params)
        else:
            logger.warning(f"Unknown model: {name}, skipping")
    
    return models


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # MAE
    metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))
    
    # RMSE
    metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # R²
    metrics["R2"] = float(r2_score(y_true, y_pred))
    
    # MAPE (only if all y_true > 0)
    if np.all(y_true > 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics["MAPE"] = float(mape)
    else:
        metrics["MAPE"] = np.nan
        logger.warning("MAPE not calculated: target contains non-positive values")
    
    return metrics


def cross_validate_model(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error"
) -> Dict[str, float]:
    """
    Perform cross-validation on training data.
    
    Args:
        pipeline: Scikit-learn pipeline
        X: Training features
        y: Training target
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with CV results
    """
    try:
        scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        # Convert to positive RMSE
        rmse_scores = -scores
        return {
            "cv_rmse_mean": float(rmse_scores.mean()),
            "cv_rmse_std": float(rmse_scores.std()),
            "cv_scores": rmse_scores.tolist()
        }
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return {
            "cv_rmse_mean": np.nan,
            "cv_rmse_std": np.nan,
            "cv_scores": []
        }


def train_single_model(
    model_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numerical_features: List[str],
    categorical_features: List[str],
    use_cv: bool = True
) -> Dict[str, Any]:
    """
    Train a single model and evaluate.
    
    Args:
        model_name: Name of the model
        model: Model instance
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        numerical_features: Numerical column names
        categorical_features: Categorical column names
        use_cv: Whether to run cross-validation
        
    Returns:
        Dictionary with training results
    """
    start_time = time.time()
    
    # Determine scaling
    scale = get_model_specific_scaling(model_name)
    
    # Create pipeline
    pipeline = create_full_pipeline(
        model, numerical_features, categorical_features, scale
    )
    
    # Train
    logger.info(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    metrics = evaluate_model(y_test.values, y_pred)
    
    # Cross-validation
    cv_results = {}
    if use_cv:
        cv_results = cross_validate_model(
            pipeline, X_train, y_train, 
            cv=config.data.cv_folds
        )
    
    # Training metrics (on training set)
    y_train_pred = pipeline.predict(X_train)
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    
    result = {
        "model_name": model_name,
        "train_time_seconds": round(train_time, 2),
        "test_metrics": metrics,
        "train_metrics": train_metrics,
        "cv_results": cv_results,
        "pipeline": pipeline,
        "scaling_used": scale,
        "model_params": model.get_params()
    }
    
    logger.info(f"{model_name} - Test RMSE: {metrics['RMSE']:.4f}, "
                f"Test R2: {metrics['R2']:.4f}, Time: {train_time:.2f}s")
    
    return result


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    numerical_features: List[str],
    categorical_features: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, Dict], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train all configured models and compare.
    
    Args:
        X: All features
        y: All targets
        numerical_features: Numerical column names
        categorical_features: Categorical column names
        test_size: Test split ratio
        random_state: Random seed
        
    Returns:
        Tuple of (results_dict, X_train, y_train, X_test, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Get models
    models = get_models()
    
    # Train each model
    results = {}
    for name, model in models.items():
        try:
            result = train_single_model(
                name, model,
                X_train, y_train,
                X_test, y_test,
                numerical_features, categorical_features,
                use_cv=True
            )
            results[name] = result
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results[name] = {
                "model_name": name,
                "error": str(e),
                "test_metrics": {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
            }
    
    return results, X_train, y_train, X_test, y_test


def select_best_model(
    results: Dict[str, Dict],
    metric: str = "RMSE",
    lower_is_better: bool = True
) -> Tuple[str, Dict]:
    """
    Select best model based on test metric.
    
    Args:
        results: Training results dictionary
        metric: Metric to use for selection
        lower_is_better: Whether lower metric is better
        
    Returns:
        Tuple of (best_model_name, best_model_result)
    """
    valid_results = {
        name: res for name, res in results.items()
        if "test_metrics" in res and metric in res["test_metrics"]
        and not np.isnan(res["test_metrics"][metric])
    }
    
    if not valid_results:
        raise ValueError("No valid models to select from")
    
    if lower_is_better:
        best_name = min(valid_results.keys(), 
                       key=lambda k: valid_results[k]["test_metrics"][metric])
    else:
        best_name = max(valid_results.keys(), 
                       key=lambda k: valid_results[k]["test_metrics"][metric])
    
    logger.info(f"Best model: {best_name} ({metric}={valid_results[best_name]['test_metrics'][metric]:.4f})")
    return best_name, valid_results[best_name]


def save_model_comparison(results: Dict[str, Dict], output_path: Optional[Path] = None) -> Path:
    """
    Save model comparison table to CSV.
    
    Args:
        results: Training results
        output_path: Optional custom path
        
    Returns:
        Path to saved CSV
    """
    if output_path is None:
        output_path = get_output_path("model_comparison.csv")
    
    rows = []
    for name, res in results.items():
        if "test_metrics" in res:
            row = {"model": name}
            row.update(res["test_metrics"])
            if "cv_results" in res and "cv_rmse_mean" in res["cv_results"]:
                row["cv_rmse_mean"] = res["cv_results"]["cv_rmse_mean"]
                row["cv_rmse_std"] = res["cv_results"]["cv_rmse_std"]
            row["train_time_seconds"] = res.get("train_time_seconds", np.nan)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by selection metric
    metric = config.model.selection_metric
    if metric in df.columns:
        df = df.sort_values(metric, ascending=config.model.selection_metric_lower_is_better)
    
    df.to_csv(output_path, index=False)
    logger.info(f"Model comparison saved to {output_path}")
    
    return output_path


def save_final_model(
    pipeline: Any,
    model_name: str,
    numerical_features: List[str],
    categorical_features: List[str],
    feature_names: List[str],
    target_name: str,
    train_config: Dict[str, Any],
    metrics: Dict[str, float],
    output_path: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Save final model pipeline and metadata.
    
    Args:
        pipeline: Fitted pipeline
        model_name: Name of the model
        numerical_features: Numerical feature names
        categorical_features: Categorical feature names
        feature_names: All input feature names
        target_name: Target column name
        train_config: Training configuration
        metrics: Test metrics
        output_path: Optional custom path for model
        
    Returns:
        Tuple of (model_path, metadata_path)
    """
    if output_path is None:
        output_path = get_model_path("final_model.joblib")
    
    # Save model
    joblib.dump(pipeline, output_path)
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "target_column": target_name,
        "feature_names": feature_names,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "train_config": train_config,
        "test_metrics": metrics,
        "n_features_in": len(feature_names),
        "n_numerical": len(numerical_features),
        "n_categorical": len(categorical_features),
        "saved_at": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = get_model_path("model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Final model saved to {output_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return output_path, metadata_path


def load_model(model_path: Optional[Path] = None) -> Any:
    """
    Load saved model pipeline.
    
    Args:
        model_path: Optional custom path
        
    Returns:
        Loaded pipeline
    """
    if model_path is None:
        model_path = get_model_path("final_model.joblib")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return pipeline


def load_model_metadata(metadata_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load model metadata.
    
    Args:
        metadata_path: Optional custom path
        
    Returns:
        Metadata dictionary
    """
    if metadata_path is None:
        metadata_path = get_model_path("model_metadata.json")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return metadata