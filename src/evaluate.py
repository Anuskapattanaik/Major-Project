"""Model evaluation and reporting module."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from .config import config, get_output_path, get_report_path, get_figure_path
from .visualization import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residuals_vs_predicted,
    plot_feature_importance,
    plot_model_comparison
)

logger = logging.getLogger(__name__)


def evaluate_model_detailed(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        pipeline: Fitted pipeline
        X_test: Test features
        y_test: Test target
        feature_names: Optional feature names for importance
        
    Returns:
        Dictionary with detailed evaluation results
    """
    y_pred = pipeline.predict(X_test)
    
    # Basic metrics
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred))
    }
    
    # MAPE (if applicable)
    if np.all(y_test > 0):
        metrics["MAPE"] = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
    else:
        metrics["MAPE"] = None
    
    # Residuals
    residuals = y_test.values - y_pred
    
    # Additional statistics
    metrics["residual_mean"] = float(np.mean(residuals))
    metrics["residual_std"] = float(np.std(residuals))
    metrics["residual_median"] = float(np.median(residuals))
    
    # Prediction statistics
    metrics["pred_mean"] = float(np.mean(y_pred))
    metrics["pred_std"] = float(np.std(y_pred))
    metrics["pred_min"] = float(np.min(y_pred))
    metrics["pred_max"] = float(np.max(y_pred))
    
    # Actual statistics
    metrics["actual_mean"] = float(np.mean(y_test))
    metrics["actual_std"] = float(np.std(y_test))
    metrics["actual_min"] = float(np.min(y_test))
    metrics["actual_max"] = float(np.max(y_test))
    
    results = {
        "metrics": metrics,
        "predictions": y_pred.tolist(),
        "residuals": residuals.tolist(),
        "n_samples": len(y_test)
    }
    
    # Feature importance (if supported)
    if feature_names is not None:
        importance = get_feature_importance(pipeline, X_test, y_test, feature_names)
        if importance is not None:
            results["feature_importance"] = importance
    
    return results


def get_feature_importance(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> Optional[Dict[str, float]]:
    """
    Calculate permutation importance on test set.
    
    Args:
        pipeline: Fitted pipeline
        X_test: Test features
        y_test: Test target
        feature_names: Feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        Dictionary of feature importance scores or None if not supported
    """
    try:
        # Check if model supports permutation importance
        model = pipeline.named_steps.get("model")
        if model is None:
            return None
        
        # Use permutation importance
        result = permutation_importance(
            pipeline, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Map to feature names
        importance_dict = dict(zip(feature_names, result.importances_mean))
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        logger.info(f"Calculated permutation importance for {len(importance_dict)} features")
        return importance_dict
        
    except Exception as e:
        logger.warning(f"Could not calculate permutation importance: {e}")
        return None


def create_evaluation_report(
    evaluation_results: Dict[str, Any],
    model_name: str,
    model_metadata: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Path:
    """
    Create detailed evaluation report in Markdown.
    
    Args:
        evaluation_results: Results from evaluate_model_detailed
        model_name: Name of the model
        model_metadata: Model metadata
        output_path: Optional custom path
        
    Returns:
        Path to saved report
    """
    if output_path is None:
        output_path = get_report_path("model_evaluation_report.md")
    
    metrics = evaluation_results["metrics"]
    
    lines = [
        "# Model Evaluation Report",
        "",
        f"## Model: {model_name}",
        "",
        "### Test Set Performance Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
        f"| MAE (Mean Absolute Error) | {metrics['MAE']:.4f} | Average absolute difference between predicted and actual values |",
        f"| RMSE (Root Mean Squared Error) | {metrics['RMSE']:.4f} | Square root of average squared errors; penalizes large errors |",
        f"| R² (Coefficient of Determination) | {metrics['R2']:.4f} | Proportion of variance explained; 1 = perfect, 0 = baseline, <0 = worse than baseline |",
    ]
    
    if metrics.get("MAPE") is not None:
        lines.append(f"| MAPE (Mean Absolute Percentage Error) | {metrics['MAPE']:.2f}% | Average percentage error (only valid if all actuals > 0) |")
    else:
        lines.append("| MAPE | N/A | Not calculated (target contains non-positive values) |")
    
    lines.extend([
        "",
        "### Residual Analysis",
        "",
        f"- Mean residual: {metrics['residual_mean']:.4f} (should be near 0 for unbiased predictions)",
        f"- Std residual: {metrics['residual_std']:.4f}",
        f"- Median residual: {metrics['residual_median']:.4f}",
        "",
        "### Prediction vs Actual Summary",
        "",
        f"- Predictions: mean={metrics['pred_mean']:.2f}, std={metrics['pred_std']:.2f}, range=[{metrics['pred_min']:.2f}, {metrics['pred_max']:.2f}]",
        f"- Actuals: mean={metrics['actual_mean']:.2f}, std={metrics['actual_std']:.2f}, range=[{metrics['actual_min']:.2f}, {metrics['actual_max']:.2f}]",
        "",
        "### Model Configuration",
        "",
        f"- Target variable: {model_metadata.get('target_column', 'Unknown')}",
        f"- Features used: {model_metadata.get('n_features_in', 'Unknown')}",
        f"- Numerical features: {model_metadata.get('n_numerical', 'Unknown')}",
        f"- Categorical features: {model_metadata.get('n_categorical', 'Unknown')}",
        f"- Training configuration: {json.dumps(model_metadata.get('train_config', {}), indent=2)}",
        "",
        "### Important Notes",
        "",
        "- R² is **not** a percentage accuracy score; it can be negative",
        "- Lower MAE and RMSE indicate better performance",
        "- MAPE is only meaningful when all target values are positive",
        "- Feature importance indicates predictive association, **not** causal impact",
        "- Test set performance is on held-out data; real-world performance may vary",
        "",
        "### Limitations",
        "",
        "- Model trained on cross-sectional data; does not learn temporal trends",
        "- Future-year scenarios require explicit assumptions (see scenario analysis)",
        "- Predictions are for individual vehicles, not fleet aggregates",
        "- Extrapolation beyond training data range is unreliable"
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Evaluation report saved to {output_path}")
    return output_path


def save_feature_importance(
    importance: Dict[str, float],
    output_path: Optional[Path] = None
) -> Path:
    """
    Save feature importance to CSV.
    
    Args:
        importance: Feature importance dictionary
        output_path: Optional custom path
        
    Returns:
        Path to saved CSV
    """
    if output_path is None:
        output_path = get_output_path("feature_importance.csv")
    
    df = pd.DataFrame([
        {"feature": k, "importance": v}
        for k, v in importance.items()
    ])
    df.to_csv(output_path, index=False)
    logger.info(f"Feature importance saved to {output_path}")
    return output_path


def run_full_evaluation(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    model_metadata: Dict[str, Any],
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Run complete evaluation with plots and reports.
    
    Args:
        pipeline: Fitted pipeline
        X_test: Test features
        y_test: Test target
        model_name: Model name
        model_metadata: Model metadata
        feature_names: Feature names
        
    Returns:
        Complete evaluation results
    """
    # Detailed evaluation
    results = evaluate_model_detailed(pipeline, X_test, y_test, feature_names)
    
    # Save feature importance if available
    if "feature_importance" in results:
        save_feature_importance(results["feature_importance"])
    
    # Create evaluation report
    create_evaluation_report(results, model_name, model_metadata)
    
    # Generate plots
    y_pred = np.array(results["predictions"])
    residuals = np.array(results["residuals"])
    
    # Actual vs Predicted
    plot_actual_vs_predicted(
        y_test.values, y_pred,
        save_path=get_figure_path("actual_vs_predicted.png"),
        title=f"Actual vs Predicted CO₂ Emissions - {model_name}"
    )
    
    # Residuals distribution
    plot_residuals(
        residuals,
        save_path=get_figure_path("residuals_distribution.png"),
        title=f"Residual Distribution - {model_name}"
    )
    
    # Residuals vs Predicted
    plot_residuals_vs_predicted(
        y_pred, residuals,
        save_path=get_figure_path("residuals_vs_predicted.png"),
        title=f"Residuals vs Predicted - {model_name}"
    )
    
    # Feature importance
    if "feature_importance" in results:
        plot_feature_importance(
            results["feature_importance"],
            top_n=20,
            save_path=get_figure_path("feature_importance.png"),
            title=f"Feature Importance (Permutation) - {model_name}"
        )
    
    return results