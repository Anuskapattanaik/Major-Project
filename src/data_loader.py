"""Data loading and quality checking module."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

from .config import config, get_data_path, get_output_path, get_report_path

logger = logging.getLogger(__name__)


def load_dataset(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the vehicle emission dataset.
    
    Args:
        data_path: Optional explicit path to CSV. If None, uses configured paths.
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If dataset cannot be found
    """
    if data_path is None:
        data_path = get_data_path()
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please ensure the CSV file exists."
        )
    
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive dataset inspection.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with inspection results
    """
    results = {}
    
    # Basic info
    results["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    results["dtypes"] = df.dtypes.astype(str).to_dict()
    results["columns"] = df.columns.tolist()
    
    # Sample records
    results["head"] = df.head(5).to_dict(orient="records")
    results["tail"] = df.tail(3).to_dict(orient="records")
    
    # Missing values
    missing = df.isnull().sum()
    results["missing_values"] = missing[missing > 0].to_dict()
    results["missing_percentage"] = (missing / len(df) * 100).round(2).to_dict()
    results["total_missing"] = int(missing.sum())
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    results["duplicate_rows"] = int(duplicates)
    
    # Infinite values (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = int(inf_count)
    results["infinite_values"] = inf_counts
    
    # Numerical summary
    results["numerical_summary"] = df.describe(include=[np.number]).round(4).to_dict()
    
    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_summary = {}
    for col in cat_cols:
        cat_summary[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(10).to_dict(),
            "missing": int(df[col].isnull().sum())
        }
    results["categorical_summary"] = cat_summary
    
    # Target column check
    target = config.data.target_column
    if target in df.columns:
        results["target_exists"] = True
        target_vals = df[target].dropna()
        results["target_stats"] = {
            "count": int(len(target_vals)),
            "mean": float(target_vals.mean()),
            "std": float(target_vals.std()),
            "min": float(target_vals.min()),
            "max": float(target_vals.max()),
            "median": float(target_vals.median()),
            "missing": int(df[target].isnull().sum())
        }
        # Check for valid numeric
        results["target_is_numeric"] = pd.api.types.is_numeric_dtype(df[target])
    else:
        results["target_exists"] = False
        results["target_stats"] = {}
        logger.warning(f"Target column '{target}' not found in dataset")
    
    return results


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform detailed data quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality check results
    """
    results = {}
    
    # 1. Missing value analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    results["missing_analysis"] = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percentage": missing_pct.values
    }).to_dict(orient="records")
    
    # 2. Duplicate analysis
    dup_mask = df.duplicated(keep=False)
    results["duplicate_count"] = int(dup_mask.sum())
    if results["duplicate_count"] > 0:
        results["duplicate_rows_sample"] = df[dup_mask].head(5).to_dict(orient="records")
    
    # 3. Outlier detection (IQR method for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            outlier_info[col] = {
                "count": int(len(outliers)),
                "percentage": round(len(outliers) / len(df) * 100, 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "min_outlier": float(outliers.min()) if len(outliers) > 0 else None,
                "max_outlier": float(outliers.max()) if len(outliers) > 0 else None
            }
    results["outlier_analysis"] = outlier_info
    
    # 4. Categorical consistency
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_consistency = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        cat_consistency[col] = {
            "unique_count": len(unique_vals),
            "values": sorted([str(v) for v in unique_vals])[:50],  # Limit display
            "has_whitespace_issues": any(str(v) != str(v).strip() for v in unique_vals),
            "has_case_issues": len(set(str(v).lower() for v in unique_vals)) < len(unique_vals)
        }
    results["categorical_consistency"] = cat_consistency
    
    # 5. Column name consistency
    results["column_names"] = {
        "original": df.columns.tolist(),
        "stripped": [c.strip() for c in df.columns],
        "has_leading_trailing_spaces": any(c != c.strip() for c in df.columns),
        "has_duplicates_after_strip": len(set(c.strip() for c in df.columns)) < len(df.columns)
    }
    
    return results


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the dataset based on quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    report = {}
    original_shape = df.shape
    df_clean = df.copy()
    
    # 1. Strip column names
    df_clean.columns = df_clean.columns.str.strip()
    report["columns_stripped"] = True
    
    # 2. Handle missing values in target
    target = config.data.target_column
    if target in df_clean.columns:
        missing_target = df_clean[target].isnull().sum()
        if missing_target > 0:
            df_clean = df_clean.dropna(subset=[target])
            report["dropped_missing_target"] = int(missing_target)
    
    # 3. Drop columns with excessive missing values
    missing_pct = df_clean.isnull().sum() / len(df_clean)
    high_missing = missing_pct[missing_pct > config.data.missing_threshold].index.tolist()
    if high_missing:
        df_clean = df_clean.drop(columns=high_missing)
        report["dropped_high_missing_columns"] = high_missing
    
    # 4. Remove exact duplicates
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        report["removed_duplicates"] = int(dup_count)
    
    # 5. Handle infinite values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inf_replaced = {}
    for col in numeric_cols:
        inf_mask = np.isinf(df_clean[col])
        if inf_mask.any():
            # Replace inf with NaN, then impute
            df_clean.loc[inf_mask, col] = np.nan
            inf_replaced[col] = int(inf_mask.sum())
    report["infinite_values_replaced"] = inf_replaced
    
    # 6. Strip whitespace from string columns
    str_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    report["string_columns_stripped"] = str_cols.tolist()
    
    # Summary
    report["original_shape"] = original_shape
    report["cleaned_shape"] = df_clean.shape
    report["rows_removed"] = original_shape[0] - df_clean.shape[0]
    report["columns_removed"] = original_shape[1] - df_clean.shape[1]
    
    logger.info(f"Cleaning complete: {original_shape} -> {df_clean.shape}")
    return df_clean, report


def save_data_quality_report(inspection: Dict, quality: Dict, cleaning: Dict) -> Path:
    """
    Save comprehensive data quality report.
    
    Args:
        inspection: Results from inspect_dataset
        quality: Results from check_data_quality
        cleaning: Results from clean_dataset
        
    Returns:
        Path to saved report
    """
    # Create combined report
    report_data = {
        "dataset_inspection": inspection,
        "quality_checks": quality,
        "cleaning_report": cleaning
    }
    
    # Save as JSON for detailed access
    json_path = get_report_path("data_quality_report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # Save key tables as CSV
    # Missing values table
    if "missing_analysis" in quality:
        missing_df = pd.DataFrame(quality["missing_analysis"])
        missing_path = get_output_path("data_quality_report.csv")
        missing_df.to_csv(missing_path, index=False)
    
    logger.info(f"Data quality report saved to {json_path}")
    return json_path


def prepare_features_targets(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features and target for modeling.
    
    Args:
        df: Cleaned DataFrame
        target_col: Target column name (uses config default if None)
        exclude_cols: Columns to exclude from features (uses config default if None)
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    if target_col is None:
        target_col = config.data.target_column
    if exclude_cols is None:
        exclude_cols = config.data.exclude_columns
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Separate features and target
    y = df[target_col].copy()
    
    # Exclude target and other leakage columns from features
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    
    logger.info(f"Prepared features: {X.shape[1]} columns, target: {target_col}")
    return X, y, feature_cols


def get_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical feature columns.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple of (numerical_columns, categorical_columns)
    """
    numerical = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    logger.info(f"Feature types: {len(numerical)} numerical, {len(categorical)} categorical")
    return numerical, categorical