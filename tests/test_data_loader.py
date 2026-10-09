"""Tests for data_loader module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_dataset, inspect_dataset, check_data_quality, 
    clean_dataset, prepare_features_targets, get_feature_types
)
from src.config import config


class TestDataLoader:
    """Tests for data loading functions."""
    
    def test_load_dataset_exists(self):
        """Test loading existing dataset."""
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0
    
    def test_load_dataset_missing(self):
        """Test loading missing dataset raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "missing.csv"
            with pytest.raises(FileNotFoundError):
                load_dataset(missing_path)
    
    def test_inspect_dataset(self):
        """Test dataset inspection."""
        df = load_dataset()
        inspection = inspect_dataset(df)
        
        assert "shape" in inspection
        assert "columns" in inspection
        assert "dtypes" in inspection
        assert "missing_values" in inspection
        assert "target_exists" in inspection
        assert inspection["target_exists"] is True
        assert "target_stats" in inspection
    
    def test_check_data_quality(self):
        """Test data quality checks."""
        df = load_dataset()
        quality = check_data_quality(df)
        
        assert "missing_analysis" in quality
        assert "duplicate_count" in quality
        assert "outlier_analysis" in quality
        assert "categorical_consistency" in quality
        assert "column_names" in quality
    
    def test_clean_dataset(self):
        """Test dataset cleaning."""
        df = load_dataset()
        df_clean, cleaning = clean_dataset(df)
        
        assert isinstance(df_clean, pd.DataFrame)
        assert "original_shape" in cleaning
        assert "cleaned_shape" in cleaning
        assert cleaning["cleaned_shape"][0] <= cleaning["original_shape"][0]
    
    def test_prepare_features_targets(self):
        """Test feature/target preparation."""
        df = load_dataset()
        df_clean, _ = clean_dataset(df)
        
        X, y, features = prepare_features_targets(df_clean)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(features) == X.shape[1]
        assert config.data.target_column not in features
        # Check excluded columns not in features
        for col in config.data.exclude_columns:
            if col in df_clean.columns:
                assert col not in features
    
    def test_get_feature_types(self):
        """Test feature type detection."""
        df = load_dataset()
        df_clean, _ = clean_dataset(df)
        X, _, _ = prepare_features_targets(df_clean)
        
        numerical, categorical = get_feature_types(X)
        
        assert isinstance(numerical, list)
        assert isinstance(categorical, list)
        assert len(numerical) + len(categorical) == X.shape[1]


class TestDataValidation:
    """Tests for data validation with synthetic data."""
    
    def test_missing_value_handling(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            "num1": [1, 2, np.nan, 4],
            "num2": [1.0, 2.0, 3.0, 4.0],
            "cat1": ["A", "B", "A", np.nan],
            "target": [10, 20, 30, 40]
        })
        
        X, y, features = prepare_features_targets(df, target_col="target")
        
        assert "num1" in features
        assert "cat1" in features
        assert X["num1"].isnull().sum() == 1
        assert X["cat1"].isnull().sum() == 1
    
    def test_categorical_detection(self):
        """Test categorical feature detection."""
        df = pd.DataFrame({
            "num": [1, 2, 3],
            "cat": ["A", "B", "A"],
            "target": [10, 20, 30]
        })
        
        X, _, _ = prepare_features_targets(df, target_col="target")
        numerical, categorical = get_feature_types(X)
        
        assert "num" in numerical
        assert "cat" in categorical
    
    def test_excluded_columns(self):
        """Test that excluded columns are removed from features."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "CO2 Emissions": [100, 200, 300],
            "NOx Emissions": [1, 2, 3],
            "Emission Level": ["High", "Medium", "Low"]
        })
        
        X, y, features = prepare_features_targets(df, target_col="CO2 Emissions")
        
        assert "CO2 Emissions" not in features
        assert "NOx Emissions" not in features
        assert "Emission Level" not in features
        assert "feature1" in features
        assert "feature2" in features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])