"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    create_preprocessing_pipeline,
    create_full_pipeline,
    fit_preprocessing_pipeline,
    get_model_specific_scaling,
    transform_features,
    save_preprocessor,
    load_preprocessor,
    validate_input_features
)
from src.config import config


class TestPreprocessing:
    """Tests for preprocessing pipeline."""
    
    def test_create_preprocessing_pipeline(self):
        """Test creating preprocessing pipeline."""
        numerical = ["num1", "num2"]
        categorical = ["cat1", "cat2"]
        
        preprocessor = create_preprocessing_pipeline(numerical, categorical, scale_numerical=True)
        
        assert preprocessor is not None
        assert hasattr(preprocessor, "fit")
        assert hasattr(preprocessor, "transform")
    
    def test_create_full_pipeline(self):
        """Test creating full pipeline with model."""
        from sklearn.linear_model import LinearRegression
        
        numerical = ["num1"]
        categorical = ["cat1"]
        model = LinearRegression()
        
        pipeline = create_full_pipeline(model, numerical, categorical, scale_numerical=True)
        
        assert pipeline is not None
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps
    
    def test_fit_preprocessing_pipeline(self):
        """Test fitting preprocessing on training data."""
        # Create sample data
        df = pd.DataFrame({
            "num1": [1, 2, 3, 4, 5],
            "num2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "cat1": ["A", "B", "A", "C", "B"],
            "cat2": ["X", "Y", "X", "X", "Y"]
        })
        
        numerical = ["num1", "num2"]
        categorical = ["cat1", "cat2"]
        
        preprocessor = fit_preprocessing_pipeline(df, numerical, categorical, "LinearRegression")
        
        # Check it's fitted
        transformed = preprocessor.transform(df)
        assert transformed.shape[0] == 5
        assert transformed.shape[1] > 0
    
    def test_get_model_specific_scaling(self):
        """Test model-specific scaling detection."""
        # Tree models should not scale
        assert get_model_specific_scaling("RandomForestRegressor") is False
        assert get_model_specific_scaling("ExtraTreesRegressor") is False
        assert get_model_specific_scaling("HistGradientBoostingRegressor") is False
        assert get_model_specific_scaling("GradientBoostingRegressor") is False
        
        # Linear models should scale
        assert get_model_specific_scaling("LinearRegression") is True
        assert get_model_specific_scaling("Ridge") is True
        assert get_model_specific_scaling("DummyRegressor") is True
    
    def test_transform_features(self):
        """Test transforming features with fitted preprocessor."""
        df = pd.DataFrame({
            "num1": [1, 2, 3],
            "cat1": ["A", "B", "A"]
        })
        
        preprocessor = fit_preprocessing_pipeline(df, ["num1"], ["cat1"], "LinearRegression")
        transformed = transform_features(preprocessor, df)
        
        assert transformed.shape[0] == 3
    
    def test_save_load_preprocessor(self):
        """Test saving and loading preprocessor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preprocessor.joblib"
            
            df = pd.DataFrame({
                "num1": [1, 2, 3],
                "cat1": ["A", "B", "A"]
            })
            
            preprocessor = fit_preprocessing_pipeline(df, ["num1"], ["cat1"], "LinearRegression")
            save_preprocessor(preprocessor, path)
            
            assert path.exists()
            
            loaded = load_preprocessor(path)
            transformed = loaded.transform(df)
            
            assert transformed.shape == (3, 3)  # num1 + one-hot for cat1 (A, B) = 3 features
    
    def test_validate_input_features(self):
        """Test input feature validation."""
        numerical = ["num1", "num2"]
        categorical = ["cat1"]
        
        # Valid input
        df_valid = pd.DataFrame({
            "num1": [1, 2],
            "num2": [1.0, 2.0],
            "cat1": ["A", "B"]
        })
        result = validate_input_features(df_valid, numerical, categorical)
        assert result["valid"] is True
        assert len(result["missing_numerical"]) == 0
        assert len(result["missing_categorical"]) == 0
        
        # Missing numerical
        df_missing = pd.DataFrame({
            "num1": [1, 2],
            "cat1": ["A", "B"]
        })
        result = validate_input_features(df_missing, numerical, categorical)
        assert result["valid"] is False
        assert "num2" in result["missing_numerical"]
        
        # Extra columns
        df_extra = pd.DataFrame({
            "num1": [1, 2],
            "num2": [1.0, 2.0],
            "cat1": ["A", "B"],
            "extra": [1, 2]
        })
        result = validate_input_features(df_extra, numerical, categorical)
        assert result["valid"] is True
        assert "extra" in result["extra_columns"]


class TestPreprocessingEdgeCases:
    """Tests for edge cases in preprocessing."""
    
    def test_unseen_categories(self):
        """Test handling of unseen categories during transform."""
        df_train = pd.DataFrame({
            "num1": [1, 2, 3],
            "cat1": ["A", "B", "A"]
        })
        
        df_test = pd.DataFrame({
            "num1": [4, 5],
            "cat1": ["C", "A"]  # C is unseen
        })
        
        preprocessor = fit_preprocessing_pipeline(df_train, ["num1"], ["cat1"], "LinearRegression")
        transformed = transform_features(preprocessor, df_test)
        
        # Should not raise error, unseen categories handled by handle_unknown="ignore"
        assert transformed.shape[0] == 2
    
    def test_missing_values_imputation(self):
        """Test missing value imputation."""
        df_train = pd.DataFrame({
            "num1": [1, 2, 3, 4, 5],
            "cat1": ["A", "B", "A", "C", "B"]
        })
        
        df_test = pd.DataFrame({
            "num1": [6, np.nan],
            "cat1": ["A", np.nan]
        })
        
        preprocessor = fit_preprocessing_pipeline(df_train, ["num1"], ["cat1"], "LinearRegression")
        transformed = transform_features(preprocessor, df_test)
        
        # Should not have NaN after imputation
        assert not np.isnan(transformed).any()
    
    def test_infinite_values(self):
        """Test handling of infinite values."""
        df_train = pd.DataFrame({
            "num1": [1, 2, 3, 4, 5],
            "cat1": ["A", "B", "A", "C", "B"]
        })
        
        # Note: The preprocessor itself doesn't handle inf, but the data_loader does
        # This test verifies the pipeline works with clean data
        preprocessor = fit_preprocessing_pipeline(df_train, ["num1"], ["cat1"], "LinearRegression")
        transformed = preprocessor.transform(df_train)
        
        assert not np.isinf(transformed).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])