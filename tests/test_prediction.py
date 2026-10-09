"""Tests for prediction module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import (
    prepare_input_data,
    predict_single,
    create_sample_input,
    validate_prediction_input
)
from src.train import save_final_model
from src.preprocessing import create_full_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor


class TestPrediction:
    """Tests for prediction functions."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a simple trained model for testing."""
        # Create sample training data with simple feature names
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "num1": np.random.randn(n),
            "num2": np.random.randn(n) * 10,
            "cat1": np.random.choice(["A", "B", "C"], n),
            "target": np.random.randn(n) * 50 + 100
        })
        
        X = df[["num1", "num2", "cat1"]]
        y = df["target"]
        
        # Create and fit pipeline
        pipeline = create_full_pipeline(
            LinearRegression(),
            ["num1", "num2"],
            ["cat1"],
            scale_numerical=True
        )
        pipeline.fit(X, y)
        
        # Save model and metadata
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.joblib"
            metadata_path = Path(tmpdir) / "test_metadata.json"
            
            import json
            metadata = {
                "model_name": "LinearRegression",
                "target_column": "target",
                "feature_names": ["num1", "num2", "cat1"],
                "numerical_features": ["num1", "num2"],
                "categorical_features": ["cat1"],
                "train_config": {"test_size": 0.2, "random_state": 42},
                "test_metrics": {"MAE": 10.0, "RMSE": 15.0, "R2": 0.5}
            }
            
            joblib.dump(pipeline, model_path)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            yield model_path, metadata_path
    
    def test_prepare_input_data_dict(self, sample_model):
        """Test preparing input from dictionary."""
        model_path, metadata_path = sample_model
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        input_data = {"num1": 1.0, "num2": 5.0, "cat1": "A"}
        df = prepare_input_data(input_data, metadata["feature_names"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == metadata["feature_names"]
    
    def test_prepare_input_data_list(self, sample_model):
        """Test preparing input from list of dicts."""
        model_path, metadata_path = sample_model
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        input_data = [
            {"num1": 1.0, "num2": 5.0, "cat1": "A"},
            {"num1": 2.0, "num2": 10.0, "cat1": "B"}
        ]
        df = prepare_input_data(input_data, metadata["feature_names"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_prepare_input_data_dataframe(self, sample_model):
        """Test preparing input from DataFrame."""
        model_path, metadata_path = sample_model
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        df_input = pd.DataFrame({
            "num1": [1.0, 2.0],
            "num2": [5.0, 10.0],
            "cat1": ["A", "B"]
        })
        df = prepare_input_data(df_input, metadata["feature_names"])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    
    def test_prepare_input_data_missing_features(self, sample_model):
        """Test preparing input with missing features (should add NaN)."""
        model_path, metadata_path = sample_model
        
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Missing num2
        input_data = {"num1": 1.0, "cat1": "A"}
        df = prepare_input_data(input_data, metadata["feature_names"])
        
        assert "num2" in df.columns
        assert df["num2"].isnull().all()
    
    def test_predict_single(self, sample_model):
        """Test single prediction."""
        model_path, metadata_path = sample_model
        
        input_data = {"num1": 1.0, "num2": 5.0, "cat1": "A"}
        prediction = predict_single(input_data, model_path, metadata_path)
        
        assert isinstance(prediction, float)
        assert not np.isnan(prediction)
    
    def test_create_sample_input(self):
        """Test creating sample input file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            created = create_sample_input(path)
            
            assert created == path
            assert path.exists()
            
            # Check content
            df = pd.read_csv(path)
            assert len(df) > 0
            assert "Vehicle Type" in df.columns
            assert "Fuel Type" in df.columns
    
    def test_validate_prediction_input(self):
        """Test prediction input validation."""
        numerical = ["num1", "num2"]
        categorical = ["cat1"]
        
        # Valid
        df = pd.DataFrame({"num1": [1], "num2": [2], "cat1": ["A"]})
        result = validate_prediction_input(df, numerical, categorical)
        assert result["valid"] is True
        
        # Missing numerical
        df = pd.DataFrame({"num1": [1], "cat1": ["A"]})
        result = validate_prediction_input(df, numerical, categorical)
        assert result["valid"] is False
        assert "num2" in result["missing_numerical"]


class TestPredictionIntegration:
    """Integration tests for prediction workflow."""
    
    def test_full_predict_workflow(self):
        """Test complete prediction workflow."""
        # Train a simple model
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "Engine Size": np.random.uniform(1, 6, n),
            "Age of Vehicle": np.random.randint(0, 30, n),
            "Fuel Type": np.random.choice(["Petrol", "Diesel", "Electric"], n),
            "CO2 Emissions": np.random.uniform(50, 500, n)
        })
        
        from src.data_loader import prepare_features_targets, get_feature_types
        from src.preprocessing import create_full_pipeline
        
        X, y, features = prepare_features_targets(df, target_col="CO2 Emissions", 
                                                   exclude_cols=["CO2 Emissions", "NOx Emissions", "PM2.5 Emissions", "VOC Emissions", "SO2 Emissions", "Emission Level"])
        numerical, categorical = get_feature_types(X)
        
        pipeline = create_full_pipeline(LinearRegression(), numerical, categorical, scale_numerical=True)
        pipeline.fit(X, y)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            metadata_path = Path(tmpdir) / "metadata.json"
            
            joblib.dump(pipeline, model_path)
            
            import json
            metadata = {
                "model_name": "LinearRegression",
                "target_column": "CO2 Emissions",
                "feature_names": features,
                "numerical_features": numerical,
                "categorical_features": categorical,
                "train_config": {},
                "test_metrics": {}
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
            
            # Predict
            new_vehicle = {
                "Engine Size": 2.0,
                "Age of Vehicle": 5,
                "Fuel Type": "Petrol"
            }
            
            prediction = predict_single(new_vehicle, model_path, metadata_path)
            
            assert isinstance(prediction, float)
            assert prediction > 0