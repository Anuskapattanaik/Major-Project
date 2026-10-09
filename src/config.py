"""Centralized configuration for the vehicle emission prediction project."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    """Configuration for data loading and paths."""
    # Primary data path
    primary_data_path: Path = Path("data/vehicle_emission_dataset.csv")
    # Fallback path (project root)
    fallback_data_path: Path = Path("vehicle_emission_dataset.csv")
    
    # Target column
    target_column: str = "CO2 Emissions"
    
    # Columns to exclude from features (target leakage prevention)
    exclude_columns: List[str] = field(default_factory=lambda: [
        "CO2 Emissions",
        "NOx Emissions", 
        "PM2.5 Emissions",
        "VOC Emissions",
        "SO2 Emissions",
        "Emission Level"
    ])
    
    # Test split configuration
    test_size: float = 0.2
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    
    # Data quality thresholds
    missing_threshold: float = 0.5  # Drop columns with >50% missing
    duplicate_threshold: float = 1.0  # Report duplicates


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # Models to train
    models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "DummyRegressor": {"strategy": "mean"},
        "LinearRegression": {},
        "Ridge": {"alpha": 1.0},
        "RandomForestRegressor": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "ExtraTreesRegressor": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "HistGradientBoostingRegressor": {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_depth": None,
            "random_state": 42
        },
        "GradientBoostingRegressor": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42
        }
    })
    
    # Selection metric (lower is better for MAE/RMSE, higher for R2)
    selection_metric: str = "RMSE"
    selection_metric_lower_is_better: bool = True
    
    # Multi-output configuration
    multi_output: bool = False
    secondary_targets: List[str] = field(default_factory=lambda: [
        "NOx Emissions",
        "PM2.5 Emissions", 
        "VOC Emissions",
        "SO2 Emissions"
    ])


@dataclass
class OutputConfig:
    """Configuration for output paths."""
    # Output directories
    figures_dir: Path = Path("outputs/figures")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    outputs_dir: Path = Path("outputs")
    
    # Output files
    data_quality_report: str = "data_quality_report.csv"
    model_comparison: str = "model_comparison.csv"
    predictions: str = "predictions.csv"
    future_scenarios: str = "future_emission_scenarios.csv"
    feature_importance: str = "feature_importance.csv"
    model_evaluation_report: str = "model_evaluation_report.md"
    final_model: str = "final_model.joblib"
    model_metadata: str = "model_metadata.json"


@dataclass
class ScenarioConfig:
    """Configuration for future emission scenarios."""
    base_year: int = 2024
    future_years: List[int] = field(default_factory=lambda: [2025, 2026, 2027, 2028, 2029, 2030])
    
    # Default scenarios (user should customize)
    scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "business_as_usual": {
            "fleet_growth_rate": 0.02,  # 2% annual fleet growth
            "emission_change_rate": 0.0,  # No change in avg emissions per vehicle
        },
        "emission_reduction": {
            "fleet_growth_rate": 0.02,
            "emission_change_rate": -0.03,  # 3% annual reduction
        },
        "high_growth": {
            "fleet_growth_rate": 0.05,  # 5% annual fleet growth
            "emission_change_rate": 0.01,  # 1% annual increase in emissions
        }
    })
    
    # Starting fleet size (if not provided, use dataset size)
    starting_fleet_size: Optional[int] = None
    
    # Average annual distance per vehicle (km/year) - needed for trip-to-annual conversion
    avg_annual_distance_km: float = 15000.0
    
    # Whether target represents per-trip or per-km emissions (for interpretation)
    target_unit: str = "unknown"  # Could be "g/trip", "g/km", "g/vehicle", etc.


@dataclass
class ProjectConfig:
    """Main project configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    def __post_init__(self):
        """Create output directories."""
        self.output.figures_dir.mkdir(parents=True, exist_ok=True)
        self.output.models_dir.mkdir(parents=True, exist_ok=True)
        self.output.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output.outputs_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = ProjectConfig()


def get_data_path() -> Path:
    """Get the data file path, checking primary then fallback."""
    if config.data.primary_data_path.exists():
        return config.data.primary_data_path
    elif config.data.fallback_data_path.exists():
        return config.data.fallback_data_path
    else:
        raise FileNotFoundError(
            f"Dataset not found at {config.data.primary_data_path} "
            f"or {config.data.fallback_data_path}"
        )


def get_output_path(filename: str) -> Path:
    """Get full path for an output file."""
    return config.output.outputs_dir / filename


def get_figure_path(filename: str) -> Path:
    """Get full path for a figure file."""
    return config.output.figures_dir / filename


def get_model_path(filename: str) -> Path:
    """Get full path for a model file."""
    return config.output.models_dir / filename


def get_report_path(filename: str) -> Path:
    """Get full path for a report file."""
    return config.output.reports_dir / filename