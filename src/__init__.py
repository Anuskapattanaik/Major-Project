"""Vehicle Carbon Emission Prediction Package."""

from .config import config, ProjectConfig, DataConfig, ModelConfig, OutputConfig, ScenarioConfig
from .data_loader import load_dataset, inspect_dataset, check_data_quality, clean_dataset, prepare_features_targets, get_feature_types
from .preprocessing import create_preprocessing_pipeline, create_full_pipeline, fit_preprocessing_pipeline, get_model_specific_scaling
from .train import train_all_models, select_best_model, save_final_model, load_model, load_model_metadata
from .evaluate import evaluate_model_detailed, run_full_evaluation, create_evaluation_report
from .predict import predict, predict_single, create_sample_input
from .future_scenarios import run_all_scenarios, run_scenarios_from_config, create_scenario_config_template
from .visualization import generate_eda_plots

__version__ = "1.0.0"
__author__ = "Vehicle Emission Prediction Project"

__all__ = [
    "config",
    "ProjectConfig",
    "DataConfig", 
    "ModelConfig",
    "OutputConfig",
    "ScenarioConfig",
    "load_dataset",
    "inspect_dataset",
    "check_data_quality",
    "clean_dataset",
    "prepare_features_targets",
    "get_feature_types",
    "create_preprocessing_pipeline",
    "create_full_pipeline",
    "fit_preprocessing_pipeline",
    "get_model_specific_scaling",
    "train_all_models",
    "select_best_model",
    "save_final_model",
    "load_model",
    "load_model_metadata",
    "evaluate_model_detailed",
    "run_full_evaluation",
    "create_evaluation_report",
    "predict",
    "predict_single",
    "create_sample_input",
    "run_all_scenarios",
    "run_scenarios_from_config",
    "create_scenario_config_template",
    "generate_eda_plots"
]