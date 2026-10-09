"""Command-line interface for the vehicle emission prediction project."""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import config, get_data_path
from src.data_loader import (
    load_dataset, inspect_dataset, check_data_quality, 
    clean_dataset, prepare_features_targets, get_feature_types
)
from src.preprocessing import get_model_specific_scaling
from src.train import (
    train_all_models, select_best_model, save_model_comparison,
    save_final_model, load_model, load_model_metadata
)
from src.evaluate import run_full_evaluation
from src.predict import predict, create_sample_input
from src.future_scenarios import (
    run_all_scenarios, run_scenarios_from_config, create_scenario_config_template
)
from src.visualization import generate_eda_plots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Set logging level."""
    logging.getLogger().setLevel(getattr(logging, level.upper()))


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect dataset command."""
    try:
        df = load_dataset()
        inspection = inspect_dataset(df)
        
        print(f"\n{'='*60}")
        print(f"DATASET INSPECTION")
        print(f"{'='*60}")
        print(f"Shape: {inspection['shape']['rows']} rows x {inspection['shape']['columns']} columns")
        print(f"\nColumns: {inspection['columns']}")
        print(f"\nDtypes:")
        for col, dtype in inspection['dtypes'].items():
            print(f"  {col}: {dtype}")
        
        print(f"\nMissing values: {inspection['total_missing']} total")
        for col, cnt in inspection['missing_values'].items():
            pct = inspection['missing_percentage'].get(col, 0)
            print(f"  {col}: {cnt} ({pct}%)")
        
        print(f"\nDuplicate rows: {inspection['duplicate_rows']}")
        
        if inspection['infinite_values']:
            print(f"\nInfinite values:")
            for col, cnt in inspection['infinite_values'].items():
                print(f"  {col}: {cnt}")
        
        if inspection.get('target_exists'):
            ts = inspection['target_stats']
            print(f"\nTarget ({config.data.target_column}):")
            print(f"  Count: {ts['count']}")
            print(f"  Mean: {ts['mean']:.2f}")
            print(f"  Std: {ts['std']:.2f}")
            print(f"  Min: {ts['min']:.2f}")
            print(f"  Max: {ts['max']:.2f}")
            print(f"  Missing: {ts['missing']}")
        
        return 0
    except Exception as e:
        logger.error(f"Inspection failed: {e}")
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run full EDA and data quality analysis."""
    try:
        df = load_dataset()
        
        # Inspection
        inspection = inspect_dataset(df)
        
        # Quality checks
        quality = check_data_quality(df)
        
        # Cleaning
        df_clean, cleaning = clean_dataset(df)
        
        # Save quality report
        from src.data_loader import save_data_quality_report
        save_data_quality_report(inspection, quality, cleaning)
        
        # Generate EDA plots
        generate_eda_plots(df_clean)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Original shape: {cleaning['original_shape']}")
        print(f"Cleaned shape: {cleaning['cleaned_shape']}")
        print(f"Rows removed: {cleaning['rows_removed']}")
        print(f"Columns removed: {cleaning['columns_removed']}")
        print(f"\nEDA plots saved to outputs/figures/")
        print(f"Quality report saved to reports/data_quality_report.json")
        
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train models command."""
    try:
        # Load and prepare data
        df = load_dataset()
        df_clean, _ = clean_dataset(df)
        
        X, y, feature_names = prepare_features_targets(df_clean)
        numerical_features, categorical_features = get_feature_types(X)
        
        print(f"\n{'='*60}")
        print(f"TRAINING MODELS")
        print(f"{'='*60}")
        print(f"Features: {len(feature_names)} ({len(numerical_features)} num, {len(categorical_features)} cat)")
        print(f"Target: {config.data.target_column}")
        print(f"Train/Test split: {1-config.data.test_size:.0%}/{config.data.test_size:.0%}")
        
        # Train all models
        results, X_train, y_train, X_test, y_test = train_all_models(
            X, y, numerical_features, categorical_features,
            test_size=config.data.test_size,
            random_state=config.data.random_state
        )
        
        # Save comparison
        save_model_comparison(results)
        
        # Select best model
        best_name, best_result = select_best_model(
            results,
            metric=config.model.selection_metric,
            lower_is_better=config.model.selection_metric_lower_is_better
        )
        
        # Save final model
        train_config = {
            "test_size": config.data.test_size,
            "random_state": config.data.random_state,
            "cv_folds": config.data.cv_folds,
            "selection_metric": config.model.selection_metric
        }
        
        save_final_model(
            best_result["pipeline"],
            best_name,
            numerical_features,
            categorical_features,
            feature_names,
            config.data.target_column,
            train_config,
            best_result["test_metrics"]
        )
        
        # Evaluate on test set
        run_full_evaluation(
            best_result["pipeline"],
            X_test, y_test,
            best_name,
            load_model_metadata(),
            feature_names
        )
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best model: {best_name}")
        print(f"Test RMSE: {best_result['test_metrics']['RMSE']:.4f}")
        print(f"Test R²: {best_result['test_metrics']['R2']:.4f}")
        print(f"Model saved to models/final_model.joblib")
        
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_predict(args: argparse.Namespace) -> int:
    """Make predictions command."""
    try:
        input_path = Path(args.input)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        output_path = Path(args.output) if args.output else None
        
        predictions = predict(input_path, output_path=output_path)
        
        print(f"\n{'='*60}")
        print(f"PREDICTIONS COMPLETE")
        print(f"{'='*60}")
        print(f"Predicted {len(predictions)} records")
        print(f"Results saved to {output_path or 'outputs/predictions.csv'}")
        
        # Show preview
        pred_col = f"predicted_{config.data.target_column}"
        if pred_col in predictions.columns:
            print(f"\nPrediction summary:")
            print(predictions[pred_col].describe())
        
        return 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_forecast(args: argparse.Namespace) -> int:
    """Run future scenarios command."""
    try:
        config_path = Path(args.config)
        
        if not config_path.exists():
            # Create template
            print(f"Config file not found. Creating template at {config_path}")
            create_scenario_config_template(config_path)
            print("Please edit the template and run again.")
            return 0
        
        # Load model if available for base emission estimation
        pipeline = None
        X_sample = None
        try:
            pipeline = load_model()
            metadata = load_model_metadata()
            # Use a small sample from training data
            df = load_dataset()
            df_clean, _ = clean_dataset(df)
            X, _, _ = prepare_features_targets(df_clean)
            X_sample = X.head(100)
        except Exception:
            logger.warning("Could not load model for base emission estimation; using config value")
        
        # Run scenarios
        results = run_scenarios_from_config(config_path, pipeline, X_sample)
        
        # Save
        output_path = Path(args.output) if args.output else None
        from src.future_scenarios import save_scenarios
        save_scenarios(results, output_path)
        
        # Plot
        from src.visualization import plot_future_scenarios
        plot_future_scenarios(
            results,
            save_path=config.output.figures_dir / "future_scenarios.png"
        )
        
        print(f"\n{'='*60}")
        print(f"SCENARIOS COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to {output_path or 'outputs/future_emission_scenarios.csv'}")
        print(f"Plot saved to outputs/figures/future_scenarios.png")
        
        # Print summary
        print("\nScenario Summary (final year):")
        final_year = results['year'].max()
        for scenario in results['scenario'].unique():
            subset = results[results['scenario'] == scenario]
            final = subset[subset['year'] == final_year].iloc[0]
            print(f"  {scenario}: fleet={final['fleet_size']:.0f}, "
                  f"avg_emission={final['avg_emission_per_vehicle']:.2f}, "
                  f"total={final['total_emissions']:.2f}")
        
        return 0
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_all(args: argparse.Namespace) -> int:
    """Run full pipeline: inspect, analyze, train, predict, forecast."""
    commands = [
        ("inspect", cmd_inspect),
        ("analyze", cmd_analyze),
        ("train", cmd_train),
    ]
    
    for name, func in commands:
        print(f"\n{'#'*60}")
        print(f"# RUNNING: {name.upper()}")
        print(f"{'#'*60}")
        result = func(args)
        if result != 0:
            print(f"\n{name} failed with exit code {result}")
            return result
    
    # Create sample input for prediction
    create_sample_input()
    
    # Predict on sample
    sample_path = config.output.outputs_dir / "sample_input.csv"
    if sample_path.exists():
        print(f"\n{'#'*60}")
        print(f"# RUNNING: predict (on sample)")
        print(f"{'#'*60}")
        cmd_predict(argparse.Namespace(input=str(sample_path), output=None))
    
    # Create scenario template and run
    print(f"\n{'#'*60}")
    print(f"# RUNNING: forecast (with template)")
    print(f"{'#'*60}")
    template_path = config.output.outputs_dir / "scenario_config_template.json"
    create_scenario_config_template(template_path)
    cmd_forecast(argparse.Namespace(config=str(template_path), output=None))
    
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE COMPLETE")
    print(f"{'='*60}")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Vehicle Carbon Emission Prediction and Future-Year Scenario Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  inspect     - Inspect dataset (shape, columns, missing values, target stats)
  analyze     - Full EDA and data quality analysis with plots
  train       - Train and compare models, save best model
  predict     - Make predictions on new data (CSV or single record)
  forecast    - Run future-year emission scenarios
  all         - Run full pipeline (inspect, analyze, train, predict, forecast)

Examples:
  python main.py inspect
  python main.py analyze
  python main.py train
  python main.py predict --input data/new_vehicles.csv
  python main.py predict --input '{"Vehicle Type": "Car", "Fuel Type": "Petrol", ...}'
  python main.py forecast --config scenario_config.json
  python main.py all
"""
    )
    
    parser.add_argument(
        "command",
        choices=["inspect", "analyze", "train", "predict", "forecast", "all"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Input file for prediction (CSV or JSON)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="scenario_config.json",
        help="Scenario configuration file (for forecast)"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Route to command
    commands = {
        "inspect": cmd_inspect,
        "analyze": cmd_analyze,
        "train": cmd_train,
        "predict": cmd_predict,
        "forecast": cmd_forecast,
        "all": cmd_all
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())