"""Future-year emission scenario forecasting module."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import warnings

from .config import config, get_output_path, get_figure_path
from .visualization import plot_future_scenarios

logger = logging.getLogger(__name__)


def calculate_fleet_size(
    base_fleet: int,
    growth_rate: float,
    years: List[int],
    base_year: int
) -> Dict[int, float]:
    """
    Calculate fleet size for each future year.
    
    Formula: Fleet(t) = Fleet(base) * (1 + growth_rate)^(t - base_year)
    
    Args:
        base_fleet: Starting fleet size
        growth_rate: Annual fleet growth rate (e.g., 0.02 for 2%)
        years: List of future years
        base_year: Base year
        
    Returns:
        Dictionary mapping year to fleet size
    """
    if growth_rate < -1:
        raise ValueError("Growth rate cannot be less than -1 (100% decline)")
    
    fleet = {}
    for year in years:
        t = year - base_year
        fleet[year] = base_fleet * ((1 + growth_rate) ** t)
    
    return fleet


def calculate_average_emissions(
    base_emission: float,
    change_rate: float,
    years: List[int],
    base_year: int
) -> Dict[int, float]:
    """
    Calculate average emissions per vehicle for each future year.
    
    Formula: AvgEmission(t) = AvgEmission(base) * (1 + change_rate)^(t - base_year)
    
    Args:
        base_emission: Base average emission per vehicle
        change_rate: Annual emission change rate (negative for reduction)
        years: List of future years
        base_year: Base year
        
    Returns:
        Dictionary mapping year to average emissions
    """
    if change_rate < -1:
        raise ValueError("Emission change rate cannot be less than -1 (100% reduction)")
    
    emissions = {}
    for year in years:
        t = year - base_year
        emissions[year] = base_emission * ((1 + change_rate) ** t)
    
    return emissions


def calculate_total_emissions(
    fleet: Dict[int, float],
    avg_emissions: Dict[int, float]
) -> Dict[int, float]:
    """
    Calculate total annual emissions.
    
    Formula: TotalEmission(t) = Fleet(t) * AvgEmission(t)
    
    Args:
        fleet: Fleet size by year
        avg_emissions: Average emissions by year
        
    Returns:
        Dictionary mapping year to total emissions
    """
    total = {}
    for year in fleet:
        if year in avg_emissions:
            total[year] = fleet[year] * avg_emissions[year]
    
    return total


def run_scenario(
    scenario_name: str,
    base_fleet: int,
    base_emission: float,
    fleet_growth_rate: float,
    emission_change_rate: float,
    years: List[int],
    base_year: int
) -> pd.DataFrame:
    """
    Run a single scenario.
    
    Args:
        scenario_name: Name of the scenario
        base_fleet: Starting fleet size
        base_emission: Base average emissions per vehicle
        fleet_growth_rate: Annual fleet growth rate
        emission_change_rate: Annual emission change rate
        years: Future years to simulate
        base_year: Base year
        
    Returns:
        DataFrame with scenario results
    """
    fleet = calculate_fleet_size(base_fleet, fleet_growth_rate, years, base_year)
    avg_emissions = calculate_average_emissions(base_emission, emission_change_rate, years, base_year)
    total = calculate_total_emissions(fleet, avg_emissions)
    
    rows = []
    for year in years:
        rows.append({
            "scenario": scenario_name,
            "year": year,
            "fleet_size": fleet[year],
            "avg_emission_per_vehicle": avg_emissions[year],
            "total_emissions": total[year],
            "fleet_growth_rate": fleet_growth_rate,
            "emission_change_rate": emission_change_rate
        })
    
    return pd.DataFrame(rows)


def run_all_scenarios(
    base_fleet: int,
    base_emission: float,
    years: List[int],
    base_year: int,
    scenarios: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Run all configured scenarios.
    
    Args:
        base_fleet: Starting fleet size
        base_emission: Base average emissions per vehicle
        years: Future years to simulate
        base_year: Base year
        scenarios: Optional custom scenarios dict
        
    Returns:
        Combined DataFrame with all scenarios
    """
    if scenarios is None:
        scenarios = config.scenario.scenarios
    
    all_results = []
    for name, params in scenarios.items():
        try:
            df = run_scenario(
                name,
                base_fleet,
                base_emission,
                params.get("fleet_growth_rate", 0.0),
                params.get("emission_change_rate", 0.0),
                years,
                base_year
            )
            all_results.append(df)
            logger.info(f"Scenario '{name}' completed")
        except Exception as e:
            logger.error(f"Scenario '{name}' failed: {e}")
    
    if not all_results:
        raise ValueError("All scenarios failed")
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined


def estimate_base_emission_from_model(
    pipeline: Any,
    X_sample: pd.DataFrame,
    target_unit: str = "unknown"
) -> Tuple[float, str]:
    """
    Estimate base average emission from model predictions on sample data.
    
    Args:
        pipeline: Fitted prediction pipeline
        X_sample: Sample feature data
        target_unit: Unit of the target variable
        
    Returns:
        Tuple of (average_emission, unit_warning)
    """
    predictions = pipeline.predict(X_sample)
    avg_emission = float(np.mean(predictions))
    
    warning = ""
    if target_unit == "unknown":
        warning = (
            "WARNING: Target unit is unknown. The model predicts values in the same "
            "unit as the training target. For annual fleet emissions, ensure the target "
            "represents annual per-vehicle emissions. If it represents per-trip or "
            "per-km emissions, you must convert using average annual distance/usage."
        )
        logger.warning(warning)
    
    return avg_emission, warning


def run_fleet_composition_scenario(
    pipeline: Any,
    fleet_composition: Dict[str, Dict[str, float]],
    feature_template: Dict[str, Any],
    years: List[int],
    base_year: int,
    fleet_growth_rate: float = 0.02,
    emission_change_rate: float = 0.0
) -> pd.DataFrame:
    """
    Run scenario with changing fleet composition.
    
    Args:
        pipeline: Fitted prediction pipeline
        fleet_composition: Dict mapping year to {vehicle_type: fraction, fuel_type: fraction}
        feature_template: Base feature values (will be modified for vehicle/fuel type)
        years: Future years
        base_year: Base year
        fleet_growth_rate: Annual fleet growth rate
        emission_change_rate: Annual emission change rate
        
    Returns:
        DataFrame with scenario results
    """
    # This is a simplified version - in practice, you'd need more detailed
    # feature generation for each vehicle/fuel combination
    logger.warning("Fleet composition scenario is simplified; results are illustrative")
    
    # For now, just use base emission with composition note
    base_emission = estimate_base_emission_from_model(pipeline, pd.DataFrame([feature_template]))
    
    return run_all_scenarios(
        base_fleet=len(X_sample) if 'X_sample' in locals() else 1000,
        base_emission=base_emission[0],
        years=years,
        base_year=base_year,
        scenarios={
            "custom": {
                "fleet_growth_rate": fleet_growth_rate,
                "emission_change_rate": emission_change_rate
            }
        }
    )


def save_scenarios(
    scenarios_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """
    Save scenarios to CSV.
    
    Args:
        scenarios_df: Scenarios DataFrame
        output_path: Optional custom path
        
    Returns:
        Path to saved CSV
    """
    if output_path is None:
        output_path = get_output_path("future_emission_scenarios.csv")
    
    scenarios_df.to_csv(output_path, index=False)
    logger.info(f"Scenarios saved to {output_path}")
    return output_path


def create_scenario_config_template(
    output_path: Optional[Path] = None
) -> Path:
    """
    Create a JSON template for scenario configuration.
    
    Args:
        output_path: Optional custom path
        
    Returns:
        Path to created template
    """
    if output_path is None:
        output_path = get_output_path("scenario_config_template.json")
    
    template = {
        "base_year": 2024,
        "future_years": [2025, 2026, 2027, 2028, 2029, 2030],
        "base_fleet_size": 100000,
        "base_avg_emission_per_vehicle": 200.0,
        "target_unit": "g/vehicle (annual)",
        "scenarios": {
            "business_as_usual": {
                "fleet_growth_rate": 0.02,
                "emission_change_rate": 0.0,
                "description": "Current trends continue unchanged"
            },
            "emission_reduction": {
                "fleet_growth_rate": 0.02,
                "emission_change_rate": -0.03,
                "description": "3% annual emission reduction per vehicle"
            },
            "high_growth": {
                "fleet_growth_rate": 0.05,
                "emission_change_rate": 0.01,
                "description": "High fleet growth with slight emission increase"
            }
        },
        "notes": [
            "All rates are annual (e.g., 0.02 = 2%)",
            "Emission change rate: negative = reduction, positive = increase",
            "Base fleet size and emission should be calibrated to your context",
            "Target unit must match model prediction unit for meaningful results",
            "These are CONDITIONAL SCENARIOS, not forecasts"
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Scenario config template saved to {output_path}")
    return output_path


def load_scenario_config(config_path: Path) -> Dict[str, Any]:
    """
    Load scenario configuration from JSON.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    # Validate
    required = ["base_year", "future_years", "base_fleet_size", "base_avg_emission_per_vehicle"]
    for key in required:
        if key not in config_data:
            raise ValueError(f"Missing required config key: {key}")
    
    return config_data


def run_scenarios_from_config(
    config_path: Path,
    pipeline: Optional[Any] = None,
    X_sample: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Run scenarios from a configuration file.
    
    Args:
        config_path: Path to scenario config JSON
        pipeline: Optional fitted pipeline to estimate base emission
        X_sample: Optional sample data for pipeline prediction
        
    Returns:
        Scenarios DataFrame
    """
    cfg = load_scenario_config(config_path)
    
    base_emission = cfg["base_avg_emission_per_vehicle"]
    
    # If pipeline provided, estimate base emission from model
    if pipeline is not None and X_sample is not None:
        base_emission, warning = estimate_base_emission_from_model(
            pipeline, X_sample, cfg.get("target_unit", "unknown")
        )
        logger.info(f"Estimated base emission from model: {base_emission:.2f}")
        if warning:
            logger.warning(warning)
    
    results = run_all_scenarios(
        base_fleet=cfg["base_fleet_size"],
        base_emission=base_emission,
        years=cfg["future_years"],
        base_year=cfg["base_year"],
        scenarios=cfg.get("scenarios")
    )
    
    return results