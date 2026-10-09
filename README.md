# Vehicle Carbon Emission Prediction and Future-Year Scenario Forecasting

A complete, reproducible machine learning project for predicting vehicle CO₂ emissions and forecasting future annual transportation emissions under configurable scenarios.

## Project Overview

This project builds a machine learning system that:
1. **Analyzes and cleans** vehicle emission datasets
2. **Predicts CO₂ emissions** for individual vehicles using multiple regression algorithms
3. **Compares models** with proper cross-validation and held-out test evaluation
4. **Forecasts future annual emissions** under transparent, user-configurable scenarios
5. **Produces publication-quality** charts, tables, and reports suitable for research

## Problem Statement

Transportation is a major contributor to global carbon emissions. Understanding the factors that influence vehicle emissions and being able to predict emissions for new vehicles is crucial for policy-making, fleet management, and environmental planning. This project provides a complete pipeline from raw data to actionable predictions and scenario forecasts.

## Objectives

- Build reproducible ML pipeline for CO₂ emission prediction
- Compare multiple regression algorithms fairly
- Provide transparent future-year scenario forecasting
- Generate research-quality documentation and visualizations
- Support both programmatic and notebook-based workflows

## Dataset Description

The project expects a CSV file with the following columns (based on the provided dataset):

| Column | Type | Description |
|--------|------|-------------|
| Vehicle Type | Categorical | Car, Truck, Bus, Motorcycle |
| Fuel Type | Categorical | Petrol, Diesel, Electric, Hybrid |
| Engine Size | Numerical | Engine displacement |
| Age of Vehicle | Numerical | Vehicle age in years |
| Mileage | Numerical | Total distance travelled |
| Speed | Numerical | Current/average speed |
| Acceleration | Numerical | Acceleration rate |
| Road Type | Categorical | Highway, City, Rural |
| Traffic Conditions | Categorical | Free flow, Moderate, Heavy |
| Temperature | Numerical | Ambient temperature |
| Humidity | Numerical | Relative humidity |
| Wind Speed | Numerical | Wind speed |
| Air Pressure | Numerical | Atmospheric pressure |
| CO2 Emissions | Numerical | **Target variable** (CO₂ emissions) |
| NOx Emissions | Numerical | NOx pollutant emissions |
| PM2.5 Emissions | Numerical | Particulate matter emissions |
| VOC Emissions | Numerical | Volatile organic compound emissions |
| SO2 Emissions | Numerical | Sulfur dioxide emissions |
| Emission Level | Categorical | Low, Medium, High |

**Note**: The dataset does not contain a year column. All records are treated as cross-sectional observations. Future-year forecasting requires explicit assumptions (see [Scenario Assumptions](#scenario-assumptions)).

### Dataset Setup

Place your CSV file at either:
- `data/vehicle_emission_dataset.csv` (preferred)
- `vehicle_emission_dataset.csv` (project root, fallback)

## Folder Structure

```
vehicle-carbon-emission-prediction/
├── data/
│   └── vehicle_emission_dataset.csv     # Your dataset (not tracked)
├── notebooks/
│   └── vehicle_emission_analysis.ipynb  # Jupyter notebook (Colab-compatible)
├── src/
│   ├── __init__.py
│   ├── config.py                        # Centralized configuration
│   ├── data_loader.py                   # Data loading & quality checks
│   ├── preprocessing.py                 # Sklearn preprocessing pipelines
│   ├── train.py                         # Model training & comparison
│   ├── evaluate.py                      # Model evaluation & reporting
│   ├── predict.py                       # Prediction interface
│   ├── future_scenarios.py              # Future-year forecasting
│   └── visualization.py                 # Plotting functions
├── models/
│   └── final_model.joblib               # Trained model (generated)
├── outputs/
│   ├── figures/                         # Generated plots
│   ├── model_comparison.csv
│   ├── predictions.csv
│   ├── future_emission_scenarios.csv
│   └── feature_importance.csv
├── reports/
│   ├── data_quality_report.json
│   ├── data_quality_report.csv
│   └── model_evaluation_report.md
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── main.py                              # CLI entry point
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (optional, for version control)

### Virtual Environment Setup

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Jupyter Kernel Setup

```bash
python -m ipykernel install --user --name=vehicle-emission --display-name "Vehicle Emission"
```

## Usage

### 1. Command-Line Interface

Run commands from the project root directory:

```bash
# Inspect dataset
python main.py inspect

# Full EDA and data quality analysis
python main.py analyze

# Train and compare models
python main.py train

# Make predictions on new data
python main.py predict --input data/new_vehicles.csv

# Run future emission scenarios
python main.py forecast --config scenario_config.json

# Run full pipeline (inspect → analyze → train → predict → forecast)
python main.py all
```

#### Command Details

| Command | Description | Key Options |
|---------|-------------|-------------|
| `inspect` | Quick dataset overview | `--log-level` |
| `analyze` | EDA + quality checks + plots | `--log-level` |
| `train` | Train all models, save best | `--log-level` |
| `predict` | Predict on CSV or JSON input | `--input`, `--output` |
| `forecast` | Run scenarios from config | `--config`, `--output` |
| `all` | Full pipeline end-to-end | `--log-level` |

#### Prediction Input Format

CSV file with required feature columns (no target column needed):
```csv
Vehicle Type,Fuel Type,Engine Size,Age of Vehicle,Mileage,Speed,Acceleration,Road Type,Traffic Conditions,Temperature,Humidity,Wind Speed,Air Pressure
Car,Petrol,2.0,5,50000,60,2.5,Highway,Moderate,20,60,5,1013
Truck,Diesel,5.5,10,150000,55,1.8,City,Heavy,15,70,10,1015
```

### 2. Jupyter Notebook (VS Code)

1. Open VS Code in the project directory
2. Open `notebooks/vehicle_emission_analysis.ipynb`
3. Select the "Vehicle Emission" kernel (or your venv kernel)
4. Run all cells (Cell → Run All)

The notebook contains:
- Project overview and scientific limitations
- Dataset inspection and quality analysis
- Exploratory data analysis with plots
- Preprocessing pipeline
- Model training and comparison
- Cross-validation and model selection
- Final evaluation with diagnostic plots
- Model saving
- Example predictions
- Future scenario analysis
- Conclusions and limitations

### 3. Google Colab

1. Upload `notebooks/vehicle_emission_analysis.ipynb` to Colab
2. Upload `vehicle_emission_dataset.csv` when prompted (or mount Google Drive)
3. Install dependencies in first cell:
   ```python
   !pip install pandas numpy scikit-learn matplotlib seaborn joblib -q
   ```
4. Run all cells

The notebook includes a Colab-specific upload section for the dataset.

## Model Training Details

### Algorithms Compared

| Model | Type | Scaling Needed |
|-------|------|----------------|
| DummyRegressor | Baseline (mean) | No |
| Linear Regression | Linear | Yes |
| Ridge Regression | Regularized Linear | Yes |
| Random Forest | Ensemble (Bagging) | No |
| Extra Trees | Ensemble (Bagging) | No |
| HistGradientBoosting | Gradient Boosting | No |
| Gradient Boosting | Gradient Boosting | No |

### Training Protocol

1. **Train/Test Split**: 80/20 with fixed random state (42)
2. **Cross-Validation**: 5-fold CV on training set for model selection
3. **Final Evaluation**: Best model evaluated on held-out test set
4. **No Data Leakage**: Preprocessing fitted only on training data
5. **Reproducibility**: All random seeds fixed

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | Mean(\|y - ŷ\|) | Average absolute error (lower = better) |
| **RMSE** | √Mean((y - ŷ)²) | Penalizes large errors (lower = better) |
| **R²** | 1 - SS_res/SS_tot | Variance explained (1 = perfect, 0 = baseline, <0 = worse) |
| **MAPE** | Mean(\|(y - ŷ)/y\|) × 100 | Percentage error (only if all y > 0) |

**Important Notes:**
- R² is **not** a percentage accuracy score
- R² can be negative (model worse than horizontal line)
- Lower MAE/RMSE is better; higher R² is better
- MAPE is only meaningful when all target values are positive

## Scenario Assumptions

Future-year forecasting requires explicit assumptions. **The model does not learn temporal trends** - it predicts individual vehicle emissions from cross-sectional data.

### Required Assumptions

1. **Base Year**: Starting year (default: 2024)
2. **Base Fleet Size**: Number of vehicles represented (default: dataset size)
3. **Base Average Emission**: Mean predicted emission per vehicle
4. **Future Years**: Years to simulate
5. **Fleet Growth Rate**: Annual % change in fleet size
6. **Emission Change Rate**: Annual % change in avg emission per vehicle

### Default Scenarios

| Scenario | Fleet Growth | Emission Change | Description |
|----------|--------------|-----------------|-------------|
| Business as Usual | +2%/yr | 0%/yr | Current trends continue |
| Emission Reduction | +2%/yr | -3%/yr | Policy-driven reductions |
| High Growth | +5%/yr | +1%/yr | Rapid fleet expansion |

### Calculation Formulas

```
Fleet(t) = Fleet(base) × (1 + fleet_growth_rate)^(t - base_year)
AvgEmission(t) = AvgEmission(base) × (1 + emission_change_rate)^(t - base_year)
TotalEmission(t) = Fleet(t) × AvgEmission(t)
```

**⚠️ Critical Limitations:**
- These are **conditional scenarios**, not forecasts
- No historical time-series data in the dataset
- Target unit must be annual per-vehicle emissions for fleet totals to be meaningful
- If target is per-trip or per-km, multiply by annual distance/usage

### Customizing Scenarios

Create a `scenario_config.json`:

```json
{
  "base_year": 2024,
  "future_years": [2025, 2026, 2027, 2028, 2029, 2030],
  "base_fleet_size": 100000,
  "base_avg_emission_per_vehicle": 250.0,
  "target_unit": "g/vehicle/year",
  "scenarios": {
    "business_as_usual": {
      "fleet_growth_rate": 0.02,
      "emission_change_rate": 0.0,
      "description": "Current trends continue"
    },
    "my_custom_scenario": {
      "fleet_growth_rate": 0.03,
      "emission_change_rate": -0.05,
      "description": "Aggressive EV adoption"
    }
  }
}
```

Run with:
```bash
python main.py forecast --config scenario_config.json
```

## Generated Outputs

| File | Description |
|------|-------------|
| `outputs/model_comparison.csv` | All model metrics sorted by selection metric |
| `outputs/predictions.csv` | Predictions for new data |
| `outputs/future_emission_scenarios.csv` | Scenario results by year |
| `outputs/feature_importance.csv` | Permutation importance scores |
| `outputs/figures/*.png` | Publication-quality plots |
| `models/final_model.joblib` | Best model pipeline |
| `models/model_metadata.json` | Model configuration & metrics |
| `reports/data_quality_report.json` | Complete data quality analysis |
| `reports/model_evaluation_report.md` | Detailed evaluation report |

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: Dataset not found` | CSV not in data/ or root | Place CSV in correct location |
| `KeyError: 'CO2 Emissions'` | Target column name mismatch | Check column names in CSV |
| `ValueError: Missing required features` | Input missing columns | Ensure all feature columns present |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `Permission denied` on venv | PowerShell execution policy | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

## Scientific Assumptions and Limitations

1. **Cross-Sectional Data**: No temporal component; cannot validate time-series forecasting
2. **No Causal Claims**: Feature importance ≠ causal impact
3. **Extrapolation Risk**: Predictions unreliable outside training data range
4. **Unit Ambiguity**: Dataset units not verified; labeled as "dataset units"
5. **Scenario Dependency**: Future totals entirely dependent on user assumptions
6. **Fleet Aggregation**: Individual predictions ≠ fleet totals without usage data

## Reproducibility

- All random seeds fixed (42)
- Dependency versions pinned in `requirements.txt`
- Complete pipeline in `main.py` and notebook
- Model metadata includes training config and metrics

## Citation and Acknowledgments

If you use this project in research, please cite:

```bibtex
@software{vehicle_emission_prediction,
  title = {Vehicle Carbon Emission Prediction and Future-Year Scenario Forecasting},
  author = {Vehicle Emission Prediction Project},
  year = {2024},
  url = {https://github.com/yourusername/vehicle-emission-prediction}
}
```

**Dataset Source**: [Please provide dataset source, collection method, and license information]

## License

MIT License - see [LICENSE](LICENSE) for details.