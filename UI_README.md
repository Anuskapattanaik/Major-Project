# CarbonIQ Web UI — Vehicle Emission Predictor

A modern, responsive web interface for the Vehicle Carbon Emission Prediction project. Built with Streamlit and Plotly.

## Features

- **Predict CO₂ Emissions**: Enter vehicle details and get instant predictions from the trained ML model
- **Model Insights**: View model performance metrics, feature importance, and algorithm comparison
- **Prediction History**: Track predictions during your current session (downloadable as CSV)
- **Responsive Design**: Works on desktop and mobile browsers
- **Transparent**: All model metrics read from actual training results — no fabricated values

## Quick Start

### 1. Activate the virtual environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-ui.txt
```

If you haven't installed the base project dependencies yet, run the first command only.

### 3. Ensure the model is trained

If no model exists, train one first:

```bash
python main.py train
```

The model is saved to `models/final_model.joblib` with metadata at `models/model_metadata.json`.

### 4. Start the application

From the project root directory:

```bash
streamlit run app.py
```

### 5. Use the application

1. Open your browser to the URL shown (typically `http://localhost:8501`)
2. Select **Predict Emissions** tab
3. Enter vehicle details in the form
4. Click **Predict CO₂ Emissions**
5. View results, including comparison with dataset statistics
6. Download predictions as CSV

### 6. Navigate between pages

Use the tabs at the top of the application:

- **🎯 Predict Emissions** — Input form and results
- **📊 Model Insights** — Model metrics and feature importance
- **📋 History** — Session-level prediction history
- **ℹ️ About** — Project documentation and limitations

## Application Sections

### A. Prediction Form

The form is organized into four sections:

1. **Vehicle Information** — Vehicle Type, Fuel Type
2. **Engine and Usage** — Engine Size, Age, Mileage, Speed, Acceleration
3. **Driving Conditions** — Road Type, Traffic Conditions
4. **Environmental Conditions** — Temperature, Humidity, Wind Speed, Air Pressure

Each input has sensible default ranges derived from the dataset.

### B. Prediction Results

After submitting, results include:

- Predicted CO₂ emission value
- Dataset unit label (units not verified — see Limitations)
- Model name used
- MAE, RMSE, R², and MAPE metrics from the trained model
- Distribution histogram showing the prediction's position relative to observed values
- Input summary table
- CSV download button

### C. Model Insights

Displays:

- Selected model name and algorithm
- Target variable
- Train/test split configuration
- Test metrics (all read from saved metadata)
- Cross-validation results
- Model comparison chart (all algorithms)
- Feature importance (permutation importance, if available)
- Interactive bar charts using Plotly

### D. Prediction History

- Records all successful predictions in the current session
- Shows timestamp, vehicle type, fuel type, and predicted value
- Download full history as CSV
- Clear history button

## Environment Variables

None required. The application reads the model from `models/` and the dataset from `data/`.

## Troubleshooting

### Model not found

```
Error: Model Artifact Missing
```

**Fix**: Train the model first:
```bash
python main.py train
```

### Port already in use

```bash
streamlit run app.py --server.port=8502
```

### Streamlit not installed

```bash
pip install -r requirements-ui.txt
```

### Model prediction returns mean value for all inputs

This is expected if the selected model is a `DummyRegressor` (baseline model). The current trained model uses this as the best performer because the dataset has limited predictive signal for CO₂ emissions based on the available features. All other evaluated models (Linear, Ridge, Random Forest, etc.) performed worse on the held-out test set.

The UI uses the actual selected model from the project's model selection procedure. It does not use a substitute model.

### "ScriptRunContext" warning in terminal

This warning appears when importing `ui_prediction_service` outside Streamlit. It is harmless when running the full Streamlit app.

## Design

### Color Palette

- Primary: Deep forest green (`#1a5f40`)
- Secondary: Teal (`#00a3a3`)
- Accent: Sage green (`#2d8c5a`)
- Background: Light neutral (`#f8f9fa`)
- Cards: White with subtle shadows

### Responsive Layout

The layout uses Streamlit columns and adjusts automatically for desktop and mobile. Plotly charts include responsive rendering.

## Limitations

1. **Units not verified**: The dataset's units for emissions and other features are not documented. All predictions are displayed as "dataset units."
2. **No causal claims**: Feature importance reflects association, not causation.
3. **Limited model performance**: The current dataset yields R² ≈ 0 for the best model, meaning predictions may not capture meaningful patterns.
4. **Input ranges are illustrative**: Derived from a single dataset sample. Values outside training data ranges should be interpreted cautiously.

## File Structure

```
vehicle-carbon-emission-prediction/
├── app.py                      # Main Streamlit application
├── src/
│   ├── ui_prediction_service.py  # Model loading & prediction service
│   ├── config.py                 # Shared configuration
│   ├── data_loader.py            # Dataset loading
│   ├── preprocessing.py          # Preprocessing pipeline
│   ├── train.py                  # Model training
│   └── ...
├── requirements.txt            # Base dependencies
├── requirements-ui.txt         # UI dependencies (this file)
├── models/
│   ├── final_model.joblib      # Trained model
│   └── model_metadata.json     # Model metadata
└── outputs/                    # Generated analysis outputs
```

## Launch Command

```bash
streamlit run app.py
```

The application automatically:
- Loads the trained model from `models/final_model.joblib`
- Displays model performance metrics from `model_metadata.json`
- Uses the same preprocessing pipeline as training (no separate inference pipeline)
- Validates user inputs and provides helpful error messages