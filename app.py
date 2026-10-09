"""Main Streamlit application for CarbonIQ Vehicle Emission Predictor.

This module provides a modern, responsive web UI for predicting vehicle CO2 emissions
using the trained machine learning model from the Vehicle Carbon Emission Prediction project.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ui_prediction_service import get_prediction_service, PredictionService


# Page configuration
st.set_page_config(
    page_title="CarbonIQ - Vehicle Emission Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for environmental color palette
st.markdown("""
    <style>
    :root {
        --primary-green: #1a5f40;
        --secondary-green: #2d8c5a;
        --teal: #00a3a3;
        --light-teal: #e0f7f7;
        --light-green: #e6f4ea;
        --white: #ffffff;
        --off-white: #f8f9fa;
        --light-gray: #f1f3f4;
        --medium-gray: #9aa0a6;
        --dark-gray: #4d5157;
        --border-radius: 12px;
        --shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .main { background-color: var(--off-white); }
    
    .header {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-green) 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        color: var(--white);
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }
    
    .header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .header .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .card {
        background: var(--white);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--light-gray);
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: linear-gradient(135deg, var(--light-teal) 0%, var(--light-green) 100%);
        border-radius: var(--border-radius);
        padding: 2rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--light-gray);
        margin: 1.5rem 0;
    }
    
    .result-card .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-green);
        text-align: center;
        margin: 1rem 0;
    }
    
    .section-title {
        color: var(--primary-green);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--teal);
        padding-bottom: 0.5rem;
    }
    
    .form-label {
        font-weight: 600;
        color: var(--dark-gray);
        margin-bottom: 0.25rem;
        font-size: 0.9rem;
    }
    
    .stButton > button {
        background: var(--secondary-green);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: var(--primary-green);
        transform: translateY(-1px);
    }
    
    .secondary-button {
        background: var(--medium-gray);
    }
    
    .secondary-button:hover {
        background: var(--dark-gray);
    }
    
    .metric-card {
        background: var(--white);
        border-radius: var(--border-radius);
        padding: 1rem;
        text-align: center;
        box-shadow: var(--shadow);
        border: 1px solid var(--light-gray);
    }
    
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-green);
    }
    
    .metric-card .label {
        font-size: 0.85rem;
        color: var(--dark-gray);
        margin-top: 0.25rem;
    }
    
    .nav-button {
        background: transparent;
        border: 1px solid var(--teal);
        color: var(--teal);
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .nav-button:hover {
        background: var(--light-teal);
    }
    
    .nav-button-active {
        background: var(--teal);
        color: white;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        color: var(--medium-gray);
        font-size: 0.85rem;
        margin-top: 2rem;
        padding: 1rem;
    }
    
    .tab-container {
        background: var(--white);
        border-radius: var(--border-radius);
        padding: 1rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--light-gray);
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


def add_to_history(prediction_data: dict):
    """Add a prediction to session history."""
    st.session_state.prediction_history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **prediction_data
    })


def clear_history():
    """Clear prediction history."""
    st.session_state.prediction_history = []


def export_history():
    """Export prediction history as DataFrame."""
    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        # Flatten input features
        if 'input_features' in df.columns:
            input_df = pd.json_normalize(df['input_features'])
            df = pd.concat([df.drop('input_features', axis=1), input_df], axis=1)
        return df
    return pd.DataFrame()


def load_service():
    """Load prediction service, handling errors gracefully."""
    try:
        return get_prediction_service()
    except FileNotFoundError as e:
        st.error(f"## Model Artifact Missing\n\n{e}")
        st.markdown("""
        ### Setup Required
        
        The trained model was not found. Please train the model first:
        ```
        python main.py train
        ```
        
        After training completes, refresh this page to use the prediction service.
        """)
        return None
    except Exception as e:
        st.error(f"## Model Loading Error\n\nFailed to load the trained model:\n\n```\n{e}\n```\n\nPlease ensure the model was trained successfully and try again.")
        return None


def render_header():
    """Render the application header."""
    st.markdown("""
        <div class="header">
            <h1>CarbonIQ</h1>
            <div class="subtitle">Vehicle Emission Prediction &mdash; Estimate vehicle CO₂ emissions using machine learning</div>
        </div>
    """, unsafe_allow_html=True)


def render_prediction_form(service: PredictionService):
    """Render the prediction input form."""
    with st.form(key="prediction_form"):
        st.markdown('<div class="section-title">Enter Vehicle Details</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Vehicle Information")
            vehicle_type = st.selectbox("Vehicle Type", options=service.get_categorical_options()['Vehicle Type'], index=0)
            fuel_type = st.selectbox("Fuel Type", options=service.get_categorical_options()['Fuel Type'], index=0)
        
        with col2:
            st.markdown("### Engine and Usage")
            engine_size = st.number_input("Engine Size", min_value=0.1, max_value=6.0, value=3.0, step=0.1)
            age = st.number_input("Age of Vehicle (years)", min_value=0, max_value=29, value=10, step=1)
            mileage = st.number_input("Mileage (km)", min_value=56, max_value=299863, value=50000, step=1000)
            speed = st.slider("Speed (dataset units)", min_value=0.0, max_value=120.0, value=60.0, step=0.5)
            acceleration = st.slider("Acceleration (dataset units)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        
        st.markdown("### Driving Conditions")
        road_type = st.selectbox("Road Type", options=service.get_categorical_options()['Road Type'], index=0)
        traffic = st.selectbox("Traffic Conditions", options=service.get_categorical_options()['Traffic Conditions'], index=0)
        
        st.markdown("### Environmental Conditions")
        col3, col4 = st.columns([1, 1])
        with col3:
            temperature = st.number_input("Temperature (dataset units)", min_value=-10.0, max_value=40.0, value=20.0, step=1.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            wind_speed = st.number_input("Wind Speed (dataset units)", min_value=0.0, max_value=20.0, value=10.0, step=0.5)
        with col4:
            air_pressure = st.number_input("Air Pressure (dataset units)", min_value=950.0, max_value=1050.0, value=1000.0, step=1.0)
        
        # Build input data
        input_data = {
            "Vehicle Type": vehicle_type,
            "Fuel Type": fuel_type,
            "Engine Size": float(engine_size),
            "Age of Vehicle": int(age),
            "Mileage": int(mileage),
            "Speed": float(speed),
            "Acceleration": float(acceleration),
            "Road Type": road_type,
            "Traffic Conditions": traffic,
            "Temperature": float(temperature),
            "Humidity": float(humidity),
            "Wind Speed": float(wind_speed),
            "Air Pressure": float(air_pressure)
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_left, col_submit = st.columns([1, 2])
        
        with col_left:
            st.form_submit_button("Reset Inputs", width='stretch')
        
        with col_submit:
            submitted = st.form_submit_button("Predict CO₂ Emissions", type="primary", width='stretch')
        
        if submitted:
            with st.spinner("Making prediction..."):
                try:
                    prediction, prediction_info = service.predict(input_data)
                    add_to_history({
                        'predicted_co2': prediction,
                        'model_name': service.model_name,
                        'input_features': input_data
                    })
                    st.session_state['last_prediction'] = {
                        'prediction': prediction,
                        'info': prediction_info,
                        'input_data': input_data,
                        'service': service
                    }
                except ValueError as e:
                    st.error(f"Input validation error: {str(e)}")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    
    # Add a separate reset button outside the form
    if st.button("Reset Inputs", key="reset_btn", width='stretch'):
        st.rerun()


def render_results(service: PredictionService):
    """Render prediction results."""
    if 'last_prediction' not in st.session_state:
        return
    
    pred_data = st.session_state['last_prediction']
    prediction = pred_data['prediction']
    prediction_info = pred_data['info']
    input_data = pred_data['input_data']
    
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    
    # Get target stats from dataset
    try:
        df = pd.read_csv(str(Path(__file__).parent / "vehicle_emission_dataset.csv"))
        target_col = "CO2 Emissions"
        avg_co2 = df[target_col].mean()
        median_co2 = df[target_col].median()
        min_co2 = df[target_col].min()
        max_co2 = df[target_col].max()
        std_co2 = df[target_col].std()
    except Exception:
        avg_co2 = None
        median_co2 = None
        min_co2 = None
        max_co2 = None
        std_co2 = None
    
    # Result card
    st.markdown(f"""
        <div class="result-card">
            <h2 style="text-align: center; color: #1a5f40; margin-top: 0;">Predicted CO₂ Emissions</h2>
            <div class="prediction-value">{prediction:.2f} <span style="font-size: 1rem; color: #4d5157;">dataset units</span></div>
            <p style="text-align: center; color: #4d5157; font-size: 0.9rem;">
                Model: {service.model_name} &mdash; Target: {service.target_column}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model info
    metrics = service.test_metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics.get('MAE', 'N/A'):.2f}</div>
            <div class="label">MAE (Mean Absolute Error)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics.get('RMSE', 'N/A'):.2f}</div>
            <div class="label">RMSE</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics.get('R2', 'N/A'):.4f}</div>
            <div class="label">R² (Coefficient of Determination)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics.get('MAPE', 'N/A'):.1f}%</div>
            <div class="label">MAPE</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribution comparison
    if avg_co2 is not None:
        st.markdown('<div class="section-title">Prediction Distribution</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Add distribution histogram
        fig.add_trace(go.Histogram(
            x=df[target_col],
            nbinsx=50,
            name='Observed Emissions',
            marker_color='#2d8c5a',
            opacity=0.6
        ))
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=[prediction, prediction],
            y=[0, df[target_col].value_counts().max() * 0.8],
            mode='lines',
            name=f'Prediction: {prediction:.2f}',
            line=dict(color='#c0392b', width=3, dash='dash')
        ))
        
        # Add mean and median lines
        if avg_co2 is not None:
            fig.add_vline(x=avg_co2, line_dash="dot", line_color="#2980b9",
                         annotation_text=f"Mean: {avg_co2:.1f}")
        if median_co2 is not None:
            fig.add_vline(x=median_co2, line_dash="dot", line_color="#8e44ad",
                         annotation_text=f"Median: {median_co2:.1f}")
        
        fig.update_layout(
            title='CO₂ Emissions Distribution and Your Prediction',
            xaxis_title='CO₂ Emissions (dataset units)',
            yaxis_title='Count',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#4d5157'),
            height=400
        )
        st.plotly_chart(fig, width='stretch')
        
        if prediction > max_co2 or prediction < min_co2:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <strong>Prediction outside training range.</strong> The predicted value falls outside the range
                of observed values in the training dataset. Use this result with caution.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="warning-box">
            ℹ️ <strong>Note on model performance.</strong> The current model ({service.model_name}) has an R² of {metrics.get('R2', 'N/A'):.4f}.
            Since R² is near 0 or negative, predictions may not capture the underlying patterns well.
            This prediction should be treated as illustrative, not definitive.
        </div>
        """, unsafe_allow_html=True)
    
    # Input summary
    st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)
    input_col1, input_col2 = st.columns(2)
    with input_col1:
        st.write("**Vehicle & Engine**")
        for k, v in input_data.items():
            if k in ["Vehicle Type", "Fuel Type", "Engine Size", "Age of Vehicle", "Road Type"]:
                st.write(f"  **{k}**: {v}")
    with input_col2:
        st.write("**Driving & Environment**")
        for k, v in input_data.items():
            if k not in ["Vehicle Type", "Fuel Type", "Engine Size", "Age of Vehicle", "Road Type"]:
                st.write(f"  **{k}**: {v}")
    
    # Download prediction
    csv_data = pd.DataFrame([{
        **input_data,
        'predicted_co2_emissions': prediction,
        'model_name': service.model_name,
        'timestamp': datetime.now().isoformat()
    }])
    csv_string = csv_data.to_csv(index=False)
    
    st.download_button(
        label="Download Prediction as CSV",
        data=csv_string,
        file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width='stretch'
    )


def render_model_insights(service: PredictionService):
    """Render model insights page."""
    st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Model Details")
        st.write(f"**Model Name:** {service.model_name}")
        st.write(f"**Algorithm:** {service.model_name}")
        st.write(f"**Target Variable:** {service.target_column}")
        st.write(f"**Random State:** {service.train_config.get('random_state', 'N/A')}")
        st.write(f"**Test Size:** {service.train_config.get('test_size', 'N/A')}")
        st.write(f"**Selection Metric:** {service.train_config.get('selection_metric', 'N/A')}")
        st.write(f"**Model Saved:** {service.metadata.get('saved_at', 'N/A')}")
        
        st.write("---")
        st.write(f"**Number of Features:** {service.metadata.get('n_features_in', 'N/A')}")
        st.write(f"**Numerical Features:** {service.metadata.get('n_numerical', 'N/A')}")
        st.write(f"**Categorical Features:** {service.metadata.get('n_categorical', 'N/A')}")
    
    with col2:
        st.markdown("### Test Set Performance")
        metrics = service.test_metrics
        st.write(f"**MAE:** {metrics.get('MAE', 'N/A'):.4f} (dataset units)")
        st.write(f"**RMSE:** {metrics.get('RMSE', 'N/A'):.4f} (dataset units)")
        st.write(f"**R²:** {metrics.get('R2', 'N/A'):.4f}")
        mape = metrics.get('MAPE', 'N/A')
        if mape is not None and not (isinstance(mape, float) and np.isnan(mape)):
            st.write(f"**MAPE:** {mape:.2f}%")
        else:
            st.write("**MAPE:** Not available (target contains non-positive values)")
        
        st.write("---")
        st.markdown("### Training Configuration")
        st.write(f"**Cross-validation folds:** {service.train_config.get('cv_folds', 'N/A')}")
        
        # CV results
        cv_rmse_mean = metrics.get('cv_rmse_mean')
        cv_rmse_std = metrics.get('cv_rmse_std')
        if cv_rmse_mean is not None and isinstance(cv_rmse_mean, (int, float)):
            st.write(f"**CV RMSE (mean):** {cv_rmse_mean:.4f}")
        else:
            st.write("**CV RMSE (mean):** Not available")
        if cv_rmse_std is not None and isinstance(cv_rmse_std, (int, float)):
            st.write(f"**CV RMSE (std):** {cv_rmse_std:.4f}")
        else:
            st.write("**CV RMSE (std):** Not available")
    
    # Load full model comparison
    model_comp_path = Path(__file__).parent / "outputs" / "model_comparison.csv"
    if model_comp_path.exists():
        st.markdown("### All Models Comparison")
        df_comp = pd.read_csv(model_comp_path)
        st.dataframe(df_comp, width='stretch')
        
        # Bar chart
        if 'RMSE' in df_comp.columns and 'model' in df_comp.columns:
            fig = px.bar(
                df_comp.sort_values('RMSE'),
                x='RMSE',
                y='model',
                orientation='h',
                color='RMSE',
                color_continuous_scale=['#2d8c5a', '#1a5f40'],
                title='Model Comparison (RMSE, lower = better)'
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=400)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("Model comparison results not available. Run `python main.py train` to generate them.")
    
    # Feature importance
    fi_path = Path(__file__).parent / "outputs" / "feature_importance.csv"
    if fi_path.exists():
        st.markdown("### Feature Importance (Permutation Importance)")
        df_fi = pd.read_csv(fi_path)
        if not df_fi.empty:
            fig = px.bar(
                df_fi.head(15).sort_values('importance'),
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale=['#00a3a3', '#1a5f40'],
                title='Top 15 Most Important Features'
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=500)
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("""
            <div class="warning-box">
                ℹ️ Feature importance reflects predictive association, <strong>not</strong> causal impact.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Feature importance not available. Run `python main.py train` to generate it.")
    
    # Metrics explanation
    with st.expander("Metric Explanations"):
        st.markdown("""
        - **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual values,
          measured in the same units as the target.
        - **RMSE (Root Mean Squared Error):** Similar to MAE but penalizes larger errors more heavily.
          Lower values indicate better model fit.
        - **R² (Coefficient of Determination):** Indicates how much of the variance in the target variable
          is explained by the model. A value of 1 means perfect prediction, 0 means the model performs no
          better than predicting the mean, and negative values indicate the model performs worse than the mean.
        - **MAPE (Mean Absolute Percentage Error):** The average percentage error between predicted and
          actual values.
        """)


def render_about():
    """Render about page."""
    st.markdown('<div class="section-title">About the Project</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Project Overview
        
        **CarbonIQ** is a machine learning system for predicting vehicle CO₂ emissions and analyzing
        future-year emission scenarios. The project uses an ensemble of regression algorithms to predict
        emissions based on vehicle characteristics, driving conditions, and environmental parameters.
        
        ### Problem Statement
        
        Transportation is a major contributor to global carbon emissions. Understanding the factors that
        influence vehicle emissions and being able to predict emissions for new vehicles is crucial for
        policy-making, fleet management, and environmental planning.
        
        ### Project Objectives
        
        1. Analyze and clean vehicle emission datasets
        2. Predict CO₂ emissions for individual vehicles using multiple regression algorithms
        3. Compare models with proper cross-validation and held-out test evaluation
        4. Forecast future annual emissions under transparent, configurable scenarios
        5. Provide a web-based interface for easy prediction
        
        ### Machine Learning Workflow
        
        1. **Data Loading & Quality**: Load CSV, check for missing values, duplicates, outliers
        2. **Preprocessing**: Imputation, one-hot encoding, feature scaling (where appropriate)
        3. **Model Training**: 7 regression algorithms compared (Dummy, Linear, Ridge, Random Forest,
           Extra Trees, HistGradientBoosting, GradientBoosting)
        4. **Model Selection**: Cross-validation on training set, final evaluation on held-out test set
        5. **Prediction**: User inputs are processed through the exact same pipeline as training data
        """)
    
    with col2:
        st.markdown("""
        ### Dataset Summary
        
        - **Records:** 10,000 vehicles
        - **Features:** 13 input features
        - **Target:** CO2 Emissions
        - **Vehicle Types:** Car, Truck, Bus, Motorcycle
        - **Fuel Types:** Petrol, Diesel, Electric, Hybrid
        - **Road Types:** City, Highway, Rural
        - **Traffic Conditions:** Free flow, Moderate, Heavy
        
        ### Key Limitations
        
        - The dataset is **cross-sectional** — no time/temporal information
        - Future-year scenarios require **explicit user assumptions**
        - **No causal claims** can be made from feature importance
        - **No units verification** — values displayed as "dataset units"
        - Model performance varies; R² near 0 indicates limited predictive power for this dataset
        """)
    
    st.markdown("---")
    
    # Best model info
    service = load_service()
    if service and service.is_loaded():
        st.markdown("### Current Model")
        st.write(f"- **Model:** {service.model_name}")
        st.write(f"- **Target:** {service.target_column}")
        st.write(f"- **Test RMSE:** {service.test_metrics.get('RMSE', 0):.2f}")
        st.write(f"- **Test R²:** {service.test_metrics.get('R2', 0):.4f}")


def render_history():
    """Render prediction history page."""
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    
    if not st.session_state.prediction_history:
        st.info("No predictions have been made yet. Go to **Predict Emissions** to make your first prediction.")
        return
    
    df_history = pd.DataFrame(st.session_state.prediction_history)
    
    # Display recent predictions
    st.markdown("### Recent Predictions")
    
    display_data = []
    for _, row in df_history.iterrows():
        record = {
            'Timestamp': row['timestamp'],
            'Vehicle Type': row['input_features'].get('Vehicle Type', ''),
            'Fuel Type': row['input_features'].get('Fuel Type', ''),
            'Engine Size': row['input_features'].get('Engine Size', ''),
            'Predicted CO₂': f"{row['predicted_co2']:.2f}",
            'Model': row.get('model_name', '')
        }
        display_data.append(record)
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, width='stretch')
    
    # Export and clear buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        df_export = export_history()
        if not df_export.empty:
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download History as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
    
    with col2:
        if st.button("Clear History", key="clear_history", width='stretch'):
            clear_history()
            st.success("History cleared!")
            st.rerun()
    
    st.markdown(f"\n**Total predictions in this session:** {len(st.session_state.prediction_history)}")


def main():
    """Main application entry point."""
    render_header()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Predict Emissions", 
        "📊 Model Insights", 
        "📋 History", 
        "ℹ️ About"
    ])
    
    with tab1:
        service = load_service()
        if service and service.is_loaded():
            render_prediction_form(service)
            if 'last_prediction' in st.session_state:
                render_results(service)
    with tab2:
        service = load_service()
        if service and service.is_loaded():
            render_model_insights(service)
        else:
            st.warning("Model not loaded. Please train the model first.")
    with tab3:
        render_history()
    with tab4:
        render_about()
    
    # Footer
    st.markdown("""
        <div class="footer">
            CarbonIQ — Vehicle Carbon Emission Prediction | 
            Powered by Machine Learning | 
            <a href="https://github.com" style="color: #2d8c5a;">Source on GitHub</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()