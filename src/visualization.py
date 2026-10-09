"""Visualization module for plots and charts."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from .config import config, get_figure_path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save figure with consistent settings."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Figure saved to {path}")


def plot_distribution(
    data: pd.Series,
    title: str,
    xlabel: str,
    save_path: Optional[Path] = None,
    bins: int = 50
) -> plt.Figure:
    """Plot distribution of a numerical variable."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data.dropna(), bins=bins, edgecolor='black', alpha=0.7, density=True)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    # Add KDE
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        ax.legend()
    except Exception:
        pass
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_categorical_counts(
    data: pd.Series,
    title: str,
    xlabel: str,
    save_path: Optional[Path] = None,
    top_n: int = 20
) -> plt.Figure:
    """Plot counts of categorical variable."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = data.value_counts().head(top_n)
    counts.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_box_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot boxplot of value by category."""
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby(category_col)[value_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=category_col, y=value_col, order=order, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(category_col, fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 12)
) -> plt.Figure:
    """Plot correlation heatmap for numerical columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[Path] = None,
    alpha: float = 0.5,
    color: str = 'steelblue'
) -> plt.Figure:
    """Plot scatter plot with optional trend line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=alpha, color=color, s=20)
    
    # Add trend line
    try:
        mask = x.notna() & y.notna()
        if mask.sum() > 1:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_range = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(x_range, p(x_range), "r--", lw=2, label=f'Trend (slope={z[0]:.3f})')
            ax.legend()
    except Exception:
        pass
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    unit: str = "dataset units"
) -> plt.Figure:
    """Plot actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue', label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Actual {unit}', fontsize=12)
    ax.set_ylabel(f'Predicted {unit}', fontsize=12)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_residuals(
    residuals: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    unit: str = "dataset units"
) -> plt.Figure:
    """Plot residual distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, density=True)
    axes[0].axvline(0, color='red', linestyle='--', lw=2)
    axes[0].set_title('Residual Distribution', fontsize=12)
    axes[0].set_xlabel(f'Residual ({unit})', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    
    # Q-Q plot
    try:
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal)', fontsize=12)
    except Exception:
        axes[1].text(0.5, 0.5, 'Q-Q plot requires scipy', ha='center', va='center')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_residuals_vs_predicted(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    unit: str = "dataset units"
) -> plt.Figure:
    """Plot residuals vs predicted values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Predicted Values ({unit})', fontsize=12)
    ax.set_ylabel(f'Residuals ({unit})', fontsize=12)
    
    # Add horizontal lines at ±2 std
    std = np.std(residuals)
    ax.axhline(2*std, color='orange', linestyle=':', lw=1, label=f'±2σ')
    ax.axhline(-2*std, color='orange', linestyle=':', lw=1)
    ax.legend()
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_feature_importance(
    importance: Dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot feature importance as horizontal bar chart."""
    # Sort and take top N
    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Permutation Importance', fontsize=12)
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "RMSE",
    title: str = "Model Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot model comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by metric
    df = results_df.sort_values(metric, ascending=True)
    
    bars = ax.barh(range(len(df)), df[metric], edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['model'])
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(metric, fontsize=12)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df[metric])):
        ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_future_scenarios(
    scenarios_df: pd.DataFrame,
    title: str = "Future Emission Scenarios",
    save_path: Optional[Path] = None,
    y_column: str = "total_emissions",
    y_label: str = "Total Annual Emissions (dataset units)"
) -> plt.Figure:
    """Plot future emission scenarios."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scenarios = scenarios_df['scenario'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        subset = scenarios_df[scenarios_df['scenario'] == scenario].sort_values('year')
        ax.plot(subset['year'], subset[y_column], 
                marker='o', linewidth=2.5, markersize=8,
                label=scenario.replace('_', ' ').title(),
                color=colors[i])
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add year labels
    years = sorted(scenarios_df['year'].unique())
    ax.set_xticks(years)
    
    if save_path:
        save_figure(fig, save_path)
    return fig


def generate_eda_plots(
    df: pd.DataFrame,
    target_col: str = "CO2 Emissions"
) -> List[Path]:
    """
    Generate all EDA plots for the dataset.
    
    Args:
        df: Dataset
        target_col: Target column name
        
    Returns:
        List of saved figure paths
    """
    saved_paths = []
    
    # 1. Distribution of CO2 emissions
    if target_col in df.columns:
        path = get_figure_path("eda_co2_distribution.png")
        plot_distribution(df[target_col], 
                         f"Distribution of {target_col}",
                         f"{target_col} (dataset units)",
                         save_path=path)
        saved_paths.append(path)
    
    # 2. Vehicle type counts
    if "Vehicle Type" in df.columns:
        path = get_figure_path("eda_vehicle_type_counts.png")
        plot_categorical_counts(df["Vehicle Type"],
                               "Vehicle Type Counts",
                               "Vehicle Type",
                               save_path=path)
        saved_paths.append(path)
    
    # 3. Fuel type counts
    if "Fuel Type" in df.columns:
        path = get_figure_path("eda_fuel_type_counts.png")
        plot_categorical_counts(df["Fuel Type"],
                               "Fuel Type Counts",
                               "Fuel Type",
                               save_path=path)
        saved_paths.append(path)
    
    # 4. CO2 by vehicle type
    if "Vehicle Type" in df.columns and target_col in df.columns:
        path = get_figure_path("eda_co2_by_vehicle_type.png")
        plot_box_by_category(df, "Vehicle Type", target_col,
                            f"{target_col} by Vehicle Type",
                            save_path=path)
        saved_paths.append(path)
    
    # 5. CO2 by fuel type
    if "Fuel Type" in df.columns and target_col in df.columns:
        path = get_figure_path("eda_co2_by_fuel_type.png")
        plot_box_by_category(df, "Fuel Type", target_col,
                            f"{target_col} by Fuel Type",
                            save_path=path)
        saved_paths.append(path)
    
    # 6. Correlation heatmap
    path = get_figure_path("eda_correlation_heatmap.png")
    plot_correlation_heatmap(df, "Correlation Heatmap (Numerical Features)", save_path=path)
    saved_paths.append(path)
    
    # 7. Mileage vs CO2
    if "Mileage" in df.columns and target_col in df.columns:
        path = get_figure_path("eda_mileage_vs_co2.png")
        plot_scatter(df["Mileage"], df[target_col],
                    f"{target_col} vs Mileage",
                    "Mileage (dataset units)",
                    f"{target_col} (dataset units)",
                    save_path=path)
        saved_paths.append(path)
    
    # 8. Engine size vs CO2
    if "Engine Size" in df.columns and target_col in df.columns:
        path = get_figure_path("eda_engine_size_vs_co2.png")
        plot_scatter(df["Engine Size"], df[target_col],
                    f"{target_col} vs Engine Size",
                    "Engine Size (dataset units)",
                    f"{target_col} (dataset units)",
                    save_path=path)
        saved_paths.append(path)
    
    # 9. Vehicle age vs CO2
    if "Age of Vehicle" in df.columns and target_col in df.columns:
        path = get_figure_path("eda_age_vs_co2.png")
        plot_scatter(df["Age of Vehicle"], df[target_col],
                    f"{target_col} vs Vehicle Age",
                    "Age of Vehicle (dataset units)",
                    f"{target_col} (dataset units)",
                    save_path=path)
        saved_paths.append(path)
    
    logger.info(f"Generated {len(saved_paths)} EDA plots")
    return saved_paths