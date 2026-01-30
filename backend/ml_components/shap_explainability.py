"""
SHAP Explainability Analysis for Cash Flow Prediction Model
Generates SHAP values and visualizations to explain model predictions

Usage:
    python shap_explainability.py
"""

import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Error: SHAP not installed. Install with: pip install shap")
    exit(1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_preparation import DataPreparator

# ============================================================================
# Load Model and Data
# ============================================================================
print("Loading model and preparing data for SHAP analysis...")

# Load saved model
try:
    with open('best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    model_name = model_data['model_name']
    model_type = model_data['model_type']
    
    print(f"Loaded model: {model_name} ({model_type})")
    
except FileNotFoundError:
    print("Error: best_model.pkl not found. Please run lstm_cashflow_prediction.py first.")
    exit(1)

# Load and prepare data
data_prep = DataPreparator(
    correlation_threshold=0.95,
    pca_variance_threshold=0.95
)
monthly_df = data_prep.load_and_aggregate("cashflow_prediction_1998_2025_v1.csv")

# Prepare features
X, y, feature_names, feature_df = data_prep.prepare_data(
    monthly_df,
    use_pca=False,
    feature_selection_method='mutual_info',
    n_features=50,
    apply_correlation_filter=True
)

# Use saved scaler or create new one
if model_data.get('feature_scaler') is not None:
    feature_scaler = model_data['feature_scaler']
else:
    feature_scaler = StandardScaler()
    feature_scaler.fit(X)

X_scaled = feature_scaler.transform(X)

# Train/test split (matching the training script)
train_idx, test_idx = train_test_split(
    np.arange(len(X_scaled)),
    test_size=0.2,
    random_state=42,
    shuffle=True
)

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"Data shape: Train={X_train.shape}, Test={X_test.shape}")
print(f"Number of features: {len(feature_names)}")

# ============================================================================
# SHAP Analysis
# ============================================================================

if model_type == 'prophet':
    print("\nWarning: SHAP is not directly supported for Prophet models.")
    print("SHAP works best with tree-based models (CatBoost, XGBoost, LightGBM, RandomForest).")
    exit(1)

print("\nGenerating SHAP values...")
print("This may take a few minutes...")

# Create SHAP explainer
# For tree models, use TreeExplainer (faster and exact)
if model_name in ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
    try:
        explainer = shap.TreeExplainer(model)
        print("Using TreeExplainer (fast and exact for tree models)")
    except:
        # Fallback to KernelExplainer if TreeExplainer fails
        print("TreeExplainer failed, using KernelExplainer (slower but more general)")
        explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Use sample for speed
else:
    # For other models, use KernelExplainer
    print("Using KernelExplainer (slower but works for all models)")
    explainer = shap.KernelExplainer(model.predict, X_train[:100])  # Use sample for speed

# Calculate SHAP values for test set (use subset for speed)
n_samples = min(100, len(X_test))  # Limit to 100 samples for faster computation
shap_values = explainer.shap_values(X_test[:n_samples])

# Convert to numpy array if needed
if isinstance(shap_values, list):
    shap_values = np.array(shap_values)

print(f"Generated SHAP values for {n_samples} samples")

# ============================================================================
# Create Visualizations
# ============================================================================

print("\nCreating SHAP visualizations...")

# Create output directory
import os
os.makedirs('shap_plots', exist_ok=True)

# 1. Feature Importance DataFrame
print("1. Creating feature importance table...")
mean_abs_shap = np.abs(shap_values).mean(0)
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Rank': range(1, len(feature_names) + 1)
})
feature_importance_df = feature_importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
feature_importance_df['Rank'] = range(1, len(feature_importance_df) + 1)

# Save to Excel
excel_file = 'shap_plots/feature_importance.xlsx'
feature_importance_df.to_excel(excel_file, index=False, sheet_name='Feature Importance')
print(f"   Saved: {excel_file}")

# 2. Plotly Bar Plot (Mean Absolute SHAP Values)
print("2. Creating Plotly bar plot (mean absolute SHAP values)...")

# Get top 20 features for better visualization
top_n = min(20, len(feature_importance_df))
top_features_df = feature_importance_df.head(top_n).sort_values('Mean_Abs_SHAP', ascending=True)

# Create Plotly bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=top_features_df['Mean_Abs_SHAP'],
    y=top_features_df['Feature'],
    orientation='h',
    marker=dict(
        color=top_features_df['Mean_Abs_SHAP'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Mean |SHAP|")
    ),
    text=[f"{val:.2f}" for val in top_features_df['Mean_Abs_SHAP']],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>' +
                  'Mean |SHAP|: %{x:.4f}<br>' +
                  'Rank: %{customdata}<extra></extra>',
    customdata=top_features_df['Rank']
))

fig.update_layout(
    title=dict(
        text=f'SHAP Feature Importance (Top {top_n} Features)',
        x=0.5,
        xanchor='center',
        font=dict(size=18)
    ),
    xaxis=dict(
        title=dict(text='Mean Absolute SHAP Value', font=dict(size=14))
    ),
    yaxis=dict(
        title=dict(text='Feature', font=dict(size=14))
    ),
    height=max(600, top_n * 25),
    margin=dict(l=200, r=50, t=80, b=50),
    template='plotly_white',
    hovermode='closest'
)

# Save as HTML
html_file = 'shap_plots/shap_bar_plot.html'
fig.write_html(html_file)
print(f"   Saved: {html_file}")

# Also save as static image (PNG) for quick viewing
try:
    fig.write_image('shap_plots/shap_bar_plot.png', width=1200, height=max(600, top_n * 25))
    print(f"   Saved: shap_plots/shap_bar_plot.png")
except Exception as e:
    print(f"   Note: PNG export requires kaleido. Install with: pip install kaleido")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*70)
print("SHAP Analysis Summary")
print("="*70)
print(f"Model: {model_name}")
print(f"Number of samples analyzed: {n_samples}")
print(f"Number of features: {len(feature_names)}")
print(f"\nTop 10 Most Important Features:")
print("-"*70)
for idx, row in feature_importance_df.head(10).iterrows():
    print(f"{row['Rank']:2d}. {row['Feature']:<40} SHAP: {row['Mean_Abs_SHAP']:>12.2f}")

print("\n" + "="*70)
print("All visualizations saved to: shap_plots/")
print("="*70)
