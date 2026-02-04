"""
Multi-Model Cash Outflow Prediction System
Predicts monthly cash_outflow_usd from cashflow_prediction_1998_2025_v1.csv

Models included:
- Prophet (time series forecasting)
- XGBoost (gradient boosting)
- LightGBM (gradient boosting)
- CatBoost (gradient boosting)
- GradientBoostingRegressor (sklearn)
- RandomForestRegressor (sklearn)

All models use hyperparameter tuning to find optimal parameters.
The script automatically selects the best performing model based on test set metrics.

Usage:
    python lstm_cashflow_prediction.py           # Interactive: choose 1-12 months
    python lstm_cashflow_prediction.py 3         # Predict next 3 months
    python lstm_cashflow_prediction.py 6         # Predict next 6 months
    python lstm_cashflow_prediction.py 12        # Predict next 12 months
"""

import sys
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import pickle
import warnings
import io
warnings.filterwarnings('ignore')

# Fix encoding issues on Windows console
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for stdout
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Import data preparation module
from data_preparation import DataPreparator

# Boosting model imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Get prediction horizon from command line or user input
if len(sys.argv) > 1:
    try:
        FORECAST_MONTHS = int(sys.argv[1])
        if FORECAST_MONTHS < 1 or FORECAST_MONTHS > 12:
            print(f"Warning: {FORECAST_MONTHS} is out of range (1-12). Using 1 month instead.")
            FORECAST_MONTHS = 1
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number. Using 1 month instead.")
        FORECAST_MONTHS = 1
else:
    # Interactive prompt for user to choose prediction window
    print("\n" + "="*70)
    print("Cash Flow Prediction - Select Prediction Window")
    print("="*70)
    print("How many months ahead would you like to predict?")
    print("Enter a number between 1 and 12:")
    
    while True:
        try:
            user_input = input("Number of months (1-12): ").strip()
            if not user_input:
                FORECAST_MONTHS = 1
                print("No input provided. Using default: 1 month")
                break
            
            FORECAST_MONTHS = int(user_input)
            if 1 <= FORECAST_MONTHS <= 12:
                print(f"\nSelected: {FORECAST_MONTHS} month(s) prediction window")
                break
            else:
                print(f"Error: {FORECAST_MONTHS} is out of range. Please enter a number between 1 and 12.")
        except ValueError:
            print("Error: Please enter a valid number between 1 and 12.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
    
    print("="*70 + "\n")

# ============================================================================
# 1. Initialize Data Preparator
# ============================================================================

data_prep = DataPreparator(
    correlation_threshold=0.95,  # Remove features with correlation > 0.95
    pca_variance_threshold=0.95  # PCA variance threshold (if used)
)

# ============================================================================
# 2. Load and Aggregate Data
# ============================================================================
monthly_df = data_prep.load_and_aggregate("cashflow_prediction_1998_2025_v1.csv")

# ============================================================================
# 3. Prepare Data with Advanced Feature Engineering
# ============================================================================
# Prepare data with feature selection (mutual_info) and correlation filtering
X, y, feature_names, feature_df = data_prep.prepare_data(
    monthly_df,
    use_pca=False,  # Set to True to use PCA instead of feature selection
    feature_selection_method='mutual_info',  # Options: 'mutual_info', 'f_regression', 'rf_importance', 'rfe'
    n_features=50,  # Number of features to select (None = auto)
    apply_correlation_filter=True
)

target_original = y.copy()

# Prepare Prophet data
prophet_df, selected_regressors = data_prep.prepare_prophet_data(monthly_df)
target_scaler = data_prep.prophet_scaler
regressor_scalers = data_prep.regressor_scalers

# Standardize features for boosting models
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
feature_cols = feature_names

# Train/test split
train_idx, test_idx = train_test_split(
    np.arange(len(X_scaled)), 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

# For Prophet
prophet_train = prophet_df.iloc[train_idx].sort_values('ds').reset_index(drop=True)
prophet_test = prophet_df.iloc[test_idx].sort_values('ds').reset_index(drop=True)

# For boosting models
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# ============================================================================
# 4. Train All Models with Hyperparameter Tuning
# ============================================================================

models = {}
model_results = {}

# 1. Prophet
try:
    best_prophet_r2 = -np.inf
    best_prophet_model = None
    best_prophet_params = None
    best_prophet_regressors = []
    
    param_combinations = [
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 0.1},
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1},
    ]
    
    regressor_sets = [
        list(selected_regressors.keys()),
        ['cash_inflow', 'interest_rate', 'is_tax_month', 'is_quarter_end', 'month_sin', 'month_cos'],
    ]
    
    for params in param_combinations:
        for regressor_set in regressor_sets:
            try:
                test_model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode=params['seasonality_mode'],
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    interval_width=0.95,
                )
                
                for reg in regressor_set:
                    test_model.add_regressor(reg, standardize=False)
                
                test_model.fit(prophet_train)
                
                test_cols = ['ds'] + regressor_set
                test_forecast = test_model.predict(prophet_test[test_cols])
                test_r2 = r2_score(prophet_test['y'].values, test_forecast['yhat'].values)
                
                if test_r2 > best_prophet_r2:
                    best_prophet_r2 = test_r2
                    best_prophet_model = test_model
                    best_prophet_params = params
                    best_prophet_regressors = regressor_set
            except:
                continue
    
    if best_prophet_model:
        models['Prophet'] = {
            'model': best_prophet_model,
            'type': 'prophet',
            'regressors': best_prophet_regressors,
            'params': best_prophet_params
        }
except Exception as e:
    pass

# 2. XGBoost with Hyperparameter Tuning
if XGBOOST_AVAILABLE:
    try:
        # Define parameter grid for XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Base model
        xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Use RandomizedSearchCV for faster tuning
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_param_grid, 
            n_iter=20,  # Try 20 random combinations
            scoring='r2',
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        xgb_search.fit(X_train, y_train)
        xgb_model = xgb_search.best_estimator_
        
        models['XGBoost'] = {
            'model': xgb_model, 
            'type': 'boosting',
            'best_params': xgb_search.best_params_
        }
    except Exception as e:
        pass

# 3. LightGBM with Hyperparameter Tuning
if LIGHTGBM_AVAILABLE:
    try:
        # Define parameter grid for LightGBM
        lgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        # Base model
        lgb_base = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        # Use RandomizedSearchCV
        lgb_search = RandomizedSearchCV(
            lgb_base, lgb_param_grid,
            n_iter=20,
            scoring='r2',
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        lgb_search.fit(X_train, y_train)
        lgb_model = lgb_search.best_estimator_
        
        models['LightGBM'] = {
            'model': lgb_model,
            'type': 'boosting',
            'best_params': lgb_search.best_params_
        }
    except Exception as e:
        pass

# 4. CatBoost with Hyperparameter Tuning
if CATBOOST_AVAILABLE:
    try:
        # Define parameter grid for CatBoost
        cat_param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        
        # Base model
        cat_base = CatBoostRegressor(random_seed=42, verbose=False)
        
        # Use RandomizedSearchCV
        cat_search = RandomizedSearchCV(
            cat_base, cat_param_grid,
            n_iter=15,
            scoring='r2',
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        cat_search.fit(X_train, y_train)
        cat_model = cat_search.best_estimator_
        
        models['CatBoost'] = {
            'model': cat_model,
            'type': 'boosting',
            'best_params': cat_search.best_params_
        }
    except Exception as e:
        pass

# 5. GradientBoostingRegressor with Hyperparameter Tuning
try:
    # Define parameter grid for GradientBoosting
    gbr_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9]
    }
    
    # Base model
    gbr_base = GradientBoostingRegressor(random_state=42)
    
    # Use RandomizedSearchCV
    gbr_search = RandomizedSearchCV(
        gbr_base, gbr_param_grid,
        n_iter=20,
        scoring='r2',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    gbr_search.fit(X_train, y_train)
    gbr_model = gbr_search.best_estimator_
    
    models['GradientBoosting'] = {
        'model': gbr_model,
        'type': 'boosting',
        'best_params': gbr_search.best_params_
    }
except Exception as e:
    pass

# 6. RandomForestRegressor with Hyperparameter Tuning
try:
    # Define parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Base model
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Use RandomizedSearchCV
    rf_search = RandomizedSearchCV(
        rf_base, rf_param_grid,
        n_iter=30,  # More iterations for RF as it's faster
        scoring='r2',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
    
    models['RandomForest'] = {
        'model': rf_model,
        'type': 'boosting',  # Treat as boosting for prediction logic
        'best_params': rf_search.best_params_
    }
except Exception as e:
    pass

# ============================================================================
# 5. Evaluate All Models (Train vs Test Comparison)
# ============================================================================

for model_name, model_info in models.items():
    model = model_info['model']
    model_type = model_info['type']
    
    try:
        if model_type == 'prophet':
            # Prophet predictions on test set
            test_cols = ['ds'] + model_info['regressors']
            test_pred_scaled = model.predict(prophet_test[test_cols])['yhat'].values
            test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
            test_actual = target_scaler.inverse_transform(prophet_test['y'].values.reshape(-1, 1)).flatten()
            
            # Prophet predictions on train set
            train_cols = ['ds'] + model_info['regressors']
            train_pred_scaled = model.predict(prophet_train[train_cols])['yhat'].values
            train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
            train_actual = target_scaler.inverse_transform(prophet_train['y'].values.reshape(-1, 1)).flatten()
        else:
            # Boosting model predictions on test set
            test_pred = model.predict(X_test)
            test_actual = y_test
            
            # Boosting model predictions on train set
            train_pred = model.predict(X_train)
            train_actual = y_train
        
        # Calculate test metrics
        test_mae = mean_absolute_error(test_actual, test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        test_r2 = r2_score(test_actual, test_pred)
        test_mape = np.mean(np.abs((test_actual - test_pred) / test_actual)) * 100
        
        # Calculate train metrics
        train_mae = mean_absolute_error(train_actual, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
        train_r2 = r2_score(train_actual, train_pred)
        train_mape = np.mean(np.abs((train_actual - train_pred) / train_actual)) * 100
        
        # Calculate performance gap (overfitting indicator)
        r2_gap = train_r2 - test_r2
        mape_gap = test_mape - train_mape
        
        model_results[model_name] = {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'train_mape': train_mape,
            'r2_gap': r2_gap,
            'mape_gap': mape_gap,
            'predictions': test_pred,
            'actual': test_actual
        }
        
        # Determine performance status
        if r2_gap > 0.15:  # Large gap indicates overfitting
            status = "⚠️  OVERFITTING"
        elif r2_gap < 0.05 and test_r2 > 0.4:  # Small gap and good performance
            status = "✓ GOOD GENERALIZATION"
        elif test_r2 < 0.3:  # Low performance on both
            status = "⚠️  UNDERFITTING"
        else:
            status = "→ ACCEPTABLE"
        
    except Exception as e:
        continue

# Select best model based on test R² score
if model_results:
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
    best_model_info = models[best_model_name]
    best_results = model_results[best_model_name]
    
    # Determine best model status
    if best_results['r2_gap'] > 0.15:
        best_status = "⚠️  OVERFITTING - Large gap between train and test performance"
    elif best_results['r2_gap'] < 0.05 and best_results['test_r2'] > 0.4:
        best_status = "✓ GOOD GENERALIZATION - Model performs similarly on train and test"
    elif best_results['test_r2'] < 0.3:
        best_status = "⚠️  UNDERFITTING - Low performance on both train and test"
    else:
        best_status = "→ ACCEPTABLE - Model performance is acceptable"
    
else:
    sys.exit(1)

# ============================================================================
# 6. Future Predictions with Best Model
# ============================================================================

# Retrain best model on full dataset

if best_model_info['type'] == 'prophet':
    # Retrain Prophet on full data
    full_prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=best_model_info['params']['seasonality_mode'],
        changepoint_prior_scale=best_model_info['params']['changepoint_prior_scale'],
        seasonality_prior_scale=best_model_info['params']['seasonality_prior_scale'],
        interval_width=0.95,
    )
    for reg in best_model_info['regressors']:
        full_prophet_model.add_regressor(reg, standardize=False)
    full_prophet_model.fit(prophet_df)
    
    # Create future dataframe
    last_date = prophet_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_MONTHS, freq='MS')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add month features
    future_df['month'] = future_df['ds'].dt.month
    future_df['is_tax_month'] = future_df['month'].isin([1, 4, 6, 9]).astype(int)
    future_df['is_quarter_end'] = future_df['month'].isin([3, 6, 9, 12]).astype(int)
    future_df['is_year_end'] = (future_df['month'] == 12).astype(int)
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    
    # Add regressors (use last 12 months average)
    lookback = min(12, len(monthly_df))
    for reg in best_model_info['regressors']:
        if reg in ['is_tax_month', 'is_quarter_end', 'is_year_end', 'month_sin', 'month_cos']:
            continue
        elif reg in regressor_scalers:
            original_col = {
                'cash_inflow': 'cash_inflow_usd',
                'interest_rate': 'interest_rate_pct',
                'inflation': 'inflation_pct',
                'economic_sentiment': 'economic_sentiment_score',
            }.get(reg, reg)
            if original_col in monthly_df.columns:
                original_avg = monthly_df[original_col].tail(lookback).mean()
                scaled_value = regressor_scalers[reg].transform([[original_avg]])[0, 0]
                future_df[reg] = scaled_value
            else:
                future_df[reg] = 0
        else:
            future_df[reg] = 0
    
    # Make predictions
    future_forecast = full_prophet_model.predict(future_df)
    predictions_scaled = future_forecast['yhat'].values
    predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    predictions_lower = target_scaler.inverse_transform(future_forecast['yhat_lower'].values.reshape(-1, 1)).flatten()
    predictions_upper = target_scaler.inverse_transform(future_forecast['yhat_upper'].values.reshape(-1, 1)).flatten()
    
else:
    # Retrain boosting model on full data with best hyperparameters
    best_params = best_model_info.get('best_params', {})
    
    if best_model_name == 'XGBoost':
        full_model = xgb.XGBRegressor(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('learning_rate', 0.05),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            random_state=42, n_jobs=-1
        )
    elif best_model_name == 'LightGBM':
        full_model = lgb.LGBMRegressor(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('learning_rate', 0.05),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            random_state=42, n_jobs=-1, verbose=-1
        )
    elif best_model_name == 'CatBoost':
        full_model = CatBoostRegressor(
            iterations=best_params.get('iterations', 200),
            depth=best_params.get('depth', 6),
            learning_rate=best_params.get('learning_rate', 0.05),
            random_seed=42, verbose=False
        )
    elif best_model_name == 'GradientBoosting':
        full_model = GradientBoostingRegressor(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 6),
            learning_rate=best_params.get('learning_rate', 0.05),
            subsample=best_params.get('subsample', 0.8),
            random_state=42
        )
    elif best_model_name == 'RandomForest':
        full_model = RandomForestRegressor(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', None),
            min_samples_split=best_params.get('min_samples_split', 2),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            max_features=best_params.get('max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1
        )
    
    full_model.fit(X_scaled, y)
    
    # Create future features
    last_date = feature_df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=FORECAST_MONTHS, freq='MS')
    
    # Use last 12 months for averaging
    lookback = min(12, len(feature_df))
    
    # Build future feature matrix using selected feature names
    future_features = []
    for date in future_dates:
        feat_row = []
        for col in feature_cols:
            # Temporal features
            if col == 'month':
                feat_row.append(date.month)
            elif col == 'quarter':
                feat_row.append((date.month - 1) // 3 + 1)
            elif col == 'year':
                feat_row.append(date.year)
            elif col == 'day_of_year':
                feat_row.append(date.timetuple().tm_yday)
            elif col == 'week_of_year':
                feat_row.append(date.isocalendar()[1])
            # Tax and fiscal indicators
            elif col == 'is_tax_month':
                feat_row.append(1 if date.month in [1, 4, 6, 9] else 0)
            elif col == 'is_q1_tax':
                feat_row.append(1 if date.month == 4 else 0)
            elif col == 'is_q2_tax':
                feat_row.append(1 if date.month == 6 else 0)
            elif col == 'is_q3_tax':
                feat_row.append(1 if date.month == 9 else 0)
            elif col == 'is_q4_tax':
                feat_row.append(1 if date.month == 1 else 0)
            elif col == 'is_april':
                feat_row.append(1 if date.month == 4 else 0)
            elif col == 'is_october':
                feat_row.append(1 if date.month == 10 else 0)
            elif col == 'is_quarter_end':
                feat_row.append(1 if date.month in [3, 6, 9, 12] else 0)
            elif col == 'is_year_end':
                feat_row.append(1 if date.month == 12 else 0)
            elif col == 'is_year_start':
                feat_row.append(1 if date.month == 1 else 0)
            # Cyclical features
            elif col == 'month_sin':
                feat_row.append(np.sin(2 * np.pi * date.month / 12))
            elif col == 'month_cos':
                feat_row.append(np.cos(2 * np.pi * date.month / 12))
            elif col == 'quarter_sin':
                feat_row.append(np.sin(2 * np.pi * ((date.month - 1) // 3 + 1) / 4))
            elif col == 'quarter_cos':
                feat_row.append(np.cos(2 * np.pi * ((date.month - 1) // 3 + 1) / 4))
            elif col == 'year_sin' or col == 'year_cos':
                # Use normalized year
                year_min = feature_df['year'].min() if 'year' in feature_df.columns else date.year
                year_max = feature_df['year'].max() if 'year' in feature_df.columns else date.year
                year_norm = (date.year - year_min) / (year_max - year_min + 1)
                if col == 'year_sin':
                    feat_row.append(np.sin(2 * np.pi * year_norm))
                else:
                    feat_row.append(np.cos(2 * np.pi * year_norm))
            # Lag, rolling, or other derived features - use last known value
            elif 'lag' in col or 'rolling' in col or 'ema' in col or 'ratio' in col or 'mom' in col or 'yoy' in col:
                if col in feature_df.columns:
                    feat_row.append(feature_df[col].iloc[-1])
                else:
                    feat_row.append(0)
            # Outlier flags
            elif col == 'is_outlier_combined' or col == 'is_outlier_iqr' or col == 'is_outlier_zscore':
                feat_row.append(0)  # Assume normal months
            # Trend
            elif col == 'trend':
                feat_row.append(len(feature_df) + len(future_features))
            # Interaction features - use last known values
            elif 'interaction' in col:
                if col in feature_df.columns:
                    feat_row.append(feature_df[col].iloc[-1])
                else:
                    feat_row.append(0)
            # All other features - use average of last 12 months
            else:
                if col in feature_df.columns:
                    feat_row.append(feature_df[col].tail(lookback).mean())
                else:
                    feat_row.append(0)
        future_features.append(feat_row)
    
    future_X = np.array(future_features)
    future_X_scaled = feature_scaler.transform(future_X)
    
    # Make predictions
    predictions = full_model.predict(future_X_scaled)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    # Simple confidence intervals (using test set std)
    pred_std = np.std(best_results['predictions'] - best_results['actual'])
    predictions_lower = predictions - 1.96 * pred_std
    predictions_upper = predictions + 1.96 * pred_std
    predictions_lower = np.maximum(predictions_lower, 0)
    predictions_upper = np.maximum(predictions_upper, 0)

total_predicted = 0
predictions_list = []

for i in range(FORECAST_MONTHS):
    month_str = future_dates[i].strftime('%Y-%m')
    pred = predictions[i]
    lower = predictions_lower[i]
    upper = predictions_upper[i]
    
    total_predicted += pred
    
    predictions_list.append({
        'month': month_str,
        'predicted_cash_outflow': pred,
        'lower_95': lower,
        'upper_95': upper
    })

# Compare to historical
historical_avg = target_original.mean()

# ============================================================================
# Display Prediction Results
# ============================================================================
print("\n" + "="*80)
print(f"PREDICTION RESULTS - {FORECAST_MONTHS} MONTH(S) FORECAST")
print("="*80)
print(f"{'Month':<12} {'Predicted':>18} {'Lower 95% CI':>18} {'Upper 95% CI':>18}")
print("-"*80)

for pred_data in predictions_list:
    month = pred_data['month']
    pred = pred_data['predicted_cash_outflow']
    lower = pred_data['lower_95']
    upper = pred_data['upper_95']
    print(f"{month:<12} ${pred:>17,.0f} ${lower:>17,.0f} ${upper:>17,.0f}")

print("-"*80)
print(f"{'TOTAL':<12} ${total_predicted:>17,.0f}")
print(f"{'AVERAGE':<12} ${total_predicted/FORECAST_MONTHS:>17,.0f}")
print(f"\nHistorical Monthly Average: ${historical_avg:,.0f}")
print(f"Predicted vs Historical: {((total_predicted/FORECAST_MONTHS)/historical_avg - 1)*100:+.1f}%")
print("="*80)

# ============================================================================
# 7. Save Results
# ============================================================================

# Save predictions
predictions_df = pd.DataFrame(predictions_list)
predictions_df.to_csv('future_predictions.csv', index=False)

# Save model comparison results with train/test comparison
comparison_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Train_R2': [model_results[m]['train_r2'] for m in model_results.keys()],
    'Test_R2': [model_results[m]['test_r2'] for m in model_results.keys()],
    'R2_Gap': [model_results[m]['r2_gap'] for m in model_results.keys()],
    'Train_MAPE': [model_results[m]['train_mape'] for m in model_results.keys()],
    'Test_MAPE': [model_results[m]['test_mape'] for m in model_results.keys()],
    'MAPE_Gap': [model_results[m]['mape_gap'] for m in model_results.keys()],
    'Train_MAE': [model_results[m]['train_mae'] for m in model_results.keys()],
    'Test_MAE': [model_results[m]['test_mae'] for m in model_results.keys()],
    'Train_RMSE': [model_results[m]['train_rmse'] for m in model_results.keys()],
    'Test_RMSE': [model_results[m]['test_rmse'] for m in model_results.keys()],
})
comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
comparison_df.to_csv('model_comparison.csv', index=False)

# Save detailed train/test comparison
train_test_comparison = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Train_R2': [model_results[m]['train_r2'] for m in model_results.keys()],
    'Test_R2': [model_results[m]['test_r2'] for m in model_results.keys()],
    'R2_Gap': [model_results[m]['r2_gap'] for m in model_results.keys()],
    'R2_Gap_Pct': [abs(model_results[m]['r2_gap']) / max(model_results[m]['train_r2'], 0.001) * 100 for m in model_results.keys()],
    'Train_MAPE': [model_results[m]['train_mape'] for m in model_results.keys()],
    'Test_MAPE': [model_results[m]['test_mape'] for m in model_results.keys()],
    'MAPE_Gap': [model_results[m]['mape_gap'] for m in model_results.keys()],
    'Generalization_Status': [
        "GOOD" if abs(model_results[m]['r2_gap']) < 0.1 and model_results[m]['test_r2'] > 0.4
        else "OVERFITTING" if model_results[m]['r2_gap'] > 0.15
        else "UNDERFITTING" if model_results[m]['test_r2'] < 0.3
        else "ACCEPTABLE"
        for m in model_results.keys()
    ]
})
train_test_comparison = train_test_comparison.sort_values('Test_R2', ascending=False)
train_test_comparison.to_csv('train_test_comparison.csv', index=False)

# Save predicted vs actual comparison for ALL months (full dataset visualization)
try:
    # Get all dates from feature_df
    all_dates = feature_df['date'].values
    all_actual = y.copy()  # All actual values (unscaled for boosting models)
    
    # Make predictions on the ENTIRE dataset using the full trained model
    if best_model_info['type'] == 'prophet':
        # For Prophet, predict on full prophet_df
        prophet_full_pred = full_prophet_model.predict(prophet_df[['ds'] + best_model_info['regressors']])
        all_predicted = target_scaler.inverse_transform(prophet_full_pred['yhat'].values.reshape(-1, 1)).flatten()
        all_actual = target_scaler.inverse_transform(prophet_df['y'].values.reshape(-1, 1)).flatten()
    else:
        # For boosting models, predict on full scaled dataset
        all_predicted = full_model.predict(X_scaled)
    
    # Create comparison dataframe with ALL months
    pred_vs_actual_df = pd.DataFrame({
        'date': all_dates,
        'actual': all_actual,
        'predicted': all_predicted
    })
    
    # Sort by date
    pred_vs_actual_df['date'] = pd.to_datetime(pred_vs_actual_df['date'])
    pred_vs_actual_df = pred_vs_actual_df.sort_values('date').reset_index(drop=True)
    
    # Format date as month string
    pred_vs_actual_df['month'] = pred_vs_actual_df['date'].dt.strftime('%Y-%m')
    
    # Calculate error metrics per row
    pred_vs_actual_df['error'] = pred_vs_actual_df['actual'] - pred_vs_actual_df['predicted']
    pred_vs_actual_df['percent_error'] = (pred_vs_actual_df['error'] / pred_vs_actual_df['actual'] * 100).round(2)
    pred_vs_actual_df['abs_percent_error'] = pred_vs_actual_df['percent_error'].abs()
    
    # Save to CSV - all months
    pred_vs_actual_df[['month', 'actual', 'predicted', 'error', 'percent_error']].to_csv('predicted_vs_actual.csv', index=False)
    print(f"\nSaved predicted vs actual comparison to 'predicted_vs_actual.csv' ({len(pred_vs_actual_df)} months - FULL DATASET)")
    print(f"  Date range: {pred_vs_actual_df['month'].min()} to {pred_vs_actual_df['month'].max()}")
    print(f"  Overall MAPE: {pred_vs_actual_df['abs_percent_error'].mean():.2f}%")
except Exception as e:
    print(f"Warning: Could not save predicted vs actual comparison: {e}")

# Save best model
import json
with open('best_model_info.json', 'w') as f:
    json.dump({
        'best_model': best_model_name,
        'model_type': best_model_info['type'],
        'test_r2': float(best_results['test_r2']),
        'test_mae': float(best_results['test_mae']),
        'test_rmse': float(best_results['test_rmse']),
        'test_mape': float(best_results['test_mape']),
        'train_r2': float(best_results['train_r2']),
        'train_mae': float(best_results['train_mae']),
        'train_rmse': float(best_results['train_rmse']),
        'train_mape': float(best_results['train_mape']),
        'r2_gap': float(best_results['r2_gap']),
        'mape_gap': float(best_results['mape_gap']),
        'generalization_status': "GOOD" if abs(best_results['r2_gap']) < 0.1 and best_results['test_r2'] > 0.4
                                 else "OVERFITTING" if best_results['r2_gap'] > 0.15
                                 else "UNDERFITTING" if best_results['test_r2'] < 0.3
                                 else "ACCEPTABLE",
        'forecast_months': FORECAST_MONTHS,
        'training_months': len(feature_df),
        'last_training_date': feature_df['date'].max().strftime('%Y-%m-%d'),
        'best_params': best_model_info.get('best_params', None),
    }, f, indent=2)

# Save model and scalers
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': full_model if best_model_info['type'] != 'prophet' else full_prophet_model,
        'model_name': best_model_name,
        'model_type': best_model_info['type'],
        'feature_scaler': feature_scaler if best_model_info['type'] != 'prophet' else None,
        'target_scaler': target_scaler if best_model_info['type'] == 'prophet' else None,
        'regressor_scalers': regressor_scalers if best_model_info['type'] == 'prophet' else None,
        'feature_cols': feature_cols if best_model_info['type'] != 'prophet' else None,
        'prophet_regressors': best_model_info.get('regressors', []) if best_model_info['type'] == 'prophet' else None,
        'best_params': best_model_info.get('best_params', None),
    }, f)

# ============================================================================
# 8. Summary
# ============================================================================

print("\nAll Models Performance (Test Set):")
print("-" * 70)
print(f"{'Model':<20} {'Test R²':>10} {'Train R²':>10} {'R² Gap':>10} {'Test MAPE':>12} {'Status':>15}")
print("-" * 70)
for model_name in comparison_df['Model']:
    test_r2 = comparison_df[comparison_df['Model'] == model_name]['Test_R2'].values[0]
    train_r2 = comparison_df[comparison_df['Model'] == model_name]['Train_R2'].values[0]
    r2_gap = comparison_df[comparison_df['Model'] == model_name]['R2_Gap'].values[0]
    test_mape = comparison_df[comparison_df['Model'] == model_name]['Test_MAPE'].values[0]
    
    # Determine status
    if abs(r2_gap) < 0.1 and test_r2 > 0.4:
        status = "GOOD"
    elif r2_gap > 0.15:
        status = "OVERFIT"
    elif test_r2 < 0.3:
        status = "UNDERFIT"
    else:
        status = "OK"
    
    marker = "*" if model_name == best_model_name else " "
    print(f"{marker} {model_name:<20} {test_r2:>10.4f} {train_r2:>10.4f} {r2_gap:>+10.4f} {test_mape:>12.2f}% {status:>15}")
