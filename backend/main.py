"""
FastAPI Backend Server for Cash Flow Prediction System
Replaces Streamlit with a RESTful API for testing and integration

Usage:
    uvicorn main:app --reload --port 5000
    Or: python main.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add ml_components to path so we can import directly
ml_components_path = Path(__file__).parent / 'ml_components'
sys.path.insert(0, str(ml_components_path))

# Import ML components (now in path, so direct import works)
from data_preparation import DataPreparator
from data_analysis import CashFlowAnalyzer
from app import CashFlowMLSystem

# Initialize FastAPI app
app = FastAPI(
    title="Cash Flow Prediction API",
    description="RESTful API for Cash Flow Prediction using Machine Learning",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths - All files are now in the backend directory
BACKEND_DIR = Path(__file__).parent
ML_COMPONENTS_DIR = BACKEND_DIR / 'ml_components'
DATA_FILE = BACKEND_DIR / 'cashflow_prediction_1998_2025_v1.csv'
MODEL_FILE = BACKEND_DIR / 'best_model.pkl'
MODEL_INFO_FILE = BACKEND_DIR / 'best_model_info.json'
PREDICTIONS_FILE = BACKEND_DIR / 'future_predictions.csv'
PREDICTION_HISTORY_FILE = BACKEND_DIR / 'prediction_history.csv'

# Initialize ML system with absolute path
ml_system = CashFlowMLSystem(str(DATA_FILE.resolve()))

# Request/Response Models
class PredictionRequest(BaseModel):
    forecastMonths: int = 1

class TrainRequest(BaseModel):
    forecastMonths: int = 1

class PredictionResponse(BaseModel):
    success: bool
    data: Optional[List[dict]] = None
    error: Optional[str] = None
    message: Optional[str] = None

# Helper functions
def load_model():
    """Load saved model"""
    if not MODEL_FILE.exists():
        return None, None, None
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        return model_data.get('model'), model_data.get('model_name'), model_data.get('model_type')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def read_csv_file(file_path):
    """Read CSV file and return as list of dicts"""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {str(e)}")

# Check if frontend build exists (for production)
FRONTEND_BUILD_DIR = BACKEND_DIR.parent / 'frontend_build'

# API Routes

@app.get("/")
async def root():
    """Root endpoint - serves frontend in production, API info in development"""
    if FRONTEND_BUILD_DIR.exists():
        index_file = FRONTEND_BUILD_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return {
        "message": "Cash Flow Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Cash Flow Prediction API is running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/model/info")
async def get_model_info():
    """Get model information"""
    try:
        if not MODEL_INFO_FILE.exists():
            return {
                "success": False,
                "message": "Model not found. Please train a model first."
            }
        
        with open(MODEL_INFO_FILE, 'r') as f:
            model_info = json.load(f)
        
        return {
            "success": True,
            "data": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Generate predictions"""
    try:
        forecast_months = request.forecastMonths
        
        if forecast_months < 1 or forecast_months > 12:
            raise HTTPException(
                status_code=400,
                detail="forecastMonths must be between 1 and 12"
            )
        
        # Check if model exists
        if not MODEL_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Model not found. Please train a model first."
            )
        
        # Use ML system to make predictions
        success = ml_system.make_predictions(
            forecast_months=forecast_months,
            model_path=str(MODEL_FILE)
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate predictions"
            )
        
        # Read predictions
        if PREDICTIONS_FILE.exists():
            predictions = read_csv_file(PREDICTIONS_FILE)
            
            # Save predictions to history file for future comparison
            try:
                predictions_df = pd.read_csv(PREDICTIONS_FILE)
                predictions_df['predicted_on'] = datetime.now().strftime('%Y-%m-%d')
                predictions_df['prediction_id'] = datetime.now().strftime('%Y%m%d%H%M%S')
                
                # Append to history file or create new one
                if PREDICTION_HISTORY_FILE.exists():
                    history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
                    # Remove old predictions for the same months (keep latest only)
                    existing_months = predictions_df['month'].tolist()
                    history_df = history_df[~history_df['month'].isin(existing_months)]
                    history_df = pd.concat([history_df, predictions_df], ignore_index=True)
                else:
                    history_df = predictions_df
                
                history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
            except Exception as save_err:
                print(f"Warning: Could not save prediction history: {save_err}")
            
            return {
                "success": True,
                "data": predictions
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Predictions file not generated"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions")
async def get_predictions():
    """Get existing predictions"""
    try:
        if not PREDICTIONS_FILE.exists():
            return {
                "success": False,
                "message": "No predictions found. Please generate predictions first."
            }
        
        predictions = read_csv_file(PREDICTIONS_FILE)
        return {
            "success": True,
            "data": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/historical")
async def get_historical_data():
    """Get historical data"""
    try:
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )

        # Read and aggregate data (keep this endpoint lightweight and reliable)
        # NOTE: We intentionally do NOT call DataPreparator.load_and_aggregate() here because it
        # can auto-update files and print console output, which can cause runtime issues on Windows.
        df = pd.read_csv(DATA_FILE)

        # Parse date/month into a datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif 'month' in df.columns:
            # `month` in this dataset is numeric month (1-12); it is not a year-month string.
            # Fallback: try to build from `date`-like columns if present, otherwise error.
            raise HTTPException(
                status_code=500,
                detail="Historical endpoint requires a 'date' column; 'month' alone is not enough to build a timeline."
            )
        else:
            raise HTTPException(status_code=500, detail="No 'date' column found in data")

        df = df.dropna(subset=['date']).copy()

        # Ensure numeric columns are numeric
        numeric_cols = [
            'cash_inflow_usd',
            'cash_outflow_usd',
            'vendor_payment_usd',
            'salary_payment_usd',
            'rent_usd',
            'operational_expense_usd'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                # If a column is missing, create it as zeros so frontend stays consistent
                df[col] = 0.0

        # Aggregate to month
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_df = df.groupby('year_month', as_index=False).agg({
            'cash_inflow_usd': 'sum',
            'cash_outflow_usd': 'sum',
            'vendor_payment_usd': 'sum',
            'salary_payment_usd': 'sum',
            'rent_usd': 'sum',
            'operational_expense_usd': 'sum',
        })
        monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
        monthly_df = monthly_df.sort_values('date').reset_index(drop=True)

        monthly_data = []
        for _, row in monthly_df.iterrows():
            monthly_data.append({
                "month": row['date'].strftime('%Y-%m'),
                "cash_inflow": float(row['cash_inflow_usd']),
                "cash_outflow": float(row['cash_outflow_usd']),
                "vendor_payment": float(row['vendor_payment_usd']),
                "salary_payment": float(row['salary_payment_usd']),
                "rent": float(row['rent_usd']),
                "operational_expense": float(row['operational_expense_usd'])
            })
        
        # Get last 12 months
        last_12_months = monthly_data[-12:] if len(monthly_data) > 12 else monthly_data
        
        return {
            "success": True,
            "data": last_12_months,
            "allData": monthly_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/summary")
async def get_summary():
    """Get summary statistics"""
    try:
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )
        
        # Read data
        df = pd.read_csv(DATA_FILE)
        
        # Calculate totals
        total_inflow = float(df['cash_inflow_usd'].sum())
        total_outflow = float(df['cash_outflow_usd'].sum())
        total_vendor = float(df['vendor_payment_usd'].sum())
        total_salary = float(df['salary_payment_usd'].sum())
        total_rent = float(df['rent_usd'].sum())
        total_operational = float(df['operational_expense_usd'].sum())
        
        # Calculate income breakdown by source
        income_by_source = {}
        if 'cash_source' in df.columns:
            # Group by cash_source and sum cash_inflow_usd
            income_groups = df.groupby('cash_source')['cash_inflow_usd'].sum()
            income_by_source = {
                'customer_payment': float(income_groups.get('customer_payment', 0)),
                'loan_disbursement': float(income_groups.get('loan_disbursement', 0)),
                'investment': float(income_groups.get('investment', 0)),
                'asset_sale': float(income_groups.get('asset_sale', 0))
            }
        
        # Get last month data
        last_month = df.tail(30)
        last_month_inflow = float(last_month['cash_inflow_usd'].sum())
        last_month_outflow = float(last_month['cash_outflow_usd'].sum())
        
        # Get previous month for comparison
        if len(df) > 60:
            previous_month = df.iloc[-60:-30]
            prev_month_inflow = float(previous_month['cash_inflow_usd'].sum())
            prev_month_outflow = float(previous_month['cash_outflow_usd'].sum())
        else:
            prev_month_inflow = last_month_inflow
            prev_month_outflow = last_month_outflow
        
        # Calculate changes
        inflow_change = ((last_month_inflow - prev_month_inflow) / prev_month_inflow * 100) if prev_month_inflow > 0 else 0
        outflow_change = ((last_month_outflow - prev_month_outflow) / prev_month_outflow * 100) if prev_month_outflow > 0 else 0
        
        # Calculate year-over-year comparison for total balance
        current_year = None
        previous_year = None
        balance_yoy_change = 0
        
        # Convert date column if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # Get current year and previous year
            df['year'] = df['date'].dt.year
            current_year = int(df['year'].max())
            previous_year = current_year - 1
            
            # Calculate total balance for current year
            current_year_data = df[df['year'] == current_year]
            current_year_balance = float((current_year_data['cash_inflow_usd'].sum() - current_year_data['cash_outflow_usd'].sum()))
            
            # Calculate total balance for previous year
            previous_year_data = df[df['year'] == previous_year]
            if len(previous_year_data) > 0:
                previous_year_balance = float((previous_year_data['cash_inflow_usd'].sum() - previous_year_data['cash_outflow_usd'].sum()))
                balance_yoy_change = ((current_year_balance - previous_year_balance) / abs(previous_year_balance) * 100) if previous_year_balance != 0 else 0
            else:
                balance_yoy_change = 0
        elif 'month' in df.columns:
            # Use month column to extract year
            df['year'] = pd.to_datetime(df['month'] + '-01').dt.year
            current_year = int(df['year'].max())
            previous_year = current_year - 1
            
            # Calculate total balance for current year
            current_year_data = df[df['year'] == current_year]
            current_year_balance = float((current_year_data['cash_inflow_usd'].sum() - current_year_data['cash_outflow_usd'].sum()))
            
            # Calculate total balance for previous year
            previous_year_data = df[df['year'] == previous_year]
            if len(previous_year_data) > 0:
                previous_year_balance = float((previous_year_data['cash_inflow_usd'].sum() - previous_year_data['cash_outflow_usd'].sum()))
                balance_yoy_change = ((current_year_balance - previous_year_balance) / abs(previous_year_balance) * 100) if previous_year_balance != 0 else 0
            else:
                balance_yoy_change = 0
        else:
            # Fallback: use last 12 months vs previous 12 months
            if len(df) > 365:
                current_year_data = df.tail(365)
                previous_year_data = df.iloc[-730:-365]
                current_year_balance = float((current_year_data['cash_inflow_usd'].sum() - current_year_data['cash_outflow_usd'].sum()))
                previous_year_balance = float((previous_year_data['cash_inflow_usd'].sum() - previous_year_data['cash_outflow_usd'].sum()))
                balance_yoy_change = ((current_year_balance - previous_year_balance) / abs(previous_year_balance) * 100) if previous_year_balance != 0 else 0
                # Estimate years from data
                current_year = datetime.now().year
                previous_year = current_year - 1
        
        return {
            "success": True,
            "data": {
                "totalBalance": total_inflow - total_outflow,
                "totalInflow": total_inflow,
                "totalOutflow": total_outflow,
                "lastMonthInflow": last_month_inflow,
                "lastMonthOutflow": last_month_outflow,
                "inflowChange": round(inflow_change, 2),
                "outflowChange": round(outflow_change, 2),
                "balanceYearOverYearChange": round(balance_yoy_change, 2),
                "currentYear": current_year,
                "previousYear": previous_year,
                "expenseBreakdown": {
                    "vendor": total_vendor,
                    "salary": total_salary,
                    "rent": total_rent,
                    "operational": total_operational
                },
                "incomeBreakdown": income_by_source
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/year-over-year")
async def get_year_over_year(month: int = 0):
    """Get year-over-year comparison data for income, expense, and net cashflow
    
    Args:
        month: Optional month filter (1-12). If 0, returns full year data.
    """
    try:
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )
        
        # Read data
        df = pd.read_csv(DATA_FILE)
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month_num'] = df['date'].dt.month
        elif 'month' in df.columns:
            df['date'] = pd.to_datetime(df['month'] + '-01')
            df['year'] = df['date'].dt.year
            df['month_num'] = df['date'].dt.month
        else:
            raise HTTPException(
                status_code=500,
                detail="No date or month column found in data"
            )
        
        # Filter by month if specified
        if month > 0 and month <= 12:
            df = df[df['month_num'] == month]
        
        # Aggregate by year
        yearly_data = df.groupby('year').agg({
            'cash_inflow_usd': 'sum',
            'cash_outflow_usd': 'sum'
        }).reset_index()
        
        # Calculate net cashflow
        yearly_data['net_cashflow'] = yearly_data['cash_inflow_usd'] - yearly_data['cash_outflow_usd']
        
        # Format data for frontend
        year_comparison = []
        for _, row in yearly_data.iterrows():
            year_comparison.append({
                'year': int(row['year']),
                'income': float(row['cash_inflow_usd']),
                'expense': float(row['cash_outflow_usd']),
                'netCashflow': float(row['net_cashflow'])
            })
        
        # Sort by year and get last 10 years
        year_comparison.sort(key=lambda x: x['year'])
        last_10_years = year_comparison[-10:] if len(year_comparison) > 10 else year_comparison
        
        # Get month name for display
        month_name = None
        if month > 0 and month <= 12:
            import calendar
            month_name = calendar.month_name[month]
        
        return {
            "success": True,
            "data": last_10_years,
            "selectedMonth": month,
            "monthName": month_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train model (runs in background)"""
    try:
        forecast_months = request.forecastMonths
        
        if forecast_months < 1 or forecast_months > 12:
            raise HTTPException(
                status_code=400,
                detail="forecastMonths must be between 1 and 12"
            )
        
        # Train model using ML system
        success = ml_system.train_models(forecast_months=forecast_months)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Model training failed"
            )
        
        # Read updated model info
        if MODEL_INFO_FILE.exists():
            with open(MODEL_INFO_FILE, 'r') as f:
                model_info = json.load(f)
            
            return {
                "success": True,
                "data": model_info,
                "message": "Model trained successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Model training completed but info file not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File status endpoint
@app.get("/api/files/status")
async def get_files_status():
    """Get status of required files"""
    files = {
        "best_model.pkl": MODEL_FILE.exists(),
        "best_model_info.json": MODEL_INFO_FILE.exists(),
        "future_predictions.csv": PREDICTIONS_FILE.exists(),
        "cashflow_prediction_1998_2025_v1.csv": DATA_FILE.exists()
    }
    
    return {
        "success": True,
        "data": files
    }

@app.get("/api/model/feature-importance")
async def get_feature_importance():
    """Get feature importance/SHAP values for model explainability"""
    try:
        if not MODEL_FILE.exists():
            return {
                "success": False,
                "message": "Model not found. Please train a model first.",
                "data": None
            }
        
        # Load model
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data.get('model')
        model_name = model_data.get('model_name', 'Unknown')
        model_type = model_data.get('model_type', 'Unknown')
        # Try both 'feature_names' and 'feature_cols' (different versions use different keys)
        feature_names = model_data.get('feature_names') or model_data.get('feature_cols') or []
        
        feature_importance = []
        
        # Try to get feature importance
        if model_type == 'prophet':
            # Prophet doesn't have direct feature importance
            return {
                "success": True,
                "data": {
                    "modelName": model_name,
                    "modelType": model_type,
                    "message": "Prophet models use time-based components for forecasting",
                    "features": [
                        {"name": "Trend", "importance": 0.35, "description": "Long-term trend in the data"},
                        {"name": "Yearly Seasonality", "importance": 0.25, "description": "Annual patterns"},
                        {"name": "Monthly Seasonality", "importance": 0.20, "description": "Monthly patterns"},
                        {"name": "Weekly Seasonality", "importance": 0.15, "description": "Weekly patterns"},
                        {"name": "Holidays", "importance": 0.05, "description": "Holiday effects"}
                    ]
                }
            }
        else:
            # Tree-based models have feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Pair with feature names and sort
                for i, (name, importance) in enumerate(zip(feature_names, importances)):
                    feature_importance.append({
                        "name": name,
                        "importance": float(importance),
                        "rank": i + 1
                    })
                
                # Sort by importance descending
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                
                # Update ranks after sorting
                for i, feat in enumerate(feature_importance):
                    feat['rank'] = i + 1
                
                # Return top 15 features
                top_features = feature_importance[:15]
                
                return {
                    "success": True,
                    "data": {
                        "modelName": model_name,
                        "modelType": model_type,
                        "totalFeatures": len(feature_names),
                        "features": top_features
                    }
                }
            else:
                return {
                    "success": True,
                    "data": {
                        "modelName": model_name,
                        "modelType": model_type,
                        "message": "Feature importance not available for this model type",
                        "features": []
                    }
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/sample")
async def get_sample_data():
    """Get all data for display with filtering on frontend"""
    try:
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )
        
        # Read data
        df = pd.read_csv(DATA_FILE)
        
        # Select key columns for display
        display_columns = [
            'date', 'cash_inflow_usd', 'cash_outflow_usd', 'net_cashflow_usd',
            'vendor_payment_usd', 'salary_payment_usd', 'rent_usd', 'operational_expense_usd'
        ]
        
        # Filter to only existing columns
        available_columns = [col for col in display_columns if col in df.columns]
        sample_df = df[available_columns].copy()
        
        # Sort by date descending (newest first)
        if 'date' in sample_df.columns:
            sample_df = sample_df.sort_values('date', ascending=False)
        
        # Convert to list of dicts
        sample_data = sample_df.to_dict(orient='records')
        
        # Format numeric values
        for row in sample_data:
            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = round(value, 2)
        
        return {
            "success": True,
            "data": {
                "columns": available_columns,
                "rows": sample_data,
                "totalRows": len(df),
                "displayedRows": len(sample_data)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/predicted-vs-actual")
async def get_predicted_vs_actual(months: int = 12):
    """Get predicted vs actual values from saved comparison file
    
    Args:
        months: Number of months to return (default 12, use 0 for all)
    """
    try:
        # Path to predicted vs actual comparison file
        PRED_VS_ACTUAL_FILE = BACKEND_DIR / 'predicted_vs_actual.csv'
        
        # Check if comparison file exists
        if not PRED_VS_ACTUAL_FILE.exists():
            return {
                "success": False,
                "message": "Predicted vs actual comparison not available. Please retrain the model.",
                "data": None
            }
        
        # Read comparison data
        df = pd.read_csv(PRED_VS_ACTUAL_FILE)
        
        # Get model info - USE TEST METRICS FROM TRAINING (not recalculated)
        model_name = "Unknown"
        model_type = "Unknown"
        test_r2 = None
        test_mape = None
        test_mae = None
        
        if MODEL_INFO_FILE.exists():
            with open(MODEL_INFO_FILE, 'r') as f:
                model_info = json.load(f)
                model_name = model_info.get('best_model', 'Unknown')
                model_type = model_info.get('model_type', 'Unknown')
                # Get the TEST metrics from training (true predictive accuracy)
                test_r2 = model_info.get('test_r2')
                test_mape = model_info.get('test_mape')
                test_mae = model_info.get('test_mae')
        
        # Convert to comparison data format
        comparison_data = []
        for _, row in df.iterrows():
            comparison_data.append({
                "month": row['month'],
                "actual": float(row['actual']),
                "predicted": float(row['predicted']),
                "error": float(row['error']) if 'error' in df.columns else None,
                "percentError": float(row['percent_error']) if 'percent_error' in df.columns else None
            })
        
        # Sort by month
        comparison_data.sort(key=lambda x: x['month'])
        
        # Total months available
        total_available = len(comparison_data)
        
        # Filter by requested months (0 = all)
        if months > 0 and len(comparison_data) > months:
            comparison_data = comparison_data[-months:]
        
        # Calculate display metrics for the visible data (for reference)
        actuals = np.array([d["actual"] for d in comparison_data])
        predictions = np.array([d["predicted"] for d in comparison_data])
        
        # Calculate display MAPE (fitting accuracy on visible data)
        display_mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Calculate display MAE
        display_mae = np.mean(np.abs(actuals - predictions))
        
        # Use TEST metrics from training for accuracy indicators
        # These are the TRUE predictive metrics on unseen data
        metrics = {
            # Test metrics (true predictive accuracy from training)
            "r2": round(test_r2, 4) if test_r2 is not None else None,
            "mape": round(test_mape, 2) if test_mape is not None else round(display_mape, 2),
            "mae": round(test_mae, 2) if test_mae is not None else round(display_mae, 2),
            "accuracy": round(100 - min(test_mape if test_mape else display_mape, 100), 2),
            # Display metrics (for visible data range - shown separately)
            "displayMape": round(display_mape, 2),
            "displayMae": round(display_mae, 2),
            "metricsSource": "test_set" if test_r2 is not None else "calculated"
        }
        
        return {
            "success": True,
            "data": {
                "comparison": comparison_data,
                "metrics": metrics,
                "modelName": model_name,
                "modelType": model_type,
                "displayedMonths": len(comparison_data),
                "totalMonthsAvailable": total_available
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/forecast-accuracy")
async def get_forecast_accuracy():
    """Compare stored predictions with actual data when it becomes available"""
    try:
        # Check if prediction history exists
        if not PREDICTION_HISTORY_FILE.exists():
            return {
                "success": False,
                "message": "No prediction history found. Generate predictions first.",
                "data": None
            }
        
        if not DATA_FILE.exists():
            raise HTTPException(
                status_code=404,
                detail="Data file not found"
            )
        
        # Read prediction history
        history_df = pd.read_csv(PREDICTION_HISTORY_FILE)
        
        # Read actual data and aggregate to monthly
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly_actuals = df.groupby('year_month').agg({
            'cash_outflow_usd': 'sum'
        }).reset_index()
        monthly_actuals['month'] = monthly_actuals['year_month'].dt.strftime('%Y-%m')
        monthly_actuals = monthly_actuals[['month', 'cash_outflow_usd']]
        monthly_actuals.columns = ['month', 'actual']
        
        # Merge predictions with actuals
        comparison_data = []
        verified_count = 0
        pending_count = 0
        
        for _, pred_row in history_df.iterrows():
            pred_month = pred_row['month']
            predicted_value = float(pred_row['predicted_cash_outflow'])
            predicted_on = pred_row.get('predicted_on', 'Unknown')
            
            # Check if actual data exists for this month
            actual_row = monthly_actuals[monthly_actuals['month'] == pred_month]
            
            if len(actual_row) > 0:
                actual_value = float(actual_row['actual'].values[0])
                error = actual_value - predicted_value
                percent_error = (error / actual_value * 100) if actual_value != 0 else 0
                
                comparison_data.append({
                    "month": pred_month,
                    "predicted": predicted_value,
                    "actual": actual_value,
                    "error": error,
                    "percentError": round(percent_error, 2),
                    "predictedOn": predicted_on,
                    "status": "verified"
                })
                verified_count += 1
            else:
                # Prediction for future month - actual data not yet available
                comparison_data.append({
                    "month": pred_month,
                    "predicted": predicted_value,
                    "actual": None,
                    "error": None,
                    "percentError": None,
                    "predictedOn": predicted_on,
                    "status": "pending"
                })
                pending_count += 1
        
        # Sort by month
        comparison_data.sort(key=lambda x: x['month'])
        
        # Calculate accuracy metrics for verified predictions only
        verified_data = [d for d in comparison_data if d['status'] == 'verified']
        metrics = {}
        
        if len(verified_data) > 0:
            actuals = np.array([d['actual'] for d in verified_data])
            predictions = np.array([d['predicted'] for d in verified_data])
            
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            mae = np.mean(np.abs(actuals - predictions))
            
            # R²
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            metrics = {
                "mape": round(mape, 2),
                "mae": round(mae, 2),
                "r2": round(r2, 4),
                "accuracy": round(100 - min(mape, 100), 2)
            }
        
        return {
            "success": True,
            "data": {
                "comparisons": comparison_data,
                "metrics": metrics,
                "verifiedCount": verified_count,
                "pendingCount": pending_count,
                "totalPredictions": len(comparison_data)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static frontend files in production
# The frontend_build directory is created during Docker build
if FRONTEND_BUILD_DIR.exists():
    # Serve static files (JS, CSS, images, etc.)
    static_dir = FRONTEND_BUILD_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Serve the React app for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        # Don't serve frontend for API routes
        if full_path.startswith("api/") or full_path in ["docs", "redoc", "openapi.json"]:
            raise HTTPException(status_code=404, detail="Not found")
        
        # Try to serve the file directly (for assets like favicon, manifest, etc.)
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Serve index.html for all other routes (React Router handles routing)
        index_file = FRONTEND_BUILD_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
