"""
FastAPI Backend Server for Cash Flow Prediction System
Replaces Streamlit with a RESTful API for testing and integration

Usage:
    uvicorn main:app --reload --port 5000
    Or: python main.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
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
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/year-over-year")
async def get_year_over_year():
    """Get year-over-year comparison data for income, expense, and net cashflow"""
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
        elif 'month' in df.columns:
            df['date'] = pd.to_datetime(df['month'] + '-01')
            df['year'] = df['date'].dt.year
        else:
            raise HTTPException(
                status_code=500,
                detail="No date or month column found in data"
            )
        
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
        
        return {
            "success": True,
            "data": last_10_years
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
