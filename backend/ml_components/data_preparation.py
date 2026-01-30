"""
Advanced Data Preparation Module for Cash Flow Prediction
Includes:
- Feature Engineering
- Correlation Analysis
- Feature Selection
- PCA Dimensionality Reduction
- Data Cleaning and Preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    SelectFromModel, RFE
)
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')


class DataPreparator:
    """Advanced data preparation class for cash flow prediction"""
    
    def __init__(self, correlation_threshold=0.95, pca_variance_threshold=0.95):
        """
        Initialize DataPreparator
        
        Parameters:
        -----------
        correlation_threshold : float
            Threshold for removing highly correlated features (default: 0.95)
        pca_variance_threshold : float
            Variance threshold for PCA (default: 0.95)
        """
        self.correlation_threshold = correlation_threshold
        self.pca_variance_threshold = pca_variance_threshold
        self.scaler = None
        self.pca = None
        self.selected_features = None
        self.feature_selector = None
        self.prophet_scaler = None
        self.regressor_scalers = {}
    
    def generate_data(self, start_date, end_date, existing_df=None):
        """
        Generate cash flow data for a specific date range
        This function replicates the logic from data_generate.py
        
        Parameters:
        -----------
        start_date : str or pd.Timestamp
            Start date for data generation
        end_date : str or pd.Timestamp
            End date for data generation (inclusive)
        existing_df : pd.DataFrame, optional
            Existing dataframe to maintain cash continuity
            
        Returns:
        --------
        df : pd.DataFrame
            Generated dataframe with cash flow data
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Convert to Timestamp if strings
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        N = len(dates)
        
        if N == 0:
            return pd.DataFrame()
        
        df_dates = pd.DataFrame({"date": dates})
        df_dates["year"] = df_dates["date"].dt.year
        
        # Get last closing cash from existing data if available
        last_closing_cash = None
        if existing_df is not None and len(existing_df) > 0:
            last_closing_cash = existing_df['closing_cash_usd'].iloc[-1]
        
        # Cashflow drivers
        cash_inflow = np.random.uniform(10_000, 500_000, N)
        cash_outflow = np.random.uniform(8_000, 480_000, N)
        net_cashflow = cash_inflow - cash_outflow
        
        # CASH CONTINUITY LOGIC
        opening_cash = np.zeros(N)
        closing_cash = np.zeros(N)
        
        # Set initial opening cash
        if last_closing_cash is not None:
            opening_cash[0] = last_closing_cash
        else:
            # For new file, use year-based logic for first day of year
            first_year = df_dates["year"].iloc[0]
            if first_year == df_dates["year"].min():
                opening_cash[0] = np.random.uniform(500_000, 5_000_000)
            else:
                # This shouldn't happen for new files, but handle it
                opening_cash[0] = np.random.uniform(500_000, 5_000_000)
        
        # Daily propagation - simple continuity
        for i in range(N):
            if i > 0:
                opening_cash[i] = closing_cash[i - 1]
            closing_cash[i] = opening_cash[i] + net_cashflow[i]
        
        # Cash source
        cash_source = np.random.choice(
            ["customer_payment", "loan_disbursement", "investment", "asset_sale"],
            N,
            p=[0.7, 0.15, 0.1, 0.05]
        )
        
        # Invoice behavior
        payment_behavior = np.random.choice(
            ["full", "partial", "unpaid"],
            N,
            p=[0.65, 0.25, 0.10]
        )
        
        invoice_amount = np.random.uniform(5_000, 300_000, N)
        invoice_paid = np.zeros(N)
        
        full_mask = payment_behavior == "full"
        invoice_paid[full_mask] = invoice_amount[full_mask]
        
        partial_mask = payment_behavior == "partial"
        invoice_paid[partial_mask] = (
            invoice_amount[partial_mask] *
            np.random.uniform(0.3, 0.9, partial_mask.sum())
        )
        
        unpaid_mask = payment_behavior == "unpaid"
        invoice_paid[unpaid_mask] = 0
        
        partial_payment_flag = payment_behavior == "partial"
        bad_debt_flag = payment_behavior == "unpaid"
        
        # Payment timing
        days_payment_delay = np.random.randint(-5, 60, N)
        invoice_due_date = dates
        invoice_payment_date = dates + pd.to_timedelta(days_payment_delay, unit="D")
        
        customer_payment_usd = invoice_paid
        
        # Expenses
        vendor_payment = np.random.uniform(5_000, 200_000, N)
        salary_payment = np.random.uniform(20_000, 150_000, N)
        rent = np.random.uniform(5_000, 50_000, N)
        tax_payment = np.random.uniform(3_000, 100_000, N)
        loan_emi = np.random.uniform(0, 80_000, N)
        operational_expense = np.random.uniform(2_000, 120_000, N)
        
        ppe_expense = np.where(
            np.random.rand(N) < 0.15,
            np.random.uniform(20_000, 1_000_000, N),
            0
        )
        
        expense_source = np.random.choice(
            ["operations", "payroll", "rent", "tax", "loan", "ppe_capex"],
            N,
            p=[0.35, 0.25, 0.1, 0.1, 0.1, 0.1]
        )
        
        # Time features
        day_of_week = dates.dayofweek + 1
        week_of_month = (dates.day - 1) // 7 + 1
        month = dates.month
        quarter = dates.quarter
        is_month_end = dates.is_month_end
        is_holiday = np.random.rand(N) < 0.1
        
        # External indicators
        interest_rate = np.random.uniform(2.0, 8.0, N)
        inflation = np.random.uniform(1.0, 6.0, N)
        fx_index = np.random.uniform(90, 120, N)
        economic_sentiment = np.random.uniform(-1, 1, N)
        
        # Lag & rolling (for new data, use last values from existing data)
        if existing_df is not None and len(existing_df) > 0 and 'net_cashflow_usd' in existing_df.columns:
            # Get last values for lag features
            last_values = existing_df['net_cashflow_usd'].tail(30).values if len(existing_df) >= 30 else existing_df['net_cashflow_usd'].values
            last_val_1d = last_values[-1] if len(last_values) > 0 else 0
            last_val_7d = last_values[-7] if len(last_values) >= 7 else (last_values[-1] if len(last_values) > 0 else 0)
            last_val_30d = last_values[-30] if len(last_values) >= 30 else (last_values[0] if len(last_values) > 0 else 0)
            
            # Build lag arrays
            cashflow_lag_1d = np.concatenate([[last_val_1d], net_cashflow[:-1]]) if N > 0 else np.array([last_val_1d])
            if N > 7:
                cashflow_lag_7d = np.concatenate([last_values[-7:], net_cashflow[:-7]])
            elif N > 0:
                cashflow_lag_7d = np.concatenate([np.full(N, last_val_7d)])
            else:
                cashflow_lag_7d = np.array([last_val_7d])
            
            if N > 30:
                cashflow_lag_30d = np.concatenate([last_values[-30:], net_cashflow[:-30]])
            elif N > 0:
                cashflow_lag_30d = np.concatenate([np.full(N, last_val_30d)])
            else:
                cashflow_lag_30d = np.array([last_val_30d])
        else:
            cashflow_lag_1d = np.roll(net_cashflow, 1)
            cashflow_lag_7d = np.roll(net_cashflow, 7)
            cashflow_lag_30d = np.roll(net_cashflow, 30)
        
        # Rolling averages (for new data, calculate from combined if needed)
        if existing_df is not None and len(existing_df) > 0 and 'net_cashflow_usd' in existing_df.columns:
            # Use last 30 days for rolling calculations
            last_30 = existing_df['net_cashflow_usd'].tail(30).values
            all_net_cashflow = np.concatenate([last_30, net_cashflow])
            rolling_avg_7d = pd.Series(all_net_cashflow).rolling(7, min_periods=1).mean().fillna(0).values[-N:]
            rolling_std_30d = pd.Series(all_net_cashflow).rolling(30, min_periods=1).std().fillna(0).values[-N:]
        else:
            rolling_avg_7d = pd.Series(net_cashflow).rolling(7, min_periods=1).mean().fillna(0).values
            rolling_std_30d = pd.Series(net_cashflow).rolling(30, min_periods=1).std().fillna(0).values
        
        # Agentic fields
        forecast_version = np.random.choice(["v1", "v2", "v3"], N)
        reforecast_trigger = np.random.choice(
            ["delay_detected", "expense_spike", "none"],
            N,
            p=[0.2, 0.1, 0.7]
        )
        
        confidence_score = np.random.uniform(0.7, 0.99, N)
        alert_flag = closing_cash < 100_000
        recommended_action = np.where(alert_flag, "short_term_borrowing", "no_action")
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "opening_cash_usd": opening_cash,
            "cash_inflow_usd": cash_inflow,
            "cash_outflow_usd": cash_outflow,
            "net_cashflow_usd": net_cashflow,
            "closing_cash_usd": closing_cash,
            
            "cash_source": cash_source,
            "expense_source": expense_source,
            "ppe_expense_usd": ppe_expense,
            
            "customer_payment_usd": customer_payment_usd,
            "invoice_amount_usd": invoice_amount,
            "invoice_paid_usd": invoice_paid,
            "invoice_due_date": invoice_due_date,
            "invoice_payment_date": invoice_payment_date,
            "days_payment_delay": days_payment_delay,
            "partial_payment_flag": partial_payment_flag,
            "bad_debt_flag": bad_debt_flag,
            
            "vendor_payment_usd": vendor_payment,
            "salary_payment_usd": salary_payment,
            "rent_usd": rent,
            "tax_payment_usd": tax_payment,
            "loan_emi_usd": loan_emi,
            "operational_expense_usd": operational_expense,
            
            "day_of_week": day_of_week,
            "week_of_month": week_of_month,
            "month": month,
            "quarter": quarter,
            "is_month_end": is_month_end,
            "is_holiday": is_holiday,
            
            "interest_rate_pct": interest_rate,
            "inflation_pct": inflation,
            "fx_rate_usd_index": fx_index,
            "economic_sentiment_score": economic_sentiment,
            
            "cashflow_lag_1d_usd": cashflow_lag_1d,
            "cashflow_lag_7d_usd": cashflow_lag_7d,
            "cashflow_lag_30d_usd": cashflow_lag_30d,
            "rolling_avg_7d_usd": rolling_avg_7d,
            "rolling_std_30d_usd": rolling_std_30d,
            
            "forecast_version": forecast_version,
            "reforecast_trigger": reforecast_trigger,
            "confidence_score": confidence_score,
            "alert_flag": alert_flag,
            "recommended_action": recommended_action
        })
        
        return df
    
    def update_data_file(self, filepath="cashflow_prediction_1998_2025_v1.csv"):
        """
        Update data file by appending new data up to yesterday
        This function is called automatically when load_and_aggregate is called
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
            
        Returns:
        --------
        updated : bool
            True if data was updated, False otherwise
        """
        yesterday = pd.Timestamp.today() - pd.Timedelta(days=1)
        start_date = pd.Timestamp("1998-01-01")
        
        # Check if file exists
        if os.path.exists(filepath):
            try:
                # Read existing data to get last date
                existing_df = pd.read_csv(filepath, parse_dates=["date"])
                existing_df = existing_df.sort_values("date").reset_index(drop=True)
                last_date = pd.Timestamp(existing_df['date'].max())
                
                # Check if update is needed
                if last_date >= yesterday:
                    # Data is already up to date
                    return False
                
                # Generate data from last_date + 1 day to yesterday
                new_start = last_date + pd.Timedelta(days=1)
                print(f"\n[Data Update] Generating data from {new_start.strftime('%Y-%m-%d')} to {yesterday.strftime('%Y-%m-%d')}...")
                
                new_df = self.generate_data(new_start, yesterday, existing_df)
                
                if len(new_df) > 0:
                    # Append new data
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.sort_values("date").reset_index(drop=True)
                    
                    # Save updated file
                    combined_df.to_csv(filepath, index=False)
                    print(f"✅ Data updated: Added {len(new_df)} new day(s) of data")
                    print(f"   Date range: {existing_df['date'].min().strftime('%Y-%m-%d')} to {combined_df['date'].max().strftime('%Y-%m-%d')}")
                    return True
                else:
                    return False
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not update existing data file: {str(e)}")
                print("   Generating new data file from scratch...")
                # Fall through to generate new file
        else:
            print(f"\n[Data Generation] Creating new data file...")
        
        # Generate complete dataset from scratch
        new_df = self.generate_data(start_date, yesterday, None)
        
        if len(new_df) > 0:
            new_df.to_csv(filepath, index=False)
            print(f"✅ Data file created: {len(new_df)} day(s) of data")
            print(f"   Date range: {new_df['date'].min().strftime('%Y-%m-%d')} to {new_df['date'].max().strftime('%Y-%m-%d')}")
            return True
        
        return False
        
    def load_and_aggregate(self, filepath="cashflow_prediction_1998_2025_v1.csv", auto_update=True):
        """
        Load and aggregate daily data to monthly
        Only includes complete months (excludes current incomplete month)
        Handles leap years automatically via pandas date functionality
        
        Automatically updates data file with yesterday's data before loading.
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file
        auto_update : bool
            If True, automatically generate and append data up to yesterday (default: True)
            
        Returns:
        --------
        monthly_df : pd.DataFrame
            Aggregated monthly data (only complete months)
        """
        # Automatically update data file with yesterday's data
        if auto_update:
            self.update_data_file(filepath)
        
        print("\n[1/7] Loading and aggregating data to monthly...")
        
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        print(f"Daily data shape: {df.shape}")
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Determine last complete month
        # Get today's date
        today = pd.Timestamp.today()
        
        # Get the first day of current month
        first_day_current_month = today.replace(day=1)
        
        # Last complete date is the day before the first day of current month
        # This automatically handles leap years (pandas knows Feb has 28 or 29 days)
        last_complete_date = first_day_current_month - pd.Timedelta(days=1)
        
        # Filter data to only include complete months
        df_complete = df[df['date'] <= last_complete_date].copy()
        
        if len(df_complete) < len(df):
            excluded_days = len(df) - len(df_complete)
            excluded_start = df[df['date'] > last_complete_date]['date'].min()
            print(f"  Excluding {excluded_days} day(s) from incomplete month(s)")
            print(f"  Last complete month: {last_complete_date.strftime('%Y-%m')}")
            print(f"  Excluded data starts from: {excluded_start.strftime('%Y-%m-%d')}")
        else:
            print(f"  All data is from complete months")
            print(f"  Last complete month: {last_complete_date.strftime('%Y-%m')}")
        
        # Aggregate to monthly (only complete months)
        df_complete['year_month'] = df_complete['date'].dt.to_period('M')
        
        monthly_df = df_complete.groupby('year_month').agg({
            # Target variable
            'cash_outflow_usd': 'sum',
            
            # Financial features
            'cash_inflow_usd': 'sum',
            'vendor_payment_usd': 'sum',
            'salary_payment_usd': 'sum',
            'rent_usd': 'sum',
            'operational_expense_usd': 'sum',
            
            # Economic indicators
            'interest_rate_pct': 'mean',
            'inflation_pct': 'mean',
            'economic_sentiment_score': 'mean',
        }).reset_index()
        
        # Convert period to datetime
        monthly_df['date'] = monthly_df['year_month'].dt.to_timestamp()
        monthly_df = monthly_df.sort_values('date').reset_index(drop=True)
        monthly_df = monthly_df.drop('year_month', axis=1)
        
        print(f"Monthly data shape: {monthly_df.shape}")
        print(f"Total months: {len(monthly_df)}")
        print(f"Monthly range: {monthly_df['date'].min().strftime('%Y-%m')} to {monthly_df['date'].max().strftime('%Y-%m')}")
        
        return monthly_df
    
    def engineer_features(self, monthly_df):
        """
        Advanced feature engineering
        
        Parameters:
        -----------
        monthly_df : pd.DataFrame
            Monthly aggregated data
            
        Returns:
        --------
        feature_df : pd.DataFrame
            DataFrame with engineered features
        """
        print("\n[2/7] Engineering advanced features...")
        
        feature_df = monthly_df.copy()
        
        # ========== Temporal Features ==========
        feature_df['month'] = feature_df['date'].dt.month
        feature_df['quarter'] = feature_df['date'].dt.quarter
        feature_df['year'] = feature_df['date'].dt.year
        feature_df['day_of_year'] = feature_df['date'].dt.dayofyear
        feature_df['week_of_year'] = feature_df['date'].dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        feature_df['month_sin'] = np.sin(2 * np.pi * feature_df['month'] / 12)
        feature_df['month_cos'] = np.cos(2 * np.pi * feature_df['month'] / 12)
        feature_df['quarter_sin'] = np.sin(2 * np.pi * feature_df['quarter'] / 4)
        feature_df['quarter_cos'] = np.cos(2 * np.pi * feature_df['quarter'] / 4)
        feature_df['year_sin'] = np.sin(2 * np.pi * (feature_df['year'] - feature_df['year'].min()) / (feature_df['year'].max() - feature_df['year'].min() + 1))
        feature_df['year_cos'] = np.cos(2 * np.pi * (feature_df['year'] - feature_df['year'].min()) / (feature_df['year'].max() - feature_df['year'].min() + 1))
        
        # Tax and fiscal indicators
        feature_df['is_tax_month'] = feature_df['month'].isin([1, 4, 6, 9]).astype(int)
        feature_df['is_q1_tax'] = (feature_df['month'] == 4).astype(int)
        feature_df['is_q2_tax'] = (feature_df['month'] == 6).astype(int)
        feature_df['is_q3_tax'] = (feature_df['month'] == 9).astype(int)
        feature_df['is_q4_tax'] = (feature_df['month'] == 1).astype(int)
        feature_df['is_april'] = (feature_df['month'] == 4).astype(int)
        feature_df['is_october'] = (feature_df['month'] == 10).astype(int)
        feature_df['is_quarter_end'] = feature_df['month'].isin([3, 6, 9, 12]).astype(int)
        feature_df['is_year_end'] = (feature_df['month'] == 12).astype(int)
        feature_df['is_year_start'] = (feature_df['month'] == 1).astype(int)
        
        # ========== Lag Features ==========
        # Target variable lags
        for lag in [1, 2, 3, 6, 12]:
            feature_df[f'cash_outflow_lag{lag}'] = feature_df['cash_outflow_usd'].shift(lag)
            feature_df[f'cash_inflow_lag{lag}'] = feature_df['cash_inflow_usd'].shift(lag)
        
        # Payment type lags
        feature_df['vendor_payment_lag1'] = feature_df['vendor_payment_usd'].shift(1)
        feature_df['salary_payment_lag1'] = feature_df['salary_payment_usd'].shift(1)
        feature_df['rent_lag1'] = feature_df['rent_usd'].shift(1)
        feature_df['operational_expense_lag1'] = feature_df['operational_expense_usd'].shift(1)
        
        # ========== Rolling Statistics ==========
        windows = [3, 6, 12]
        for window in windows:
            # Rolling mean
            feature_df[f'cash_outflow_rolling_mean_{window}'] = feature_df['cash_outflow_usd'].rolling(window=window, min_periods=1).mean()
            feature_df[f'cash_inflow_rolling_mean_{window}'] = feature_df['cash_inflow_usd'].rolling(window=window, min_periods=1).mean()
            
            # Rolling std
            feature_df[f'cash_outflow_rolling_std_{window}'] = feature_df['cash_outflow_usd'].rolling(window=window, min_periods=1).std().fillna(0)
            feature_df[f'cash_inflow_rolling_std_{window}'] = feature_df['cash_inflow_usd'].rolling(window=window, min_periods=1).std().fillna(0)
            
            # Rolling min/max
            feature_df[f'cash_outflow_rolling_min_{window}'] = feature_df['cash_outflow_usd'].rolling(window=window, min_periods=1).min()
            feature_df[f'cash_outflow_rolling_max_{window}'] = feature_df['cash_outflow_usd'].rolling(window=window, min_periods=1).max()
        
        # ========== Rate of Change Features ==========
        feature_df['cash_outflow_mom'] = feature_df['cash_outflow_usd'].pct_change(1).fillna(0)  # Month-over-month
        feature_df['cash_outflow_yoy'] = feature_df['cash_outflow_usd'].pct_change(12).fillna(0)  # Year-over-year
        feature_df['cash_inflow_mom'] = feature_df['cash_inflow_usd'].pct_change(1).fillna(0)
        feature_df['cash_inflow_yoy'] = feature_df['cash_inflow_usd'].pct_change(12).fillna(0)
        
        # ========== Ratio Features ==========
        feature_df['vendor_payment_ratio'] = feature_df['vendor_payment_usd'] / (feature_df['cash_outflow_usd'] + 1e-8)
        feature_df['salary_payment_ratio'] = feature_df['salary_payment_usd'] / (feature_df['cash_outflow_usd'] + 1e-8)
        feature_df['rent_ratio'] = feature_df['rent_usd'] / (feature_df['cash_outflow_usd'] + 1e-8)
        feature_df['operational_expense_ratio'] = feature_df['operational_expense_usd'] / (feature_df['cash_outflow_usd'] + 1e-8)
        feature_df['inflow_outflow_ratio'] = feature_df['cash_inflow_usd'] / (feature_df['cash_outflow_usd'] + 1e-8)
        
        # ========== Trend Features ==========
        # Linear trend
        feature_df['trend'] = np.arange(len(feature_df))
        
        # Exponential moving averages
        feature_df['cash_outflow_ema_3'] = feature_df['cash_outflow_usd'].ewm(span=3, adjust=False).mean()
        feature_df['cash_outflow_ema_12'] = feature_df['cash_outflow_usd'].ewm(span=12, adjust=False).mean()
        feature_df['cash_inflow_ema_3'] = feature_df['cash_inflow_usd'].ewm(span=3, adjust=False).mean()
        feature_df['cash_inflow_ema_12'] = feature_df['cash_inflow_usd'].ewm(span=12, adjust=False).mean()
        
        # ========== Interaction Features ==========
        feature_df['interest_inflation_interaction'] = feature_df['interest_rate_pct'] * feature_df['inflation_pct']
        feature_df['sentiment_inflow_interaction'] = feature_df['economic_sentiment_score'] * feature_df['cash_inflow_usd']
        feature_df['tax_month_outflow_interaction'] = feature_df['is_tax_month'] * feature_df['cash_outflow_usd']
        
        # ========== Outlier Detection ==========
        Q1 = feature_df['cash_outflow_usd'].quantile(0.25)
        Q3 = feature_df['cash_outflow_usd'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        feature_df['is_outlier_iqr'] = ((feature_df['cash_outflow_usd'] < lower_bound) | 
                                       (feature_df['cash_outflow_usd'] > upper_bound)).astype(int)
        
        z_scores = np.abs(stats.zscore(feature_df['cash_outflow_usd'].fillna(0)))
        feature_df['is_outlier_zscore'] = (z_scores > 3).astype(int)
        feature_df['is_outlier_combined'] = ((feature_df['is_outlier_iqr'] == 1) | 
                                            (feature_df['is_outlier_zscore'] == 1)).astype(int)
        
        # ========== Fill NaN values ==========
        # Forward fill for lag features, then backward fill, then zero fill
        feature_df = feature_df.ffill().bfill().fillna(0)
        
        # Replace inf values
        feature_df = feature_df.replace([np.inf, -np.inf], 0)
        
        print(f"Created {len(feature_df.columns)} features")
        
        return feature_df
    
    def analyze_correlations(self, feature_df, target_col='cash_outflow_usd'):
        """
        Analyze correlations and remove highly correlated features
        
        Parameters:
        -----------
        feature_df : pd.DataFrame
            DataFrame with features
        target_col : str
            Name of target column
            
        Returns:
        --------
        feature_df : pd.DataFrame
            DataFrame with highly correlated features removed
        removed_features : list
            List of removed feature names
        """
        print("\n[3/7] Analyzing correlations and removing redundant features...")
        
        # Get numeric columns only
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_col and col != 'date']
        
        if len(numeric_cols) == 0:
            return feature_df, []
        
        # Calculate correlation matrix
        corr_matrix = feature_df[numeric_cols].corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove (keep the one with higher correlation to target)
        to_remove = set()
        target_corrs = feature_df[numeric_cols + [target_col]].corr()[target_col].abs()
        
        for col in upper_triangle.columns:
            for row in upper_triangle.index:
                if upper_triangle.loc[row, col] > self.correlation_threshold:
                    # Keep feature with higher correlation to target
                    if target_corrs[col] > target_corrs[row]:
                        to_remove.add(row)
                    else:
                        to_remove.add(col)
        
        removed_features = list(to_remove)
        
        if removed_features:
            print(f"  Removed {len(removed_features)} highly correlated features:")
            for feat in removed_features[:10]:  # Show first 10
                print(f"    - {feat}")
            if len(removed_features) > 10:
                print(f"    ... and {len(removed_features) - 10} more")
            feature_df = feature_df.drop(columns=removed_features)
        else:
            print("  No highly correlated features found to remove")
        
        return feature_df, removed_features
    
    def select_features(self, X, y, method='mutual_info', n_features=None):
        """
        Select best features using various methods
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target vector
        method : str
            Method to use: 'mutual_info', 'f_regression', 'rf_importance', 'rfe'
        n_features : int
            Number of features to select (None = auto)
            
        Returns:
        --------
        X_selected : np.array
            Selected feature matrix
        selected_indices : np.array
            Indices of selected features
        """
        print(f"\n[4/7] Selecting features using {method}...")
        
        if n_features is None:
            n_features = min(50, X.shape[1])  # Default to 50 or all features if less
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=n_features)
        elif method == 'rf_importance':
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            selector = SelectFromModel(rf, prefit=True, max_features=n_features)
        elif method == 'rfe':
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            selector = RFE(rf, n_features_to_select=n_features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        self.feature_selector = selector
        
        if hasattr(selector, 'get_support'):
            selected_indices = selector.get_support(indices=True)
        else:
            selected_indices = np.arange(X.shape[1])
        
        print(f"  Selected {len(selected_indices)} features out of {X.shape[1]}")
        
        return X_selected, selected_indices
    
    def apply_pca(self, X, variance_threshold=None):
        """
        Apply PCA for dimensionality reduction
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        variance_threshold : float
            Variance threshold (None = use default)
            
        Returns:
        --------
        X_pca : np.array
            PCA-transformed feature matrix
        n_components : int
            Number of components used
        """
        if variance_threshold is None:
            variance_threshold = self.pca_variance_threshold
        
        print(f"\n[5/7] Applying PCA (variance threshold: {variance_threshold})...")
        
        # Standardize before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        X_pca_full = pca.fit_transform(X_scaled)
        
        # Find number of components to explain variance threshold
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        # Reapply PCA with selected components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        self.pca = pca
        self.scaler = scaler
        
        explained_variance = cumsum_variance[n_components - 1]
        print(f"  Selected {n_components} components explaining {explained_variance:.2%} of variance")
        print(f"  Reduced from {X.shape[1]} to {n_components} features")
        
        return X_pca, n_components
    
    def prepare_data(self, monthly_df, use_pca=False, feature_selection_method='mutual_info', 
                     n_features=None, apply_correlation_filter=True):
        """
        Complete data preparation pipeline
        
        Parameters:
        -----------
        monthly_df : pd.DataFrame
            Monthly aggregated data
        use_pca : bool
            Whether to apply PCA
        feature_selection_method : str
            Method for feature selection
        n_features : int
            Number of features to select
        apply_correlation_filter : bool
            Whether to remove highly correlated features
            
        Returns:
        --------
        X : np.array
            Prepared feature matrix
        y : np.array
            Target vector
        feature_names : list
            List of feature names
        feature_df : pd.DataFrame
            DataFrame with all features
        """
        # Engineer features
        feature_df = self.engineer_features(monthly_df)
        
        # Remove highly correlated features
        if apply_correlation_filter:
            feature_df, removed = self.analyze_correlations(feature_df, 'cash_outflow_usd')
        
        # Prepare target
        y = feature_df['cash_outflow_usd'].values
        
        # Get feature columns (exclude target and date)
        exclude_cols = ['cash_outflow_usd', 'date', 'year_month']
        feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Extract feature matrix
        X = feature_df[feature_cols].values
        
        print(f"\n[6/7] Final feature matrix shape: {X.shape}")
        
        # Feature selection
        if feature_selection_method and not use_pca:
            X, selected_indices = self.select_features(X, y, method=feature_selection_method, n_features=n_features)
            feature_names = [feature_cols[i] for i in selected_indices]
            self.selected_features = feature_names
        else:
            feature_names = feature_cols
        
        # Apply PCA if requested
        if use_pca:
            X, n_components = self.apply_pca(X)
            feature_names = [f'PC{i+1}' for i in range(n_components)]
        
        print(f"\n[7/7] Data preparation complete!")
        print(f"  Final feature count: {len(feature_names)}")
        print(f"  Sample count: {len(y)}")
        
        return X, y, feature_names, feature_df
    
    def prepare_prophet_data(self, monthly_df):
        """
        Prepare data for Prophet model
        
        Parameters:
        -----------
        monthly_df : pd.DataFrame
            Monthly aggregated data
            
        Returns:
        --------
        prophet_df : pd.DataFrame
            DataFrame formatted for Prophet
        selected_regressors : dict
            Dictionary of selected regressors
        """
        print("\nPreparing Prophet data...")
        
        prophet_df = monthly_df[['date', 'cash_outflow_usd']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Add potential regressors
        potential_regressors = {
            'cash_inflow': monthly_df['cash_inflow_usd'].values,
            'interest_rate': monthly_df['interest_rate_pct'].values,
            'inflation': monthly_df['inflation_pct'].values,
            'economic_sentiment': monthly_df['economic_sentiment_score'].values,
        }
        
        # Select regressors based on correlation
        selected_regressors = {}
        for name, values in potential_regressors.items():
            correlation = np.corrcoef(prophet_df['y'].values, values)[0, 1]
            if abs(correlation) > 0.1 or name in ['cash_inflow', 'interest_rate']:
                prophet_df[name] = values
                selected_regressors[name] = values
        
        # Add temporal features
        prophet_df['month'] = prophet_df['ds'].dt.month
        prophet_df['is_tax_month'] = prophet_df['month'].isin([1, 4, 6, 9]).astype(int)
        prophet_df['is_quarter_end'] = prophet_df['month'].isin([3, 6, 9, 12]).astype(int)
        prophet_df['is_year_end'] = (prophet_df['month'] == 12).astype(int)
        prophet_df['month_sin'] = np.sin(2 * np.pi * prophet_df['month'] / 12)
        prophet_df['month_cos'] = np.cos(2 * np.pi * prophet_df['month'] / 12)
        
        for feat in ['is_tax_month', 'is_quarter_end', 'is_year_end', 'month_sin', 'month_cos']:
            if feat not in selected_regressors:
                selected_regressors[feat] = prophet_df[feat].values
        
        # Normalize target for Prophet
        self.prophet_scaler = MinMaxScaler()
        prophet_df['y_original'] = prophet_df['y'].copy()
        prophet_df['y'] = self.prophet_scaler.fit_transform(prophet_df[['y']]).flatten()
        
        # Standardize regressors
        for reg_name in selected_regressors.keys():
            if reg_name not in ['is_tax_month', 'is_quarter_end', 'is_year_end', 'month_sin', 'month_cos']:
                scaler = StandardScaler()
                original_values = prophet_df[reg_name].values.reshape(-1, 1)
                scaled_values = scaler.fit_transform(original_values).flatten()
                prophet_df[reg_name] = scaled_values
                self.regressor_scalers[reg_name] = scaler
        
        print(f"  Selected {len(selected_regressors)} regressors for Prophet")
        
        return prophet_df, selected_regressors

