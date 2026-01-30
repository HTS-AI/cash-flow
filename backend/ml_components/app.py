"""
Cash Flow Prediction System - Integration Application
This is the main integration file that combines all ML components:
- Data Generation
- Data Preprocessing
- Data Analysis
- Model Training (LSTM Cashflow Prediction)
- SHAP Explainability

Usage:
    python app.py [command] [options]
    
Commands:
    generate      - Generate/update cashflow data
    analyze       - Perform data analysis and visualization
    train         - Train ML models and generate predictions
    explain       - Generate SHAP explainability analysis
    full          - Run complete pipeline (generate -> analyze -> train -> explain)
"""

import sys
import os
import argparse
from datetime import datetime

# Safe print function to handle encoding issues on Windows
def safe_print(*args, **kwargs):
    """Print function that handles encoding errors gracefully"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace problematic characters with ASCII equivalents
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_arg = arg.replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARNING]')
                safe_args.append(safe_arg)
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)

# Import all ML components
from data_preparation import DataPreparator
from data_analysis import CashFlowAnalyzer
import subprocess


class CashFlowMLSystem:
    """Integrated Cash Flow ML System"""
    
    def __init__(self, data_file="cashflow_prediction_1998_2025_v1.csv"):
        """
        Initialize the ML system
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV data file (absolute path or relative to backend directory)
        """
        from pathlib import Path
        
        # Get backend directory (parent of ml_components)
        self.backend_dir = Path(__file__).parent.parent
        
        # If data_file is not absolute, assume it's in backend directory
        if not os.path.isabs(data_file):
            self.data_file = str(self.backend_dir / data_file)
        else:
            self.data_file = data_file
        
        self.data_prep = DataPreparator(
            correlation_threshold=0.95,
            pca_variance_threshold=0.95
        )
        self.analyzer = CashFlowAnalyzer(self.data_file)
    
    def generate_data(self):
        """Generate or update cashflow data"""
        print("\n" + "="*70)
        print("STEP 1: DATA GENERATION")
        print("="*70)
        
        try:
            import os
            from pathlib import Path
            
            # Run data generation script from backend directory
            backend_dir = Path(__file__).parent.parent
            data_gen_script = backend_dir / 'ml_components' / 'data_generate.py'
            print("Generating cashflow data...")
            result = subprocess.run(
                [sys.executable, str(data_gen_script)],
                cwd=str(backend_dir),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Data generation completed successfully")
                return True
            else:
                print("[ERROR] Error during data generation")
                return False
        except Exception as e:
            print(f"[ERROR] Error during data generation: {str(e)}")
            return False
    
    def analyze_data(self, forecast_months=0):
        """
        Perform comprehensive data analysis
        
        Parameters:
        -----------
        forecast_months : int
            Number of forecast months to include in visualizations
        """
        print("\n" + "="*70)
        print("STEP 2: DATA ANALYSIS")
        print("="*70)
        
        try:
            # Load data
            print("Loading data for analysis...")
            self.analyzer.load_data(include_current_month=True)
            
            # Create all visualizations
            print("Creating visualizations...")
            charts = self.analyzer.create_all_charts(forecast_months=forecast_months)
            
            print("\n[SUCCESS] Data analysis completed successfully")
            print("Available charts:")
            for chart_name in charts.keys():
                print(f"  - {chart_name}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Error during data analysis: {str(e)}")
            return False
    
    def train_models(self, forecast_months=1):
        """
        Train ML models and generate predictions
        
        Parameters:
        -----------
        forecast_months : int
            Number of months to predict (1-12)
        """
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING & PREDICTION")
        print("="*70)
        
        try:
            from pathlib import Path
            
            # Get backend directory
            backend_dir = Path(__file__).parent.parent
            train_script = backend_dir / 'ml_components' / 'lstm_cashflow_prediction.py'
            
            print(f"Training models for {forecast_months} month(s) prediction...")
            
            # Run training script from backend directory
            result = subprocess.run(
                [sys.executable, str(train_script), str(forecast_months)],
                cwd=str(backend_dir),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print("[SUCCESS] Model training completed successfully")
                return True
            else:
                print("[ERROR] Error during model training")
                return False
        except Exception as e:
            print(f"[ERROR] Error during model training: {str(e)}")
            return False
    
    def make_predictions(self, forecast_months=1, model_path=None):
        """
        Make predictions using saved model
        
        Parameters:
        -----------
        forecast_months : int
            Number of months to predict (1-12)
        model_path : str, optional
            Path to the saved model pkl file. If None, uses default location.
            
        Returns:
        --------
        bool : True if predictions generated successfully, False otherwise
        """
        try:
            import pickle
            import json
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            from pathlib import Path
            
            # Get backend directory
            backend_dir = Path(__file__).parent.parent
            
            # Determine model path
            if model_path is None:
                model_path = backend_dir / 'best_model.pkl'
            else:
                model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"ERROR: Model file not found: {model_path}")
                return False
            
            # Load model
            print(f"Loading model from {model_path}...")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            model_name = model_data.get('model_name', 'Unknown')
            model_type = model_data.get('model_type', 'Unknown')
            feature_scaler = model_data.get('feature_scaler')
            target_scaler = model_data.get('target_scaler')
            
            print(f"Loaded model: {model_name} ({model_type})")
            
            # Load model info
            model_info_path = backend_dir / 'best_model_info.json'
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                model_info = {}
            
            # Load and prepare data
            print("Loading and preparing data...")
            monthly_df = self.data_prep.load_and_aggregate(self.data_file)
            
            # Prepare features for training data (to get feature names)
            X, y, feature_names, feature_df = self.data_prep.prepare_data(
                monthly_df,
                use_pca=False,
                feature_selection_method='mutual_info',
                n_features=50,
                apply_correlation_filter=True
            )
            
            # Generate future dates
            last_date = pd.to_datetime(monthly_df['date'].max())
            future_dates = []
            for i in range(1, forecast_months + 1):
                next_month = last_date + pd.DateOffset(months=i)
                future_dates.append(next_month)
            
            # Prepare future features
            print(f"Generating predictions for {forecast_months} month(s)...")
            
            if model_type == 'prophet':
                # Prophet model prediction
                from prophet import Prophet
                
                # Create future dataframe for Prophet
                future_df = pd.DataFrame({
                    'ds': future_dates
                })
                
                # Add regressors if available
                regressors = model_info.get('regressors', [])
                regressor_scalers = model_data.get('regressor_scalers', {})
                
                lookback = min(12, len(monthly_df))
                for reg in regressors:
                    if reg in ['is_tax_month', 'is_quarter_end', 'is_year_end', 'month_sin', 'month_cos']:
                        # Calculate these for future dates
                        if reg == 'month_sin':
                            future_df[reg] = [np.sin(2 * np.pi * d.month / 12) for d in future_dates]
                        elif reg == 'month_cos':
                            future_df[reg] = [np.cos(2 * np.pi * d.month / 12) for d in future_dates]
                        elif reg == 'is_tax_month':
                            future_df[reg] = [1 if d.month == 4 else 0 for d in future_dates]
                        elif reg == 'is_quarter_end':
                            future_df[reg] = [1 if d.month in [3, 6, 9, 12] else 0 for d in future_dates]
                        elif reg == 'is_year_end':
                            future_df[reg] = [1 if d.month == 12 else 0 for d in future_dates]
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
                future_forecast = model.predict(future_df)
                predictions_scaled = future_forecast['yhat'].values
                predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
                predictions_lower = target_scaler.inverse_transform(future_forecast['yhat_lower'].values.reshape(-1, 1)).flatten()
                predictions_upper = target_scaler.inverse_transform(future_forecast['yhat_upper'].values.reshape(-1, 1)).flatten()
            else:
                # Boosting model prediction
                # Prepare future features similar to training
                future_features = []
                for i, future_date in enumerate(future_dates):
                    feat_row = []
                    for feat_name in feature_names:
                        # Handle lag features (format: {column}_lag{number}, e.g., cash_outflow_lag1)
                        if '_lag' in feat_name:
                            try:
                                # Split by '_lag' and extract number from second part
                                parts = feat_name.split('_lag')
                                if len(parts) == 2:
                                    # Extract number from parts[1] (could be just number or have more text)
                                    lag_str = parts[1]
                                    # Extract first sequence of digits
                                    import re
                                    lag_match = re.search(r'^(\d+)', lag_str)
                                    if lag_match:
                                        lag_num = int(lag_match.group(1))
                                        # Use cash_outflow_usd for lag features
                                        if lag_num <= len(monthly_df) and lag_num > 0:
                                            feat_row.append(monthly_df['cash_outflow_usd'].iloc[-lag_num])
                                        else:
                                            feat_row.append(0)
                                    else:
                                        # If can't extract number, use last value from feature_df
                                        if feat_name in feature_df.columns and len(feature_df) > 0:
                                            feat_row.append(feature_df[feat_name].iloc[-1])
                                        else:
                                            feat_row.append(0)
                                else:
                                    # If format doesn't match, use last value from feature_df
                                    if feat_name in feature_df.columns and len(feature_df) > 0:
                                        feat_row.append(feature_df[feat_name].iloc[-1])
                                    else:
                                        feat_row.append(0)
                            except (ValueError, IndexError, KeyError):
                                # If any error, use last value or default
                                if feat_name in feature_df.columns and len(feature_df) > 0:
                                    feat_row.append(feature_df[feat_name].iloc[-1])
                                else:
                                    feat_row.append(0)
                        # Handle rolling features (format: {column}_rolling_{stat}_{window}, e.g., cash_outflow_rolling_mean_3)
                        elif 'rolling' in feat_name.lower():
                            try:
                                # Extract window number from end (format: ..._rolling_mean_3 or ..._rolling_std_6)
                                import re
                                # Look for pattern: _rolling_{stat}_{number} or _rolling{number}
                                match = re.search(r'rolling[_\w]*_?(\d+)$', feat_name, re.IGNORECASE)
                                if match:
                                    window = int(match.group(1))
                                    if window <= len(monthly_df) and window > 0:
                                        # Use appropriate rolling statistic based on feature name
                                        if 'mean' in feat_name.lower():
                                            feat_row.append(monthly_df['cash_outflow_usd'].tail(window).mean())
                                        elif 'std' in feat_name.lower():
                                            feat_row.append(monthly_df['cash_outflow_usd'].tail(window).std() if window > 1 else 0)
                                        elif 'min' in feat_name.lower():
                                            feat_row.append(monthly_df['cash_outflow_usd'].tail(window).min())
                                        elif 'max' in feat_name.lower():
                                            feat_row.append(monthly_df['cash_outflow_usd'].tail(window).max())
                                        else:
                                            feat_row.append(monthly_df['cash_outflow_usd'].tail(window).mean())
                                    else:
                                        feat_row.append(monthly_df['cash_outflow_usd'].mean())
                                else:
                                    # If can't extract window, use last value or average
                                    if feat_name in feature_df.columns and len(feature_df) > 0:
                                        feat_row.append(feature_df[feat_name].iloc[-1])
                                    else:
                                        feat_row.append(monthly_df['cash_outflow_usd'].mean())
                            except (ValueError, IndexError, KeyError):
                                # If any error, use last value or default
                                if feat_name in feature_df.columns and len(feature_df) > 0:
                                    feat_row.append(feature_df[feat_name].iloc[-1])
                                else:
                                    feat_row.append(monthly_df['cash_outflow_usd'].mean())
                        elif feat_name == 'month':
                            feat_row.append(future_date.month)
                        elif feat_name == 'year':
                            feat_row.append(future_date.year)
                        elif feat_name == 'month_sin':
                            feat_row.append(np.sin(2 * np.pi * future_date.month / 12))
                        elif feat_name == 'month_cos':
                            feat_row.append(np.cos(2 * np.pi * future_date.month / 12))
                        elif feat_name == 'is_tax_month':
                            feat_row.append(1 if future_date.month == 4 else 0)
                        elif feat_name == 'is_quarter_end':
                            feat_row.append(1 if future_date.month in [3, 6, 9, 12] else 0)
                        elif feat_name == 'is_year_end':
                            feat_row.append(1 if future_date.month == 12 else 0)
                        else:
                            # Use average of last 12 months for other features
                            if feat_name in monthly_df.columns:
                                feat_row.append(monthly_df[feat_name].tail(12).mean())
                            else:
                                feat_row.append(0)
                    future_features.append(feat_row)
                
                future_X = np.array(future_features)
                future_X_scaled = feature_scaler.transform(future_X)
                
                # Make predictions
                predictions = model.predict(future_X_scaled)
                predictions = np.maximum(predictions, 0)  # Ensure non-negative
                
                # Simple confidence intervals
                if 'test_std' in model_data:
                    pred_std = model_data['test_std']
                else:
                    pred_std = predictions.std() * 0.1  # Default estimate
                
                predictions_lower = predictions - 1.96 * pred_std
                predictions_upper = predictions + 1.96 * pred_std
                predictions_lower = np.maximum(predictions_lower, 0)
                predictions_upper = np.maximum(predictions_upper, 0)
            
            # Create predictions dataframe
            predictions_list = []
            for i in range(forecast_months):
                month_str = future_dates[i].strftime('%Y-%m')
                predictions_list.append({
                    'month': month_str,
                    'predicted_cash_outflow': float(predictions[i]),
                    'lower_95': float(predictions_lower[i]),
                    'upper_95': float(predictions_upper[i])
                })
            
            # Save predictions
            predictions_df = pd.DataFrame(predictions_list)
            predictions_file = backend_dir / 'future_predictions.csv'
            predictions_df.to_csv(predictions_file, index=False)
            
            print(f"[SUCCESS] Predictions generated successfully")
            print(f"   Saved to: {predictions_file}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def explain_model(self):
        """Generate SHAP explainability analysis"""
        print("\n" + "="*70)
        print("STEP 4: MODEL EXPLAINABILITY (SHAP)")
        print("="*70)
        
        try:
            import os
            from pathlib import Path
            
            # Check if model exists in backend directory
            backend_dir = Path(__file__).parent.parent
            model_path = backend_dir / 'best_model.pkl'
            
            if not model_path.exists():
                print("[WARNING] best_model.pkl not found.")
                print("   Please run model training first (python app.py train)")
                return False
            
            # Run SHAP analysis (from backend directory)
            print("Generating SHAP explainability analysis...")
            shap_script = backend_dir / 'ml_components' / 'shap_explainability.py'
            result = subprocess.run(
                [sys.executable, str(shap_script)],
                cwd=str(backend_dir),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print("[SUCCESS] SHAP analysis completed successfully")
                print("   Results saved to: shap_plots/")
                return True
            else:
                print("[ERROR] Error during SHAP analysis")
                return False
        except Exception as e:
            print(f"[ERROR] Error during SHAP analysis: {str(e)}")
            return False
    
    def run_full_pipeline(self, forecast_months=1):
        """
        Run the complete ML pipeline
        
        Parameters:
        -----------
        forecast_months : int
            Number of months to predict (1-12)
        """
        print("\n" + "="*70)
        print("CASH FLOW ML SYSTEM - FULL PIPELINE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        results = {
            'generate': False,
            'analyze': False,
            'train': False,
            'explain': False
        }
        
        # Step 1: Generate data
        results['generate'] = self.generate_data()
        if not results['generate']:
            print("\n[WARNING] Data generation failed, but continuing...")
        
        # Step 2: Analyze data
        results['analyze'] = self.analyze_data(forecast_months=forecast_months)
        if not results['analyze']:
            print("\n[WARNING] Data analysis failed, but continuing...")
        
        # Step 3: Train models
        results['train'] = self.train_models(forecast_months=forecast_months)
        if not results['train']:
            print("\n[ERROR] Model training failed. Cannot proceed with explainability.")
            return results
        
        # Step 4: Explain model
        results['explain'] = self.explain_model()
        if not results['explain']:
            print("\n[WARNING] SHAP analysis failed, but pipeline completed.")
        
        # Summary
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"Data Generation:     {'[SUCCESS]' if results['generate'] else '[FAILED]'}")
        print(f"Data Analysis:      {'[SUCCESS]' if results['analyze'] else '[FAILED]'}")
        print(f"Model Training:     {'[SUCCESS]' if results['train'] else '[FAILED]'}")
        print(f"SHAP Explainability: {'[SUCCESS]' if results['explain'] else '[FAILED]'}")
        print("="*70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Cash Flow Prediction ML System - Integration Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py generate              # Generate/update data
  python app.py analyze               # Perform data analysis
  python app.py train 6               # Train models for 6 months prediction
  python app.py explain               # Generate SHAP analysis
  python app.py full 3                # Run full pipeline for 3 months prediction
        """
    )
    
    parser.add_argument(
        'command',
        choices=['generate', 'analyze', 'train', 'explain', 'full'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'forecast_months',
        type=int,
        nargs='?',
        default=1,
        help='Number of months to predict (1-12, default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validate forecast_months
    if args.forecast_months < 1 or args.forecast_months > 12:
        print(f"Error: forecast_months must be between 1 and 12, got {args.forecast_months}")
        sys.exit(1)
    
    # Initialize system
    system = CashFlowMLSystem()
    
    # Execute command
    if args.command == 'generate':
        success = system.generate_data()
        sys.exit(0 if success else 1)
    
    elif args.command == 'analyze':
        success = system.analyze_data(forecast_months=args.forecast_months)
        sys.exit(0 if success else 1)
    
    elif args.command == 'train':
        success = system.train_models(forecast_months=args.forecast_months)
        sys.exit(0 if success else 1)
    
    elif args.command == 'explain':
        success = system.explain_model()
        sys.exit(0 if success else 1)
    
    elif args.command == 'full':
        results = system.run_full_pipeline(forecast_months=args.forecast_months)
        # Exit with error if critical steps failed
        if not results['train']:
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
