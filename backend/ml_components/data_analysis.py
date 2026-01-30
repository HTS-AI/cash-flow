"""
Data Analysis and Visualization Module for Cash Flow Prediction
Provides comprehensive visualizations of cash flow data

Usage:
    from data_analysis import CashFlowAnalyzer
    analyzer = CashFlowAnalyzer()
    analyzer.create_all_charts()
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from data_preparation import DataPreparator
import os
import json


class CashFlowAnalyzer:
    """Comprehensive cash flow data analysis and visualization"""
    
    def __init__(self, data_file="cashflow_prediction_1998_2025_v1.csv"):
        """
        Initialize CashFlowAnalyzer
        
        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing cash flow data
        """
        self.data_file = data_file
        self.data_prep = DataPreparator(
            correlation_threshold=0.95,
            pca_variance_threshold=0.95
        )
        self.daily_df = None
        self.monthly_df = None
        self.predictions_df = None
        
    def load_data(self, include_current_month=True):
        """
        Load and prepare data for analysis
        
        Parameters:
        -----------
        include_current_month : bool
            If True, includes data up to end of current month (default: True)
            If False, only includes complete months
        """
        # Load daily data
        if os.path.exists(self.data_file):
            self.daily_df = pd.read_csv(self.data_file, parse_dates=["date"])
            self.daily_df = self.daily_df.sort_values("date").reset_index(drop=True)
        else:
            raise FileNotFoundError(f"Data file {self.data_file} not found")
        
        # Load monthly aggregated data
        self.monthly_df = self.data_prep.load_and_aggregate(self.data_file)
        
        # If include_current_month is True, add current month data (even if incomplete)
        if include_current_month:
            today = pd.Timestamp.today()
            current_month_start = today.replace(day=1)
            
            # Get current month daily data
            current_month_daily = self.daily_df[
                self.daily_df['date'] >= current_month_start
            ].copy()
            
            if len(current_month_daily) > 0:
                # Create period column for grouping
                current_month_daily['year_month'] = current_month_daily['date'].dt.to_period('M')
                
                # Aggregate current month
                current_month_agg = current_month_daily.groupby('year_month').agg({
                    'cash_outflow_usd': 'sum',
                    'cash_inflow_usd': 'sum',
                    'vendor_payment_usd': 'sum',
                    'salary_payment_usd': 'sum',
                    'rent_usd': 'sum',
                    'operational_expense_usd': 'sum',
                    'interest_rate_pct': 'mean',
                    'inflation_pct': 'mean',
                    'economic_sentiment_score': 'mean',
                }).reset_index()
                
                # Convert period to timestamp
                current_month_agg['date'] = current_month_agg['year_month'].dt.to_timestamp()
                current_month_agg = current_month_agg.drop('year_month', axis=1)
                
                # Append to monthly_df if not already there
                last_monthly_date = self.monthly_df['date'].max()
                if current_month_agg['date'].iloc[0] > last_monthly_date:
                    self.monthly_df = pd.concat([self.monthly_df, current_month_agg], ignore_index=True)
                    self.monthly_df = self.monthly_df.sort_values('date').reset_index(drop=True)
        
        # Load predictions if available
        if os.path.exists('future_predictions.csv'):
            try:
                self.predictions_df = pd.read_csv('future_predictions.csv')
                self.predictions_df['date'] = pd.to_datetime(self.predictions_df['month'])
            except:
                self.predictions_df = None
        else:
            self.predictions_df = None
    
    def filter_by_date_range(self, start_date=None, end_date=None, year=None, start_month=None, end_month=None):
        """
        Filter monthly data by date range or year/month range
        
        Parameters:
        -----------
        start_date : str or pd.Timestamp, optional
            Start date for filtering (YYYY-MM-DD format)
        end_date : str or pd.Timestamp, optional
            End date for filtering (YYYY-MM-DD format)
        year : int, optional
            Specific year to filter
        start_month : int, optional
            Start month (1-12) for year filtering
        end_month : int, optional
            End month (1-12) for year filtering
            
        Returns:
        --------
        filtered_df : pd.DataFrame
            Filtered monthly dataframe
        """
        if self.monthly_df is None or len(self.monthly_df) == 0:
            return self.monthly_df
        
        filtered_df = self.monthly_df.copy()
        
        # Filter by year and month range
        if year is not None:
            filtered_df = filtered_df[filtered_df['date'].dt.year == year]
            
            if start_month is not None:
                filtered_df = filtered_df[filtered_df['date'].dt.month >= start_month]
            
            if end_month is not None:
                filtered_df = filtered_df[filtered_df['date'].dt.month <= end_month]
        
        # Filter by date range (overrides year/month if both provided)
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.Timestamp(start_date)
            filtered_df = filtered_df[filtered_df['date'] >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.Timestamp(end_date)
            filtered_df = filtered_df[filtered_df['date'] <= end_date]
        
        return filtered_df.sort_values('date').reset_index(drop=True)
    
    def create_time_series_chart(self, forecast_months=0):
        """
        Create time series chart showing historical and predicted cash outflow
        
        Parameters:
        -----------
        forecast_months : int
            Number of forecast months to display (0 = no forecast)
            
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Time series chart
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=self.monthly_df['date'],
            y=self.monthly_df['cash_outflow_usd'],
            mode='lines+markers',
            name='Historical Cash Outflow',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Add predictions if available and requested
        if self.predictions_df is not None and forecast_months > 0:
            pred_subset = self.predictions_df.head(forecast_months).copy()
            
            # Predicted values
            fig.add_trace(go.Scatter(
                x=pred_subset['date'],
                y=pred_subset['predicted_cash_outflow'],
                mode='lines+markers',
                name='Predicted Cash Outflow',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=pred_subset['date'],
                y=pred_subset['upper_95'],
                mode='lines',
                name='Upper 95% CI',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=pred_subset['date'],
                y=pred_subset['lower_95'],
                mode='lines',
                name='95% Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(width=0)
            ))
        
        fig.update_layout(
            title='Cash Outflow Over Time',
            xaxis_title='Date',
            yaxis_title='Cash Outflow (USD)',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def create_inflow_outflow_comparison(self):
        """
        Create comparison chart of cash inflow vs outflow
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Comparison chart
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.monthly_df['date'],
            y=self.monthly_df['cash_inflow_usd'],
            mode='lines+markers',
            name='Cash Inflow',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.monthly_df['date'],
            y=self.monthly_df['cash_outflow_usd'],
            mode='lines+markers',
            name='Cash Outflow',
            line=dict(color='#d62728', width=2),
            marker=dict(size=4)
        ))
        
        # Net cashflow
        net_cashflow = self.monthly_df['cash_inflow_usd'] - self.monthly_df['cash_outflow_usd']
        fig.add_trace(go.Scatter(
            x=self.monthly_df['date'],
            y=net_cashflow,
            mode='lines+markers',
            name='Net Cashflow',
            line=dict(color='#9467bd', width=2),
            marker=dict(size=4)
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='Cash Inflow vs Outflow Comparison',
            xaxis_title='Date',
            yaxis_title='Amount (USD)',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def create_expense_breakdown(self):
        """
        Create pie chart and bar chart showing expense breakdown
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Subplot with pie and bar charts
        """
        # Calculate totals for last 12 months
        recent_df = self.monthly_df.tail(12)
        
        expense_totals = {
            'Vendor Payment': recent_df['vendor_payment_usd'].sum(),
            'Salary Payment': recent_df['salary_payment_usd'].sum(),
            'Rent': recent_df['rent_usd'].sum(),
            'Operational Expense': recent_df['operational_expense_usd'].sum()
        }
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Expense Distribution (Last 12 Months)", "Expense Comparison")
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=list(expense_totals.keys()),
                values=list(expense_totals.values()),
                name="Expenses",
                hole=0.4
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=list(expense_totals.keys()),
                y=list(expense_totals.values()),
                name="Expenses",
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Expense Breakdown Analysis',
            height=500,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Expense Type", row=1, col=2)
        fig.update_yaxes(title_text="Amount (USD)", row=1, col=2)
        
        return fig
    
    def create_seasonal_analysis(self):
        """
        Create seasonal analysis chart showing patterns by month
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Seasonal analysis chart
        """
        # Add month name
        monthly_with_month = self.monthly_df.copy()
        monthly_with_month['month_name'] = monthly_with_month['date'].dt.strftime('%B')
        monthly_with_month['month_num'] = monthly_with_month['date'].dt.month
        
        # Group by month
        seasonal = monthly_with_month.groupby('month_num').agg({
            'cash_outflow_usd': ['mean', 'std', 'min', 'max'],
            'cash_inflow_usd': 'mean'
        }).reset_index()
        
        seasonal.columns = ['month', 'outflow_mean', 'outflow_std', 'outflow_min', 'outflow_max', 'inflow_mean']
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal['month_name'] = [month_names[i-1] for i in seasonal['month']]
        
        fig = go.Figure()
        
        # Mean outflow
        fig.add_trace(go.Scatter(
            x=seasonal['month_name'],
            y=seasonal['outflow_mean'],
            mode='lines+markers',
            name='Average Outflow',
            line=dict(color='#d62728', width=3),
            marker=dict(size=8)
        ))
        
        # Mean inflow
        fig.add_trace(go.Scatter(
            x=seasonal['month_name'],
            y=seasonal['inflow_mean'],
            mode='lines+markers',
            name='Average Inflow',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        # Error bars for outflow
        fig.add_trace(go.Scatter(
            x=seasonal['month_name'],
            y=seasonal['outflow_mean'] + seasonal['outflow_std'],
            mode='lines',
            name='Outflow +1 Std',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=seasonal['month_name'],
            y=seasonal['outflow_mean'] - seasonal['outflow_std'],
            mode='lines',
            name='Outflow Std Dev',
            fill='tonexty',
            fillcolor='rgba(214, 39, 40, 0.2)',
            line=dict(width=0)
        ))
        
        fig.update_layout(
            title='Seasonal Cash Flow Patterns (Average by Month)',
            xaxis_title='Month',
            yaxis_title='Amount (USD)',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def create_trend_analysis(self):
        """
        Create trend analysis with moving averages
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Trend analysis chart
        """
        monthly_trend = self.monthly_df.copy()
        
        # Calculate moving averages
        monthly_trend['ma_3'] = monthly_trend['cash_outflow_usd'].rolling(window=3, min_periods=1).mean()
        monthly_trend['ma_6'] = monthly_trend['cash_outflow_usd'].rolling(window=6, min_periods=1).mean()
        monthly_trend['ma_12'] = monthly_trend['cash_outflow_usd'].rolling(window=12, min_periods=1).mean()
        
        fig = go.Figure()
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=monthly_trend['date'],
            y=monthly_trend['cash_outflow_usd'],
            mode='lines+markers',
            name='Actual Cash Outflow',
            line=dict(color='#1f77b4', width=1),
            marker=dict(size=3),
            opacity=0.6
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=monthly_trend['date'],
            y=monthly_trend['ma_3'],
            mode='lines',
            name='3-Month MA',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_trend['date'],
            y=monthly_trend['ma_6'],
            mode='lines',
            name='6-Month MA',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=monthly_trend['date'],
            y=monthly_trend['ma_12'],
            mode='lines',
            name='12-Month MA',
            line=dict(color='#d62728', width=2)
        ))
        
        fig.update_layout(
            title='Cash Outflow Trend Analysis with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Cash Outflow (USD)',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    def create_economic_indicators(self):
        """
        Create chart showing economic indicators
        
        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Economic indicators chart
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Interest Rate", "Inflation Rate", "Economic Sentiment"),
            vertical_spacing=0.1
        )
        
        # Interest rate
        fig.add_trace(
            go.Scatter(
                x=self.monthly_df['date'],
                y=self.monthly_df['interest_rate_pct'],
                mode='lines+markers',
                name='Interest Rate',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Inflation
        fig.add_trace(
            go.Scatter(
                x=self.monthly_df['date'],
                y=self.monthly_df['inflation_pct'],
                mode='lines+markers',
                name='Inflation Rate',
                line=dict(color='#ff7f0e', width=2)
            ),
            row=2, col=1
        )
        
        # Economic sentiment
        fig.add_trace(
            go.Scatter(
                x=self.monthly_df['date'],
                y=self.monthly_df['economic_sentiment_score'],
                mode='lines+markers',
                name='Economic Sentiment',
                line=dict(color='#2ca02c', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Interest Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Inflation Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sentiment Score", row=3, col=1)
        
        fig.update_layout(
            title='Economic Indicators Over Time',
            height=700,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_summary_statistics(self):
        """
        Create summary statistics table
        
        Returns:
        --------
        stats_df : pd.DataFrame
            Summary statistics dataframe
        """
        stats = {
            'Metric': [
                'Total Months',
                'Average Monthly Outflow',
                'Average Monthly Inflow',
                'Average Net Cashflow',
                'Max Monthly Outflow',
                'Min Monthly Outflow',
                'Std Dev Outflow',
                'Total Outflow',
                'Total Inflow',
                'Data Range Start',
                'Data Range End'
            ],
            'Value': [
                str(len(self.monthly_df)),  # Convert to string
                f"${self.monthly_df['cash_outflow_usd'].mean():,.0f}",
                f"${self.monthly_df['cash_inflow_usd'].mean():,.0f}",
                f"${(self.monthly_df['cash_inflow_usd'] - self.monthly_df['cash_outflow_usd']).mean():,.0f}",
                f"${self.monthly_df['cash_outflow_usd'].max():,.0f}",
                f"${self.monthly_df['cash_outflow_usd'].min():,.0f}",
                f"${self.monthly_df['cash_outflow_usd'].std():,.0f}",
                f"${self.monthly_df['cash_outflow_usd'].sum():,.0f}",
                f"${self.monthly_df['cash_inflow_usd'].sum():,.0f}",
                self.monthly_df['date'].min().strftime('%Y-%m-%d'),
                self.monthly_df['date'].max().strftime('%Y-%m-%d')
            ]
        }
        
        # Ensure all values are strings to avoid Arrow conversion issues
        stats_df = pd.DataFrame(stats)
        stats_df['Value'] = stats_df['Value'].astype(str)
        
        return stats_df
    
    def create_all_charts(self, forecast_months=0, filtered_df=None):
        """
        Create all visualization charts
        
        Parameters:
        -----------
        forecast_months : int
            Number of forecast months to include in predictions (default: 0)
        filtered_df : pd.DataFrame, optional
            Filtered dataframe to use instead of self.monthly_df
            
        Returns:
        --------
        charts : dict
            Dictionary containing all chart figures
        """
        # Use filtered dataframe if provided, otherwise use full data
        original_df = self.monthly_df
        if filtered_df is not None and len(filtered_df) > 0:
            self.monthly_df = filtered_df
        
        try:
            charts = {
                'time_series': self.create_time_series_chart(forecast_months),
                'inflow_outflow': self.create_inflow_outflow_comparison(),
                'expense_breakdown': self.create_expense_breakdown(),
                'seasonal': self.create_seasonal_analysis(),
                'trend': self.create_trend_analysis(),
                'economic_indicators': self.create_economic_indicators(),
                'summary_stats': self.create_summary_statistics()
            }
        finally:
            # Restore original dataframe
            self.monthly_df = original_df
        
        return charts
