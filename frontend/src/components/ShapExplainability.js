import React, { useState, useEffect } from 'react';
import { cashflowAPI } from '../services/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { FiInfo, FiRefreshCw, FiRotateCw } from 'react-icons/fi';

// Feature descriptions mapping - explains what each preprocessed feature means
const featureDescriptions = {
  // EMA (Exponential Moving Average) features
  'cash_outflow_ema_3': {
    name: 'Cash Outflow EMA-3',
    description: '3-period Exponential Moving Average of expenses',
    detail: 'Smooths out expense fluctuations, giving more weight to recent values. Helps identify expense trends.',
    category: 'Trend'
  },
  'cash_outflow_ema_6': {
    name: 'Cash Outflow EMA-6',
    description: '6-period Exponential Moving Average of expenses',
    detail: 'Longer-term smoothed expense trend for identifying medium-term patterns.',
    category: 'Trend'
  },
  'cash_inflow_ema_3': {
    name: 'Cash Inflow EMA-3',
    description: '3-period Exponential Moving Average of income',
    detail: 'Smooths income fluctuations to show short-term income trends.',
    category: 'Trend'
  },
  
  // MoM (Month-over-Month) features
  'cash_outflow_mom': {
    name: 'Cash Outflow MoM',
    description: 'Month-over-Month % change in expenses',
    detail: 'Shows how much expenses increased or decreased compared to the previous month.',
    category: 'Change'
  },
  'cash_inflow_mom': {
    name: 'Cash Inflow MoM',
    description: 'Month-over-Month % change in income',
    detail: 'Shows how much income increased or decreased compared to the previous month.',
    category: 'Change'
  },
  
  // YoY (Year-over-Year) features
  'cash_outflow_yoy': {
    name: 'Cash Outflow YoY',
    description: 'Year-over-Year % change in expenses',
    detail: 'Compares expenses to the same month last year, accounting for seasonality.',
    category: 'Change'
  },
  'cash_inflow_yoy': {
    name: 'Cash Inflow YoY',
    description: 'Year-over-Year % change in income',
    detail: 'Compares income to the same month last year.',
    category: 'Change'
  },
  
  // Ratio features
  'salary_payment_ratio': {
    name: 'Salary Payment Ratio',
    description: 'Salary payments as % of total expenses',
    detail: 'Shows what portion of expenses goes to salaries. High ratios indicate labor-intensive operations.',
    category: 'Ratio'
  },
  'vendor_payment_ratio': {
    name: 'Vendor Payment Ratio',
    description: 'Vendor payments as % of total expenses',
    detail: 'Shows what portion of expenses goes to vendors/suppliers.',
    category: 'Ratio'
  },
  'rent_ratio': {
    name: 'Rent Ratio',
    description: 'Rent as % of total expenses',
    detail: 'Shows what portion of expenses goes to rent/facilities.',
    category: 'Ratio'
  },
  'operational_expense_ratio': {
    name: 'Operational Expense Ratio',
    description: 'Operational costs as % of total expenses',
    detail: 'Shows what portion goes to day-to-day operational costs.',
    category: 'Ratio'
  },
  'inflow_outflow_ratio': {
    name: 'Inflow/Outflow Ratio',
    description: 'Income to Expense ratio',
    detail: 'Values > 1 mean profit, < 1 mean loss. Key indicator of financial health.',
    category: 'Ratio'
  },
  
  // Lag features
  'cash_outflow_lag1': {
    name: 'Cash Outflow Lag-1',
    description: 'Expenses from 1 month ago',
    detail: 'Previous month expenses used to predict current/future expenses.',
    category: 'Historical'
  },
  'cash_outflow_lag2': {
    name: 'Cash Outflow Lag-2',
    description: 'Expenses from 2 months ago',
    detail: 'Expenses from 2 months prior for pattern recognition.',
    category: 'Historical'
  },
  'cash_outflow_lag3': {
    name: 'Cash Outflow Lag-3',
    description: 'Expenses from 3 months ago',
    detail: 'Quarterly historical reference point.',
    category: 'Historical'
  },
  
  // Rolling statistics
  'cash_outflow_rolling_mean_3': {
    name: 'Rolling Mean (3-month)',
    description: 'Average expenses over last 3 months',
    detail: 'Simple moving average showing short-term expense baseline.',
    category: 'Statistics'
  },
  'cash_outflow_rolling_mean_6': {
    name: 'Rolling Mean (6-month)',
    description: 'Average expenses over last 6 months',
    detail: 'Medium-term expense baseline.',
    category: 'Statistics'
  },
  'cash_outflow_rolling_std_3': {
    name: 'Rolling Std Dev (3-month)',
    description: 'Expense volatility over last 3 months',
    detail: 'High values indicate unpredictable expenses.',
    category: 'Statistics'
  },
  'cash_outflow_rolling_std_6': {
    name: 'Rolling Std Dev (6-month)',
    description: 'Expense volatility over last 6 months',
    detail: 'Medium-term expense stability measure.',
    category: 'Statistics'
  },
  'cash_outflow_rolling_min_3': {
    name: 'Rolling Min (3-month)',
    description: 'Minimum expenses in last 3 months',
    detail: 'Shows the lowest expense level recently achieved.',
    category: 'Statistics'
  },
  'cash_outflow_rolling_max_3': {
    name: 'Rolling Max (3-month)',
    description: 'Maximum expenses in last 3 months',
    detail: 'Shows the highest expense level recently reached.',
    category: 'Statistics'
  },
  'cash_outflow_rolling_max_6': {
    name: 'Rolling Max (6-month)',
    description: 'Maximum expenses in last 6 months',
    detail: 'Peak expense level in the medium term.',
    category: 'Statistics'
  },
};

// Category colors
const categoryColors = {
  'Trend': '#3b82f6',
  'Change': '#10b981',
  'Ratio': '#f59e0b',
  'Historical': '#8b5cf6',
  'Statistics': '#ec4899',
};

const ShapExplainability = () => {
  const [loading, setLoading] = useState(true);
  const [featureData, setFeatureData] = useState(null);
  const [error, setError] = useState(null);
  const [isFlipped, setIsFlipped] = useState(false);

  useEffect(() => {
    loadFeatureImportance();
  }, []);

  const loadFeatureImportance = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await cashflowAPI.getFeatureImportance();
      if (response.data.success) {
        setFeatureData(response.data.data);
      } else {
        setError(response.data.message || 'Failed to load feature importance');
      }
    } catch (err) {
      setError(err.message || 'Failed to load feature importance');
    } finally {
      setLoading(false);
    }
  };

  const getFeatureDescription = (featureName) => {
    return featureDescriptions[featureName] || {
      name: featureName,
      description: 'Engineered feature for prediction',
      detail: 'This feature is derived from source data to help the model identify patterns.',
      category: 'Other'
    };
  };

  if (loading) {
    return (
      <div className="chart-card">
        <div className="chart-title">Model Explainability (Feature Importance)</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <div className="spinner" style={{ margin: '0 auto' }}></div>
          Loading feature importance...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="chart-card">
        <div className="chart-title">Model Explainability (Feature Importance)</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#f59e0b' }}>
          <FiInfo size={24} style={{ marginBottom: '0.5rem' }} />
          <p>{error}</p>
          <button
            onClick={loadFeatureImportance}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
            }}
          >
            <FiRefreshCw /> Retry
          </button>
        </div>
      </div>
    );
  }

  if (!featureData || !featureData.features || featureData.features.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">Model Explainability (Feature Importance)</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <FiInfo size={24} style={{ marginBottom: '0.5rem' }} />
          <p>No feature importance data available.</p>
          <p style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
            Train a model to see which features are most important for predictions.
          </p>
        </div>
      </div>
    );
  }

  // Prepare chart data
  const maxImportance = Math.max(...featureData.features.map(f => f.importance));
  const chartData = featureData.features.map(f => {
    const desc = getFeatureDescription(f.name);
    return {
      name: f.name.length > 20 ? f.name.substring(0, 20) + '...' : f.name,
      fullName: f.name,
      importance: f.importance,
      percentage: maxImportance > 0 ? (f.importance / maxImportance * 100) : 0,
      description: desc.description,
      category: desc.category,
    };
  });

  // Colors for bars
  const getBarColor = (index) => {
    const colors = [
      '#10b981', '#22c55e', '#84cc16', '#eab308', '#f59e0b',
      '#f97316', '#ef4444', '#dc2626', '#b91c1c', '#991b1b',
      '#7f1d1d', '#6b7280', '#4b5563', '#374151', '#1f2937'
    ];
    return colors[Math.min(index, colors.length - 1)];
  };

  return (
    <div 
      className={`shap-flip-card ${isFlipped ? 'flipped' : ''}`}
      style={{
        perspective: '1000px',
        minHeight: '650px',
      }}
    >
      <div 
        className="shap-flip-card-inner"
        style={{
          position: 'relative',
          width: '100%',
          height: '100%',
          transition: 'transform 0.6s',
          transformStyle: 'preserve-3d',
          transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
        }}
      >
        {/* Front - Chart */}
        <div 
          className="chart-card"
          style={{
            position: isFlipped ? 'absolute' : 'relative',
            width: '100%',
            backfaceVisibility: 'hidden',
            opacity: isFlipped ? 0 : 1,
            transition: 'opacity 0.3s',
          }}
        >
          <div className="chart-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Model Explainability (Feature Importance)</span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                onClick={() => setIsFlipped(true)}
                style={{
                  padding: '0.25rem 0.5rem',
                  backgroundColor: '#3b82f6',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  fontSize: '0.75rem',
                }}
              >
                <FiRotateCw size={12} /> View Feature Descriptions
              </button>
              <button
                onClick={loadFeatureImportance}
                style={{
                  padding: '0.25rem 0.5rem',
                  backgroundColor: 'transparent',
                  color: '#94a3b8',
                  border: '1px solid #334155',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.25rem',
                  fontSize: '0.75rem',
                }}
              >
                <FiRefreshCw size={12} /> Refresh
              </button>
            </div>
          </div>
          
          {/* Model info */}
          <div style={{ 
            marginBottom: '1rem', 
            padding: '0.75rem', 
            backgroundColor: '#1e293b', 
            borderRadius: '8px',
            fontSize: '0.875rem',
            color: '#94a3b8',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '1rem',
            justifyContent: 'center'
          }}>
            <span><strong>Model:</strong> {featureData.modelName}</span>
            <span><strong>Type:</strong> {featureData.modelType}</span>
            {featureData.totalFeatures && (
              <span><strong>Total Features:</strong> {featureData.totalFeatures}</span>
            )}
          </div>

          {featureData.message ? (
            <div style={{ padding: '1rem', textAlign: 'center', color: '#94a3b8' }}>
              <p>{featureData.message}</p>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  type="number"
                  stroke="#94a3b8"
                  style={{ fontSize: '11px' }}
                  tickFormatter={(value) => `${value.toFixed(1)}%`}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  stroke="#94a3b8"
                  style={{ fontSize: '11px' }}
                  width={100}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '2px solid #334155',
                    borderRadius: '12px',
                    color: '#e2e8f0',
                    padding: '12px',
                  }}
                  formatter={(value, name, props) => [
                    <div key="tooltip">
                      <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                        {props.payload.fullName}
                      </div>
                      <div style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '4px' }}>
                        {props.payload.description}
                      </div>
                      <div style={{ color: '#10b981' }}>
                        Importance: {props.payload.importance.toFixed(4)} ({value.toFixed(1)}%)
                      </div>
                    </div>,
                    ''
                  ]}
                  labelFormatter={() => ''}
                />
                <Bar dataKey="percentage" name="Feature Importance">
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getBarColor(index)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}

          {/* Explanation */}
          <div style={{ 
            marginTop: '1rem', 
            padding: '0.75rem', 
            backgroundColor: '#1e293b', 
            borderRadius: '8px',
            fontSize: '0.8rem',
            color: '#94a3b8'
          }}>
            <p style={{ margin: 0 }}>
              <strong>How to interpret:</strong> Feature importance shows which factors have the most influence on predictions. 
              Click "View Feature Descriptions" to understand what each feature means.
            </p>
          </div>
        </div>

        {/* Back - Feature Descriptions */}
        <div 
          className="chart-card"
          style={{
            position: isFlipped ? 'relative' : 'absolute',
            width: '100%',
            backfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)',
            opacity: isFlipped ? 1 : 0,
            transition: 'opacity 0.3s',
          }}
        >
          <div className="chart-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Preprocessed Features Dictionary</span>
            <button
              onClick={() => setIsFlipped(false)}
              style={{
                padding: '0.25rem 0.5rem',
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.25rem',
                fontSize: '0.75rem',
              }}
            >
              <FiRotateCw size={12} /> Back to Chart
            </button>
          </div>

          {/* Category Legend */}
          <div style={{ 
            display: 'flex', 
            gap: '1rem', 
            flexWrap: 'wrap', 
            marginBottom: '1rem',
            padding: '0.75rem',
            backgroundColor: '#1e293b',
            borderRadius: '8px',
          }}>
            {Object.entries(categoryColors).map(([category, color]) => (
              <div key={category} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ 
                  width: '12px', 
                  height: '12px', 
                  backgroundColor: color, 
                  borderRadius: '3px' 
                }} />
                <span style={{ color: '#94a3b8', fontSize: '0.75rem' }}>{category}</span>
              </div>
            ))}
          </div>

          {/* Feature Descriptions List */}
          <div style={{ 
            maxHeight: '480px', 
            overflowY: 'auto',
            paddingRight: '0.5rem',
          }}>
            {featureData.features.map((feature, index) => {
              const desc = getFeatureDescription(feature.name);
              return (
                <div 
                  key={feature.name}
                  style={{
                    padding: '0.75rem',
                    backgroundColor: index % 2 === 0 ? '#1e293b' : '#0f172a',
                    borderRadius: '8px',
                    marginBottom: '0.5rem',
                    borderLeft: `4px solid ${categoryColors[desc.category] || '#6b7280'}`,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.25rem' }}>
                    <div style={{ color: '#e2e8f0', fontWeight: 'bold', fontSize: '0.9rem' }}>
                      {feature.name}
                    </div>
                    <div style={{ 
                      color: categoryColors[desc.category] || '#6b7280',
                      fontSize: '0.7rem',
                      padding: '0.15rem 0.5rem',
                      backgroundColor: '#0f172a',
                      borderRadius: '4px',
                    }}>
                      {desc.category}
                    </div>
                  </div>
                  <div style={{ color: '#10b981', fontSize: '0.85rem', marginBottom: '0.25rem' }}>
                    {desc.description}
                  </div>
                  <div style={{ color: '#64748b', fontSize: '0.8rem' }}>
                    {desc.detail}
                  </div>
                  <div style={{ 
                    color: '#f59e0b', 
                    fontSize: '0.75rem', 
                    marginTop: '0.25rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}>
                    <span>Importance:</span>
                    <span style={{ fontWeight: 'bold' }}>{feature.importance.toFixed(4)}</span>
                    <span style={{ color: '#94a3b8' }}>
                      ({(feature.importance / maxImportance * 100).toFixed(1)}%)
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Source vs Preprocessed Explanation */}
          <div style={{ 
            marginTop: '1rem', 
            padding: '0.75rem', 
            backgroundColor: '#1e293b', 
            borderRadius: '8px',
            fontSize: '0.8rem',
            color: '#94a3b8'
          }}>
            <p style={{ margin: 0 }}>
              <strong>Why preprocessed features?</strong> The ML model uses engineered features (like moving averages, ratios, and lag values) 
              derived from source data to identify patterns and trends. The Sample Data section shows the original source data.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ShapExplainability;
