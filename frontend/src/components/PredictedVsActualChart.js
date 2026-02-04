import React, { useState, useEffect } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';
import { cashflowAPI } from '../services/api';
import { FiTrendingUp, FiTarget, FiPercent, FiCheckCircle, FiRefreshCw } from 'react-icons/fi';
import FlipCard from './FlipCard';

const PredictedVsActualChart = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMonths, setSelectedMonths] = useState(12);

  useEffect(() => {
    loadData(selectedMonths);
  }, [selectedMonths]);

  const loadData = async (months) => {
    try {
      setLoading(true);
      setError(null);
      const response = await cashflowAPI.getPredictedVsActual(months);
      if (response.data.success) {
        setData(response.data.data);
      } else {
        setError(response.data.message || 'Failed to load data');
      }
    } catch (err) {
      setError(err.message || 'Failed to load predicted vs actual data');
    } finally {
      setLoading(false);
    }
  };

  const handleMonthsChange = (months) => {
    setSelectedMonths(months);
  };

  // Format month labels
  const formatMonth = (month) => {
    if (!month) return '';
    const parts = month.split('-');
    if (parts.length < 2) return month;
    const [year, monthNum] = parts;
    const date = new Date(parseInt(year), parseInt(monthNum) - 1);
    return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
  };

  // Format currency
  const formatCurrency = (value) => {
    if (value === null || value === undefined) return '-';
    const absValue = Math.abs(value);
    if (absValue >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`;
    } else if (absValue >= 1000) {
      return `$${(value / 1000).toFixed(1)}K`;
    }
    return `$${value.toFixed(0)}`;
  };

  if (loading) {
    return (
      <div className="chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="chart-title">Predicted vs Actual Cash Outflow</div>
        <div className="loading" style={{ minHeight: '300px' }}>
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="chart-title">Predicted vs Actual Cash Outflow</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!data || !data.comparison || data.comparison.length === 0) {
    return (
      <div className="chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="chart-title">Predicted vs Actual Cash Outflow</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No comparison data available. Please train the model first.
        </div>
      </div>
    );
  }

  const { comparison, metrics, modelName } = data;

  // Prepare chart data with error calculations
  const chartData = comparison.map((item) => {
    const diff = item.predicted !== null ? item.actual - item.predicted : null;
    const percentErr = item.predicted !== null && item.actual !== 0 
      ? ((item.actual - item.predicted) / item.actual * 100) 
      : null;
    return {
      name: formatMonth(item.month),
      month: item.month,
      actual: item.actual,
      predicted: item.predicted,
      difference: diff,
      percentError: percentErr,
      // For area chart - show the range
      range: item.predicted !== null ? [Math.min(item.actual, item.predicted), Math.max(item.actual, item.predicted)] : null,
    };
  });

  // Check if we have predictions
  const hasPredictions = chartData.some(d => d.predicted !== null);
  const totalAvailable = data.totalMonthsAvailable || comparison.length;

  // Calculate min/max for better Y-axis scaling
  const allValues = chartData.flatMap(d => [d.actual, d.predicted]).filter(v => v !== null);
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const padding = (maxValue - minValue) * 0.1;

  // Custom tooltip for main chart
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const actual = payload.find(p => p.dataKey === 'actual')?.value;
      const predicted = payload.find(p => p.dataKey === 'predicted')?.value;
      const diff = actual && predicted ? actual - predicted : null;
      const pctErr = actual && predicted && actual !== 0 ? ((actual - predicted) / actual * 100) : null;
      
      return (
        <div style={{
          backgroundColor: '#1e293b',
          border: '1px solid #334155',
          borderRadius: '8px',
          padding: '0.75rem',
          color: '#e2e8f0',
        }}>
          <p style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#f1f5f9' }}>{label}</p>
          <p style={{ color: '#10b981', margin: '0.25rem 0' }}>
            Actual: {formatCurrency(actual)}
          </p>
          {predicted !== null && (
            <>
              <p style={{ color: '#f97316', margin: '0.25rem 0' }}>
                Predicted: {formatCurrency(predicted)}
              </p>
              <hr style={{ border: 'none', borderTop: '1px solid #334155', margin: '0.5rem 0' }} />
              <p style={{ 
                color: diff > 0 ? '#ef4444' : '#10b981', 
                margin: '0.25rem 0',
                fontWeight: '600'
              }}>
                Difference: {diff > 0 ? '+' : ''}{formatCurrency(diff)}
              </p>
              <p style={{ 
                color: pctErr > 0 ? '#ef4444' : '#10b981', 
                margin: '0.25rem 0',
                fontSize: '0.85rem'
              }}>
                Error: {pctErr > 0 ? '+' : ''}{pctErr?.toFixed(2)}%
              </p>
            </>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-card" style={{ gridColumn: 'span 2' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="chart-title">
          Predicted vs Actual Cash Outflow
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          {/* Month Selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Show:</span>
            <select
              value={selectedMonths}
              onChange={(e) => handleMonthsChange(parseInt(e.target.value))}
              style={{
                background: 'rgba(30, 41, 59, 0.8)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '6px',
                padding: '0.4rem 0.75rem',
                color: '#e2e8f0',
                fontSize: '0.8rem',
                cursor: 'pointer',
                outline: 'none'
              }}
            >
              <option value={12}>Last 12 Months</option>
              <option value={24}>Last 24 Months</option>
              <option value={36}>Last 36 Months</option>
              <option value={60}>Last 5 Years</option>
              <option value={120}>Last 10 Years</option>
              <option value={0}>All Data ({totalAvailable} months)</option>
            </select>
          </div>
          {modelName && (
            <span style={{ 
              fontSize: '0.75rem', 
              color: '#60a5fa', 
              background: 'rgba(59, 130, 246, 0.1)',
              padding: '0.25rem 0.75rem',
              borderRadius: '12px'
            }}>
              Model: {modelName}
            </span>
          )}
        </div>
      </div>

      {/* Metrics Cards - Using TEST SET metrics (true predictive accuracy) */}
      {hasPredictions && metrics && Object.keys(metrics).length > 0 && (
        <>
          <div style={{ 
            fontSize: '0.75rem', 
            color: '#64748b', 
            marginBottom: '0.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span style={{ fontStyle: 'italic' }}>Click cards to learn more</span>
            <span>* Metrics from test set (20% unseen data)</span>
          </div>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(4, 1fr)', 
            gap: '1rem', 
            marginBottom: '1.5rem' 
          }}>
            <FlipCard
              icon={FiCheckCircle}
              value={`${metrics.accuracy}%`}
              label="Accuracy*"
              bgColor="rgba(16, 185, 129, 0.1)"
              borderColor="rgba(16, 185, 129, 0.2)"
              iconColor="#10b981"
              valueColor="#10b981"
              explanation="How close predictions are to actual values. Higher is better. 100% means perfect predictions."
              example="98% accuracy = predictions are off by only 2% on average"
            />
            
            <FlipCard
              icon={FiTarget}
              value={metrics.r2}
              label="R² Score*"
              bgColor="rgba(59, 130, 246, 0.1)"
              borderColor="rgba(59, 130, 246, 0.2)"
              iconColor="#3b82f6"
              valueColor="#3b82f6"
              explanation="Measures how well the model explains data patterns. Ranges from 0 to 1. Higher = better fit."
              example="0.96 = model explains 96% of the variation in your data"
            />
            
            <FlipCard
              icon={FiPercent}
              value={`${metrics.mape}%`}
              label="MAPE*"
              bgColor="rgba(249, 115, 22, 0.1)"
              borderColor="rgba(249, 115, 22, 0.2)"
              iconColor="#f97316"
              valueColor="#f97316"
              explanation="Mean Absolute Percentage Error. Average % difference between predicted and actual. Lower is better."
              example="2% MAPE = on average, predictions miss by 2%"
            />
            
            <FlipCard
              icon={FiTrendingUp}
              value={formatCurrency(metrics.mae)}
              label="MAE*"
              bgColor="rgba(168, 85, 247, 0.1)"
              borderColor="rgba(168, 85, 247, 0.2)"
              iconColor="#a855f7"
              valueColor="#a855f7"
              explanation="Mean Absolute Error. Average dollar amount the predictions are off by. Lower is better."
              example="$50K MAE = predictions typically miss by $50K"
            />
          </div>
        </>
      )}

      {/* Main Chart - Time Series with Area Fill */}
      <div style={{ marginBottom: '1rem' }}>
        <h4 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
          Cash Outflow Comparison
        </h4>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 60 }}>
            <defs>
              <linearGradient id="colorDiffArea" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="name"
              stroke="#94a3b8"
              style={{ fontSize: '10px' }}
              angle={-45}
              textAnchor="end"
              height={70}
              interval={selectedMonths > 36 ? Math.floor(chartData.length / 12) : 0}
            />
            <YAxis
              stroke="#94a3b8"
              style={{ fontSize: '11px' }}
              domain={[minValue - padding, maxValue + padding]}
              tickFormatter={(value) => {
                const absValue = Math.abs(value);
                if (absValue >= 1000000) {
                  return `$${(value / 1000000).toFixed(1)}M`;
                }
                return `$${(value / 1000).toFixed(0)}K`;
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {/* Actual values line */}
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#10b981"
              strokeWidth={2}
              dot={selectedMonths <= 24 ? { fill: '#10b981', strokeWidth: 1, r: 3 } : false}
              activeDot={{ r: 6, fill: '#10b981' }}
              name="Actual"
            />
            
            {/* Predicted values line */}
            {hasPredictions && (
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#f97316"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={selectedMonths <= 24 ? { fill: '#f97316', strokeWidth: 1, r: 3 } : false}
                activeDot={{ r: 6, fill: '#f97316' }}
                name="Predicted"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend explanation */}
      <div style={{ 
        marginTop: '1rem', 
        padding: '0.75rem 1rem', 
        backgroundColor: 'rgba(30, 41, 59, 0.5)', 
        borderRadius: '8px',
        fontSize: '0.8rem',
        color: '#94a3b8',
        display: 'flex',
        justifyContent: 'center',
        gap: '2rem',
        flexWrap: 'wrap'
      }}>
        <span>
          <span style={{ 
            display: 'inline-block', 
            width: '20px', 
            height: '3px', 
            background: '#10b981', 
            marginRight: '0.5rem',
            verticalAlign: 'middle'
          }}></span>
          Actual Cash Outflow
        </span>
        {hasPredictions && (
          <span>
            <span style={{ 
              display: 'inline-block', 
              width: '20px', 
              height: '3px', 
              background: '#f97316', 
              marginRight: '0.5rem',
              verticalAlign: 'middle',
              borderTop: '2px dashed #f97316'
            }}></span>
            Model Predictions
          </span>
        )}
      </div>

      {!hasPredictions && (
        <div style={{ 
          marginTop: '1rem', 
          padding: '1rem', 
          background: 'rgba(249, 115, 22, 0.1)',
          border: '1px solid rgba(249, 115, 22, 0.2)',
          borderRadius: '8px',
          textAlign: 'center',
          color: '#f97316',
          fontSize: '0.875rem'
        }}>
          <FiRefreshCw style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} />
          Predictions not available. Please retrain the model using the "Retrain Model" button above to generate the comparison data.
        </div>
      )}
    </div>
  );
};

export default PredictedVsActualChart;
