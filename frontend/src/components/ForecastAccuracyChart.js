import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { cashflowAPI } from '../services/api';
import { FiCheckCircle, FiClock } from 'react-icons/fi';
import FlipCard from './FlipCard';

const ForecastAccuracyChart = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await cashflowAPI.getForecastAccuracy();
      if (response.data.success) {
        setData(response.data.data);
      } else {
        setError(response.data.message || 'Failed to load data');
      }
    } catch (err) {
      setError(err.message || 'Failed to load forecast accuracy data');
    } finally {
      setLoading(false);
    }
  };

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

  const formatMonth = (month) => {
    if (!month) return '';
    const parts = month.split('-');
    if (parts.length < 2) return month;
    const [year, monthNum] = parts;
    const date = new Date(parseInt(year), parseInt(monthNum) - 1);
    return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
  };

  if (loading) {
    return (
      <div className="chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="chart-title">Forecast Accuracy Tracking</div>
        <div className="loading" style={{ minHeight: '300px' }}>
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="chart-title">Forecast Accuracy Tracking</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!data || !data.comparisons || data.comparisons.length === 0) {
    return (
      <div className="chart-card" style={{ gridColumn: 'span 2' }}>
        <div className="chart-title">Forecast Accuracy Tracking</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <FiClock style={{ fontSize: '2rem', marginBottom: '1rem', opacity: 0.5 }} />
          <div>No predictions to track yet.</div>
          <div style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
            Generate predictions using "Future Predictions" above, then check back when actual data becomes available.
          </div>
        </div>
      </div>
    );
  }

  const { comparisons, verifiedCount, pendingCount } = data;

  // Prepare chart data - only show verified predictions
  const verifiedData = comparisons.filter(c => c.status === 'verified');
  const pendingData = comparisons.filter(c => c.status === 'pending');

  const chartData = verifiedData.map((item) => ({
    name: formatMonth(item.month),
    month: item.month,
    predicted: item.predicted,
    actual: item.actual,
    error: item.error,
    percentError: item.percentError,
  }));

  return (
    <div className="chart-card" style={{ gridColumn: 'span 2' }}>
      <div className="chart-title">Forecast Accuracy Tracking</div>
      <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '1rem' }}>
        Compares predictions made for future months with actual data when it becomes available
      </p>

      {/* Summary Cards */}
      <div style={{ 
        fontSize: '0.75rem', 
        color: '#64748b', 
        marginBottom: '0.5rem',
        fontStyle: 'italic'
      }}>
        Click cards to learn more
      </div>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(2, 1fr)', 
        gap: '1rem', 
        marginBottom: '1.5rem',
        maxWidth: '400px'
      }}>
        <FlipCard
          icon={FiCheckCircle}
          value={verifiedCount}
          label="Verified"
          bgColor="rgba(16, 185, 129, 0.1)"
          borderColor="rgba(16, 185, 129, 0.2)"
          iconColor="#10b981"
          valueColor="#10b981"
          explanation="Number of predictions where actual data is now available for comparison."
          example="3 verified = we can now check 3 past predictions against reality"
        />
        
        <FlipCard
          icon={FiClock}
          value={pendingCount}
          label="Pending"
          bgColor="rgba(249, 115, 22, 0.1)"
          borderColor="rgba(249, 115, 22, 0.2)"
          iconColor="#f97316"
          valueColor="#f97316"
          explanation="Predictions waiting for actual data. These are for future months that haven't happened yet."
          example="2 pending = waiting for 2 months to pass to verify those predictions"
        />
      </div>

      {/* Verified Predictions Chart */}
      {chartData.length > 0 ? (
        <>
          <h4 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
            Verified Predictions vs Actual
          </h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="name"
                stroke="#94a3b8"
                style={{ fontSize: '11px' }}
                angle={-45}
                textAnchor="end"
                height={70}
              />
              <YAxis
                stroke="#94a3b8"
                style={{ fontSize: '11px' }}
                tickFormatter={(value) => {
                  if (value >= 1000000) {
                    return `$${(value / 1000000).toFixed(1)}M`;
                  }
                  return `$${(value / 1000).toFixed(0)}K`;
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0',
                }}
                formatter={(value, name) => [formatCurrency(value), name]}
                labelFormatter={(label) => `Month: ${label}`}
              />
              <Legend />
              <Bar dataKey="predicted" name="Predicted" fill="#f97316" radius={[4, 4, 0, 0]} />
              <Bar dataKey="actual" name="Actual" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </>
      ) : (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8', background: 'rgba(30, 41, 59, 0.5)', borderRadius: '8px' }}>
          <FiClock style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }} />
          <div>No verified predictions yet.</div>
          <div style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
            Predictions will be verified once actual data for those months becomes available.
          </div>
        </div>
      )}

      {/* Pending Predictions List */}
      {pendingData.length > 0 && (
        <div style={{ marginTop: '1.5rem' }}>
          <h4 style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '0.75rem' }}>
            Pending Predictions (Awaiting Actual Data)
          </h4>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', 
            gap: '0.75rem' 
          }}>
            {pendingData.map((item, index) => (
              <div 
                key={index}
                style={{
                  background: 'rgba(249, 115, 22, 0.1)',
                  border: '1px solid rgba(249, 115, 22, 0.2)',
                  borderRadius: '8px',
                  padding: '0.75rem',
                }}
              >
                <div style={{ fontSize: '0.9rem', fontWeight: '600', color: '#f97316' }}>
                  {formatMonth(item.month)}
                </div>
                <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#e2e8f0', marginTop: '0.25rem' }}>
                  {formatCurrency(item.predicted)}
                </div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.25rem' }}>
                  Predicted on: {item.predictedOn}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{ 
        marginTop: '1rem', 
        padding: '0.75rem', 
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
          <FiCheckCircle style={{ color: '#10b981', marginRight: '0.25rem', verticalAlign: 'middle' }} />
          Verified = Actual data available
        </span>
        <span>
          <FiClock style={{ color: '#f97316', marginRight: '0.25rem', verticalAlign: 'middle' }} />
          Pending = Awaiting actual data
        </span>
      </div>
    </div>
  );
};

export default ForecastAccuracyChart;
