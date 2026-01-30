import React, { useState } from 'react';
import { cashflowAPI } from '../services/api';
import { FiTrendingUp, FiRefreshCw } from 'react-icons/fi';

const PredictionsPanel = ({ predictions, modelInfo, onGeneratePredictions, onModelRetrained }) => {
  const [forecastMonths, setForecastMonths] = useState(1);
  const [generating, setGenerating] = useState(false);
  const [training, setTraining] = useState(false);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(amount);
  };

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      await onGeneratePredictions(forecastMonths);
    } finally {
      setGenerating(false);
    }
  };

  const handleRetrainModel = async () => {
    if (!window.confirm('Are you sure you want to retrain the model? This may take several minutes.')) {
      return;
    }
    
    setTraining(true);
    try {
      const response = await cashflowAPI.trainModel(forecastMonths);
      if (response.data.success) {
        alert('Model retrained successfully!');
        if (onModelRetrained) {
          onModelRetrained();
        }
      } else {
        alert('Model training failed: ' + (response.data.error || response.data.message));
      }
    } catch (err) {
      console.error('Error training model:', err);
      alert('Error training model: ' + (err.response?.data?.detail || err.message || 'Unknown error'));
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="chart-card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div className="chart-title">Future Predictions</div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {modelInfo && (
            <div style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
              Model: {modelInfo.best_model} (R²: {modelInfo.test_r2?.toFixed(4)})
            </div>
          )}
          <button
            className="btn btn-secondary"
            onClick={handleRetrainModel}
            disabled={training}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem',
              fontSize: '0.875rem',
              padding: '0.5rem 1rem'
            }}
          >
            <FiRefreshCw style={{ animation: training ? 'spin 1s linear infinite' : 'none' }} />
            {training ? 'Training...' : 'Retrain Model'}
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', alignItems: 'center' }}>
        <label style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
          Forecast Months:
          <select
            value={forecastMonths}
            onChange={(e) => setForecastMonths(parseInt(e.target.value))}
            style={{
              marginLeft: '0.5rem',
              padding: '0.5rem',
              borderRadius: '6px',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              background: 'rgba(15, 23, 42, 0.5)',
              color: '#e2e8f0',
              width: '100px',
              cursor: 'pointer',
            }}
          >
            {Array.from({ length: 12 }, (_, i) => i + 1).map((month) => (
              <option key={month} value={month} style={{ background: '#1e293b', color: '#e2e8f0' }}>
                {month} {month === 1 ? 'Month' : 'Months'}
              </option>
            ))}
          </select>
        </label>
        <button
          className="btn btn-primary"
          onClick={handleGenerate}
          disabled={generating}
        >
          {generating ? 'Generating...' : 'Generate Predictions'}
        </button>
      </div>

      {predictions && predictions.length > 0 ? (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.1)' }}>
                <th style={{ padding: '0.75rem', textAlign: 'left', color: '#94a3b8', fontSize: '0.875rem' }}>Month</th>
                <th style={{ padding: '0.75rem', textAlign: 'right', color: '#94a3b8', fontSize: '0.875rem' }}>Predicted</th>
                <th style={{ padding: '0.75rem', textAlign: 'right', color: '#94a3b8', fontSize: '0.875rem' }}>Lower 95% CI</th>
                <th style={{ padding: '0.75rem', textAlign: 'right', color: '#94a3b8', fontSize: '0.875rem' }}>Upper 95% CI</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((pred, index) => (
                <tr key={index} style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.05)' }}>
                  <td style={{ padding: '0.75rem', color: '#e2e8f0' }}>{pred.month}</td>
                  <td style={{ padding: '0.75rem', textAlign: 'right', color: '#3b82f6', fontWeight: '600' }}>
                    {formatCurrency(parseFloat(pred.predicted_cash_outflow || 0))}
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'right', color: '#10b981' }}>
                    {formatCurrency(parseFloat(pred.lower_95 || 0))}
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'right', color: '#ef4444' }}>
                    {formatCurrency(parseFloat(pred.upper_95 || 0))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <FiTrendingUp style={{ fontSize: '2rem', marginBottom: '0.5rem', opacity: 0.5 }} />
          <div>No predictions available. Click "Generate Predictions" to create forecasts.</div>
        </div>
      )}
    </div>
  );
};

export default PredictionsPanel;
