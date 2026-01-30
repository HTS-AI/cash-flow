import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const MonthlySpendingChart = ({ data, predictions }) => {
  if (!data || data.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">Monthly Spending & Predictions</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No historical data available
        </div>
      </div>
    );
  }

  // Format month labels with year
  const formatMonth = (month) => {
    if (!month) return '';
    const parts = month.split('-');
    if (parts.length < 2) return month;
    const [year, monthNum] = parts;
    const date = new Date(parseInt(year), parseInt(monthNum) - 1);
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  // Get only the last 12 months for display
  const recentData = data.slice(-12);

  // Build historical chart data
  const historicalChartData = recentData.map((item) => {
    const outflow = parseFloat(item.cash_outflow) || 0;
    const label = formatMonth(item.month);
    return {
      name: label,
      month: item.month,
      historical: outflow,
      predicted: null,
    };
  });

  // Build prediction chart data
  const predictionChartData = predictions && predictions.length > 0
    ? predictions.map((pred) => {
        const outflow = parseFloat(pred.predicted_cash_outflow) || 0;
        const label = formatMonth(pred.month);
        return {
          name: label,
          month: pred.month,
          historical: null,
          predicted: outflow,
        };
      })
    : [];

  // Combine historical and prediction data
  const chartData = [...historicalChartData, ...predictionChartData];

  // Sort by month
  chartData.sort((a, b) => {
    if (a.month && b.month) {
      return a.month.localeCompare(b.month);
    }
    return 0;
  });

  // Check if we have any valid data
  const hasValidData = chartData.some(d => (d.historical && d.historical > 0) || (d.predicted && d.predicted > 0));
  if (!hasValidData) {
    return (
      <div className="chart-card">
        <div className="chart-title">Monthly Spending & Predictions</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No valid spending data to display
        </div>
      </div>
    );
  }

  const hasPredictions = predictionChartData.length > 0;

  return (
    <div className="chart-card">
      <div className="chart-title">
        Monthly Spending {hasPredictions ? '& Predictions' : '(Last 12 Months)'}
      </div>
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 60 }}>
          <defs>
            <linearGradient id="colorHistorical" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
            </linearGradient>
            <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f97316" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#f97316" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="name"
            stroke="#94a3b8"
            style={{ fontSize: '11px' }}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
            tickFormatter={(value) => `$${(value / 1000000).toFixed(1)}M`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
              color: '#e2e8f0',
            }}
            formatter={(value, name) => {
              if (value === null || value === undefined) return ['-', name];
              return [`$${value.toLocaleString()}`, name];
            }}
          />
          <Legend />
          {/* Historical data - Blue */}
          <Area
            type="monotone"
            dataKey="historical"
            stroke="#3b82f6"
            fillOpacity={1}
            fill="url(#colorHistorical)"
            name="Historical Spending"
            connectNulls={false}
          />
          {/* Predictions - Orange */}
          <Area
            type="monotone"
            dataKey="predicted"
            stroke="#f97316"
            fillOpacity={1}
            fill="url(#colorPredicted)"
            name="Predicted Spending"
            connectNulls={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      {hasPredictions && (
        <div style={{ 
          marginTop: '0.5rem', 
          padding: '0.5rem 1rem', 
          backgroundColor: '#1e293b', 
          borderRadius: '8px',
          fontSize: '0.75rem',
          color: '#94a3b8',
          display: 'flex',
          justifyContent: 'center',
          gap: '2rem'
        }}>
          <span><strong style={{ color: '#3b82f6' }}>Blue</strong> = Historical data</span>
          <span><strong style={{ color: '#f97316' }}>Orange</strong> = Predictions</span>
        </div>
      )}
    </div>
  );
};

export default MonthlySpendingChart;
