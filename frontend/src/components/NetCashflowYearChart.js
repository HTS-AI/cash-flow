import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';

const NetCashflowYearChart = ({ data }) => {
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  if (!data || data.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">Net Cashflow - Last 10 Years</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No data available
        </div>
      </div>
    );
  }

  // Prepare chart data
  const chartData = data.map(item => ({
    year: item.year.toString(),
    'Net Cashflow': item.netCashflow,
  }));

  return (
    <div className="chart-card">
      <div className="chart-title">Net Cashflow - Last 10 Years</div>
      <ResponsiveContainer width="100%" height={350}>
        <AreaChart
          data={chartData}
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorNetCashflow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="year"
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
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
            formatter={(value) => formatCurrency(value)}
            labelFormatter={(label) => `Year: ${label}`}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="Net Cashflow"
            stroke="#3b82f6"
            fillOpacity={1}
            fill="url(#colorNetCashflow)"
            name="Net Cashflow"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default NetCashflowYearChart;
