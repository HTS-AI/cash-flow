import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const IncomeExpenseChart = ({ data }) => {
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
        <div className="chart-title">Income vs Expense - Last 10 Years</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No data available
        </div>
      </div>
    );
  }

  // Prepare chart data
  const chartData = data.map(item => ({
    year: item.year.toString(),
    Income: item.income,
    Expense: item.expense,
  }));

  return (
    <div className="chart-card">
      <div className="chart-title">Income vs Expense - Last 10 Years</div>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
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
            tickFormatter={(value) => `$${(value / 1000000).toFixed(0)}M`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
              color: '#e2e8f0',
            }}
            formatter={(value, name) => [formatCurrency(value), name]}
            labelFormatter={(label) => `Year: ${label}`}
          />
          <Legend />
          <Bar
            dataKey="Income"
            fill="#10b981"
            name="Income"
            radius={[4, 4, 0, 0]}
          />
          <Bar
            dataKey="Expense"
            fill="#ef4444"
            name="Expense"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
      <div style={{ 
        marginTop: '1rem', 
        padding: '0.75rem', 
        backgroundColor: '#1e293b', 
        borderRadius: '8px',
        fontSize: '0.85rem',
        color: '#94a3b8',
        display: 'flex',
        justifyContent: 'center',
        gap: '2rem'
      }}>
        <span><strong style={{ color: '#10b981' }}>Green</strong> = Income (Cash Inflow)</span>
        <span><strong style={{ color: '#ef4444' }}>Red</strong> = Expense (Cash Outflow)</span>
      </div>
    </div>
  );
};

export default IncomeExpenseChart;
