import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const ExpenseCategoriesChart = ({ expenseBreakdown }) => {
  const colors = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444'];

  const data = [
    { name: 'Vendor Payment', value: expenseBreakdown.vendor },
    { name: 'Salary Payment', value: expenseBreakdown.salary },
    { name: 'Rent', value: expenseBreakdown.rent },
    { name: 'Operational Expense', value: expenseBreakdown.operational },
  ].filter(item => item.value > 0);

  // Calculate total for percentages
  const total = data.reduce((sum, item) => sum + item.value, 0);

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  // Custom tooltip component
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      const percentage = ((data.value / total) * 100).toFixed(1);
      const colorIndex = ['Vendor Payment', 'Salary Payment', 'Rent', 'Operational Expense'].indexOf(data.name);
      
      return (
        <div style={{
          backgroundColor: '#0f172a',
          border: '2px solid #334155',
          borderRadius: '12px',
          padding: '16px',
          boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
          minWidth: '200px',
        }}>
          <div style={{ 
            color: colors[colorIndex] || '#e2e8f0', 
            fontSize: '1rem', 
            fontWeight: 'bold',
            marginBottom: '12px',
            borderBottom: `2px solid ${colors[colorIndex] || '#334155'}`,
            paddingBottom: '8px'
          }}>
            {data.name}
          </div>
          <div style={{ 
            color: '#e2e8f0', 
            fontSize: '1.25rem', 
            fontWeight: 'bold',
            marginBottom: '8px'
          }}>
            {formatCurrency(data.value)}
          </div>
          <div style={{ 
            color: '#94a3b8', 
            fontSize: '0.875rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: colors[colorIndex] || '#94a3b8',
            }} />
            {percentage}% of total expenses
          </div>
        </div>
      );
    }
    return null;
  };

  // Custom label that only shows percentage
  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
    const RADIAN = Math.PI / 180;
    const radius = outerRadius + 25;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    if (percent < 0.05) return null; // Don't show label for small slices

    return (
      <text
        x={x}
        y={y}
        fill="#e2e8f0"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
        fontSize="12"
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="chart-card">
      <div className="chart-title">Expense Categories</div>
      <ResponsiveContainer width="100%" height={350}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="45%"
            labelLine={true}
            label={renderCustomLabel}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
            stroke="#0f172a"
            strokeWidth={2}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
            formatter={(value, entry) => (
              <span style={{ color: '#e2e8f0', fontSize: '0.875rem' }}>{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ExpenseCategoriesChart;
