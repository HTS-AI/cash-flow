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
} from 'recharts';
import { cashflowAPI } from '../services/api';

const MONTHS = [
  { value: 0, label: 'All Months (Full Year)' },
  { value: 1, label: 'January' },
  { value: 2, label: 'February' },
  { value: 3, label: 'March' },
  { value: 4, label: 'April' },
  { value: 5, label: 'May' },
  { value: 6, label: 'June' },
  { value: 7, label: 'July' },
  { value: 8, label: 'August' },
  { value: 9, label: 'September' },
  { value: 10, label: 'October' },
  { value: 11, label: 'November' },
  { value: 12, label: 'December' },
];

const IncomeExpenseChart = ({ data: initialData }) => {
  const [selectedMonth, setSelectedMonth] = useState(0);
  const [chartData, setChartData] = useState(initialData || []);
  const [loading, setLoading] = useState(false);
  const [monthName, setMonthName] = useState(null);

  // Load data when month changes
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const response = await cashflowAPI.getYearOverYear(selectedMonth);
        if (response.data.success) {
          setChartData(response.data.data);
          setMonthName(response.data.monthName);
        }
      } catch (err) {
        console.error('Error loading year-over-year data:', err);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [selectedMonth]);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const handleMonthChange = (e) => {
    setSelectedMonth(parseInt(e.target.value));
  };

  // Get title based on selection
  const getTitle = () => {
    if (selectedMonth === 0) {
      return 'Income vs Expense - Last 10 Years';
    }
    return `Income vs Expense - ${monthName || MONTHS[selectedMonth].label} (Last 10 Years)`;
  };

  if (!chartData || chartData.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">{getTitle()}</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          {loading ? 'Loading...' : 'No data available'}
        </div>
      </div>
    );
  }

  // Prepare chart data
  const formattedData = chartData.map(item => ({
    year: item.year.toString(),
    Income: item.income,
    Expense: item.expense,
  }));

  return (
    <div className="chart-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="chart-title">{getTitle()}</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Filter by Month:</span>
          <select
            value={selectedMonth}
            onChange={handleMonthChange}
            style={{
              background: 'rgba(30, 41, 59, 0.8)',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              borderRadius: '6px',
              padding: '0.4rem 0.75rem',
              color: '#e2e8f0',
              fontSize: '0.8rem',
              cursor: 'pointer',
              outline: 'none',
              minWidth: '150px'
            }}
          >
            {MONTHS.map((month) => (
              <option key={month.value} value={month.value}>
                {month.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '400px' }}>
          <div className="spinner"></div>
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={formattedData}
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
                tickFormatter={(value) => {
                  if (value >= 1000000) {
                    return `$${(value / 1000000).toFixed(0)}M`;
                  } else if (value >= 1000) {
                    return `$${(value / 1000).toFixed(0)}K`;
                  }
                  return `$${value}`;
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
                labelFormatter={(label) => selectedMonth === 0 ? `Year: ${label}` : `${monthName} ${label}`}
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
            gap: '2rem',
            flexWrap: 'wrap'
          }}>
            <span><strong style={{ color: '#10b981' }}>Green</strong> = Income (Cash Inflow)</span>
            <span><strong style={{ color: '#ef4444' }}>Red</strong> = Expense (Cash Outflow)</span>
            {selectedMonth > 0 && (
              <span style={{ color: '#60a5fa' }}>
                Showing {monthName} data across all years
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default IncomeExpenseChart;
