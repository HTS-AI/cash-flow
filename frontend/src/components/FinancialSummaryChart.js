import React, { useState, useMemo } from 'react';
import { FiTrendingUp, FiTrendingDown, FiCalendar } from 'react-icons/fi';

const FinancialSummaryChart = ({ data }) => {
  const [selectedYear, setSelectedYear] = useState('last10'); // 'last10' or specific year

  const formatCurrency = (amount) => {
    if (amount >= 1000000000) {
      return `$${(amount / 1000000000).toFixed(2)}B`;
    }
    if (amount >= 1000000) {
      return `$${(amount / 1000000).toFixed(1)}M`;
    }
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatFullCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  // Get available years from data
  const availableYears = useMemo(() => {
    if (!data || data.length === 0) return [];
    return [...new Set(data.map(item => item.year))].sort((a, b) => b - a); // Newest first
  }, [data]);

  // Calculate max year for display
  const maxYear = availableYears.length > 0 ? Math.max(...availableYears) : 2025;

  // Filter data based on selection
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    if (selectedYear === 'last10') {
      // Get last 10 years of data
      return data.slice(-10);
    } else {
      // Filter for specific year
      const yearNum = parseInt(selectedYear);
      return data.filter(item => item.year === yearNum);
    }
  }, [data, selectedYear]);

  // Get period label for title
  const getPeriodLabel = () => {
    if (selectedYear === 'last10') {
      if (filteredData.length > 0) {
        const years = filteredData.map(d => d.year);
        const startYear = Math.min(...years);
        const endYear = Math.max(...years);
        return `Last 10 Years (${startYear} - ${endYear})`;
      }
      return 'Last 10 Years';
    } else {
      return `Year ${selectedYear}`;
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">Financial Summary</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No data available
        </div>
      </div>
    );
  }

  // Calculate totals from filtered data
  const totalIncome = filteredData.reduce((sum, item) => sum + item.income, 0);
  const totalExpense = filteredData.reduce((sum, item) => sum + item.expense, 0);
  const totalProfit = totalIncome - totalExpense;
  const profitMargin = totalIncome > 0 ? (totalProfit / totalIncome) * 100 : 0;
  const expenseRatio = totalIncome > 0 ? (totalExpense / totalIncome) * 100 : 0;
  const isProfit = totalProfit >= 0;

  // Calculate bar widths for visualization
  const maxValue = Math.max(totalIncome, totalExpense);
  const incomeWidth = maxValue > 0 ? (totalIncome / maxValue) * 100 : 0;
  const expenseWidth = maxValue > 0 ? (totalExpense / maxValue) * 100 : 0;

  return (
    <div className="chart-card">
      {/* Header with Title and Filter */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="chart-title" style={{ margin: 0 }}>
          Financial Summary - {getPeriodLabel()}
        </div>
        
        {/* Year Filter */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FiCalendar size={16} style={{ color: '#94a3b8' }} />
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(e.target.value)}
            style={{
              padding: '0.4rem 0.75rem',
              backgroundColor: '#1e293b',
              color: '#e2e8f0',
              border: '1px solid #334155',
              borderRadius: '6px',
              fontSize: '0.85rem',
              cursor: 'pointer',
              minWidth: '160px',
            }}
          >
            <option value="last10">Last 10 Years ({maxYear - 9} - {maxYear})</option>
            <optgroup label="Select Specific Year">
              {availableYears.map(year => (
                <option key={year} value={year}>{year}</option>
              ))}
            </optgroup>
          </select>
        </div>
      </div>

      {/* Period Info Badge */}
      {selectedYear !== 'last10' && (
        <div style={{ 
          marginTop: '0.75rem', 
          display: 'flex', 
          alignItems: 'center', 
          gap: '0.5rem' 
        }}>
          <span style={{
            padding: '0.25rem 0.75rem',
            backgroundColor: '#3b82f6',
            color: 'white',
            borderRadius: '20px',
            fontSize: '0.75rem',
            fontWeight: 'bold',
          }}>
            Viewing: {selectedYear}
          </span>
          <button
            onClick={() => setSelectedYear('last10')}
            style={{
              padding: '0.25rem 0.5rem',
              backgroundColor: 'transparent',
              color: '#94a3b8',
              border: '1px solid #334155',
              borderRadius: '4px',
              fontSize: '0.7rem',
              cursor: 'pointer',
            }}
          >
            Reset to Last 10 Years
          </button>
        </div>
      )}
      
      {/* Main Stats Cards */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '1rem',
        marginTop: '1rem'
      }}>
        {/* Income Card */}
        <div style={{
          backgroundColor: '#064e3b',
          borderRadius: '12px',
          padding: '1.25rem',
          textAlign: 'center',
        }}>
          <div style={{ color: '#6ee7b7', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
            Total Income
          </div>
          <div style={{ color: '#10b981', fontSize: '1.75rem', fontWeight: 'bold' }}>
            {formatCurrency(totalIncome)}
          </div>
          <div style={{ color: '#6ee7b7', fontSize: '0.75rem', marginTop: '0.25rem' }}>
            {formatFullCurrency(totalIncome)}
          </div>
        </div>

        {/* Expense Card */}
        <div style={{
          backgroundColor: '#7f1d1d',
          borderRadius: '12px',
          padding: '1.25rem',
          textAlign: 'center',
        }}>
          <div style={{ color: '#fca5a5', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
            Total Expense
          </div>
          <div style={{ color: '#ef4444', fontSize: '1.75rem', fontWeight: 'bold' }}>
            {formatCurrency(totalExpense)}
          </div>
          <div style={{ color: '#fca5a5', fontSize: '0.75rem', marginTop: '0.25rem' }}>
            {formatFullCurrency(totalExpense)}
          </div>
        </div>

        {/* Profit Card */}
        <div style={{
          backgroundColor: isProfit ? '#1e3a5f' : '#78350f',
          borderRadius: '12px',
          padding: '1.25rem',
          textAlign: 'center',
        }}>
          <div style={{ color: isProfit ? '#93c5fd' : '#fcd34d', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
            Net {isProfit ? 'Profit' : 'Loss'}
          </div>
          <div style={{ 
            color: isProfit ? '#3b82f6' : '#f59e0b', 
            fontSize: '1.75rem', 
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.5rem'
          }}>
            {isProfit ? <FiTrendingUp /> : <FiTrendingDown />}
            {formatCurrency(Math.abs(totalProfit))}
          </div>
          <div style={{ color: isProfit ? '#93c5fd' : '#fcd34d', fontSize: '0.75rem', marginTop: '0.25rem' }}>
            {formatFullCurrency(Math.abs(totalProfit))}
          </div>
        </div>
      </div>

      {/* Visual Comparison Bars */}
      <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#1e293b', borderRadius: '12px' }}>
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ color: '#10b981', fontWeight: 'bold' }}>Income</span>
            <span style={{ color: '#94a3b8' }}>{formatCurrency(totalIncome)}</span>
          </div>
          <div style={{ 
            height: '24px', 
            backgroundColor: '#0f172a', 
            borderRadius: '12px', 
            overflow: 'hidden' 
          }}>
            <div style={{
              width: `${incomeWidth}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #059669, #10b981)',
              borderRadius: '12px',
              transition: 'width 0.5s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              paddingRight: '0.5rem',
            }}>
              <span style={{ color: 'white', fontSize: '0.75rem', fontWeight: 'bold' }}>100%</span>
            </div>
          </div>
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ color: '#ef4444', fontWeight: 'bold' }}>Expense</span>
            <span style={{ color: '#94a3b8' }}>{formatCurrency(totalExpense)} ({expenseRatio.toFixed(1)}% of income)</span>
          </div>
          <div style={{ 
            height: '24px', 
            backgroundColor: '#0f172a', 
            borderRadius: '12px', 
            overflow: 'hidden' 
          }}>
            <div style={{
              width: `${expenseWidth}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #dc2626, #ef4444)',
              borderRadius: '12px',
              transition: 'width 0.5s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
              paddingRight: '0.5rem',
            }}>
              <span style={{ color: 'white', fontSize: '0.75rem', fontWeight: 'bold' }}>{expenseRatio.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Profit Margin Gauge */}
      <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#1e293b', borderRadius: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <span style={{ color: '#94a3b8', fontWeight: 'bold' }}>Profit Margin</span>
          <span style={{ 
            color: profitMargin >= 5 ? '#10b981' : profitMargin >= 0 ? '#f59e0b' : '#ef4444',
            fontSize: '1.5rem',
            fontWeight: 'bold'
          }}>
            {profitMargin.toFixed(2)}%
          </span>
        </div>
        <div style={{ 
          height: '12px', 
          backgroundColor: '#0f172a', 
          borderRadius: '6px', 
          overflow: 'hidden',
          position: 'relative'
        }}>
          {/* Background gradient showing scale */}
          <div style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, #ef4444 0%, #f59e0b 25%, #eab308 50%, #84cc16 75%, #10b981 100%)',
            opacity: 0.3,
          }} />
          {/* Marker */}
          <div style={{
            position: 'absolute',
            left: `${Math.min(Math.max(profitMargin, 0), 20) * 5}%`,
            top: '-4px',
            width: '4px',
            height: '20px',
            backgroundColor: '#fff',
            borderRadius: '2px',
            boxShadow: '0 0 10px rgba(255,255,255,0.5)',
            transition: 'left 0.5s ease',
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.25rem' }}>
          <span style={{ color: '#ef4444', fontSize: '0.7rem' }}>0%</span>
          <span style={{ color: '#f59e0b', fontSize: '0.7rem' }}>5%</span>
          <span style={{ color: '#84cc16', fontSize: '0.7rem' }}>10%</span>
          <span style={{ color: '#10b981', fontSize: '0.7rem' }}>15%</span>
          <span style={{ color: '#10b981', fontSize: '0.7rem' }}>20%+</span>
        </div>
      </div>

      {/* Formula Explanation */}
      <div style={{ 
        marginTop: '1rem', 
        padding: '0.75rem', 
        backgroundColor: '#0f172a', 
        borderRadius: '8px',
        textAlign: 'center',
        color: '#64748b',
        fontSize: '0.875rem'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <span style={{ color: '#10b981' }}>{formatCurrency(totalIncome)}</span>
          <span>−</span>
          <span style={{ color: '#ef4444' }}>{formatCurrency(totalExpense)}</span>
          <span>=</span>
          <span style={{ color: isProfit ? '#3b82f6' : '#f59e0b', fontWeight: 'bold' }}>
            {isProfit ? '' : '-'}{formatCurrency(Math.abs(totalProfit))}
          </span>
          <span style={{ color: '#94a3b8' }}>({isProfit ? 'Profit' : 'Loss'})</span>
        </div>
        {selectedYear !== 'last10' && (
          <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#64748b' }}>
            Data for year {selectedYear}
          </div>
        )}
      </div>
    </div>
  );
};

export default FinancialSummaryChart;
