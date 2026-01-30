import React, { useState, useEffect, useMemo } from 'react';
import { cashflowAPI } from '../services/api';
import { FiRefreshCw, FiChevronLeft, FiChevronRight, FiCalendar } from 'react-icons/fi';

const SampleDataDisplay = () => {
  const [loading, setLoading] = useState(true);
  const [sampleData, setSampleData] = useState(null);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [aggregation, setAggregation] = useState('day'); // 'day', 'month', 'year'
  const [selectedYear, setSelectedYear] = useState('all');
  const [selectedMonth, setSelectedMonth] = useState('all');
  const [selectedDay, setSelectedDay] = useState('all');
  const rowsPerPage = 10;

  useEffect(() => {
    loadSampleData();
  }, []);

  const loadSampleData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await cashflowAPI.getSampleData();
      if (response.data.success) {
        setSampleData(response.data.data);
      } else {
        setError(response.data.message || 'Failed to load sample data');
      }
    } catch (err) {
      setError(err.message || 'Failed to load sample data');
    } finally {
      setLoading(false);
    }
  };

  const formatColumnName = (col) => {
    if (col === 'period') {
      if (aggregation === 'year') return 'Year';
      if (aggregation === 'month') return 'Month';
      return 'Date';
    }
    return col
      .replace(/_/g, ' ')
      .replace(/usd/gi, '(USD)')
      .replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const formatValue = (value, column) => {
    if (value === null || value === undefined) return '-';
    if (column === 'date' || column === 'period') return value;
    if (typeof value === 'number') {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      }).format(value);
    }
    return value;
  };

  // Generate years from 1998 to current year (2026)
  const availableYears = useMemo(() => {
    const currentYear = new Date().getFullYear();
    const years = [];
    for (let year = currentYear; year >= 1998; year--) {
      years.push(year.toString());
    }
    return years;
  }, []);

  // Get available months for the selected year from actual data
  const availableMonthsForYear = useMemo(() => {
    if (!sampleData || !sampleData.rows || selectedYear === 'all') {
      return [];
    }
    const months = new Set();
    sampleData.rows.forEach(row => {
      if (row.date && row.date.startsWith(selectedYear)) {
        const month = row.date.substring(5, 7);
        months.add(month);
      }
    });
    return Array.from(months).sort();
  }, [sampleData, selectedYear]);

  // Get available days for the selected year and month from actual data
  const availableDaysForMonth = useMemo(() => {
    if (!sampleData || !sampleData.rows || selectedYear === 'all' || selectedMonth === 'all') {
      return [];
    }
    const days = new Set();
    const prefix = `${selectedYear}-${selectedMonth}`;
    sampleData.rows.forEach(row => {
      if (row.date && row.date.startsWith(prefix)) {
        const day = row.date.substring(8, 10);
        days.add(day);
      }
    });
    return Array.from(days).sort();
  }, [sampleData, selectedYear, selectedMonth]);

  // Month names
  const monthNames = {
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December'
  };

  // Filter and aggregate data
  const aggregatedData = useMemo(() => {
    if (!sampleData || !sampleData.rows) return { columns: [], rows: [] };

    // First, filter by year, month, and day
    let filteredRows = sampleData.rows;
    
    if (selectedYear !== 'all') {
      filteredRows = filteredRows.filter(row => 
        row.date && row.date.startsWith(selectedYear)
      );
      
      if (selectedMonth !== 'all') {
        const monthPrefix = `${selectedYear}-${selectedMonth}`;
        filteredRows = filteredRows.filter(row => 
          row.date && row.date.startsWith(monthPrefix)
        );
        
        if (selectedDay !== 'all' && aggregation === 'day') {
          const dayPrefix = `${selectedYear}-${selectedMonth}-${selectedDay}`;
          filteredRows = filteredRows.filter(row => 
            row.date && row.date.startsWith(dayPrefix)
          );
        }
      }
    }

    const numericColumns = sampleData.columns.filter(col => col !== 'date');

    if (aggregation === 'day') {
      // No aggregation, show filtered raw data
      return {
        columns: sampleData.columns,
        rows: filteredRows
      };
    }

    // Group data by period
    const groups = {};
    
    filteredRows.forEach(row => {
      let period;
      const date = row.date;
      
      if (aggregation === 'month') {
        period = date ? date.substring(0, 7) : 'Unknown';
      } else if (aggregation === 'year') {
        period = date ? date.substring(0, 4) : 'Unknown';
      }

      if (!groups[period]) {
        groups[period] = {
          period: period,
          count: 0
        };
        numericColumns.forEach(col => {
          groups[period][col] = 0;
        });
      }

      groups[period].count++;
      numericColumns.forEach(col => {
        const val = parseFloat(row[col]) || 0;
        groups[period][col] += val;
      });
    });

    const aggregatedRows = Object.values(groups).sort((a, b) => {
      return b.period.localeCompare(a.period);
    });

    return {
      columns: ['period', ...numericColumns],
      rows: aggregatedRows
    };
  }, [sampleData, aggregation, selectedYear, selectedMonth, selectedDay]);

  // Handle aggregation change
  const handleAggregationChange = (newAggregation) => {
    setAggregation(newAggregation);
    setCurrentPage(0);
    // Reset lower-level selections based on aggregation
    if (newAggregation === 'year') {
      setSelectedMonth('all');
      setSelectedDay('all');
    } else if (newAggregation === 'month') {
      setSelectedDay('all');
    }
  };

  // Handle year change
  const handleYearChange = (year) => {
    setSelectedYear(year);
    setSelectedMonth('all');
    setSelectedDay('all');
    setCurrentPage(0);
  };

  // Handle month change
  const handleMonthChange = (month) => {
    setSelectedMonth(month);
    setSelectedDay('all');
    setCurrentPage(0);
  };

  // Handle day change
  const handleDayChange = (day) => {
    setSelectedDay(day);
    setCurrentPage(0);
  };

  // Check if any filters are active
  const hasActiveFilters = selectedYear !== 'all' || selectedMonth !== 'all' || selectedDay !== 'all';

  // Get filter description
  const getFilterDescription = () => {
    if (selectedYear === 'all') return 'All data (1998 - 2026)';
    let desc = selectedYear;
    if (selectedMonth !== 'all') {
      desc += ` - ${monthNames[selectedMonth]}`;
      if (selectedDay !== 'all' && aggregation === 'day') {
        desc += ` ${selectedDay}`;
      }
    }
    return desc;
  };

  if (loading) {
    return (
      <div className="chart-card">
        <div className="chart-title">Sample Data</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          <div className="spinner" style={{ margin: '0 auto' }}></div>
          Loading sample data...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="chart-card">
        <div className="chart-title">Sample Data</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#ef4444' }}>
          <p>{error}</p>
          <button
            onClick={loadSampleData}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            <FiRefreshCw /> Retry
          </button>
        </div>
      </div>
    );
  }

  if (!sampleData || !sampleData.rows || sampleData.rows.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-title">Sample Data</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No sample data available.
        </div>
      </div>
    );
  }

  const displayRows = aggregatedData.rows;
  const displayColumns = aggregatedData.columns;
  const totalPages = Math.ceil(displayRows.length / rowsPerPage);
  const startIdx = currentPage * rowsPerPage;
  const endIdx = Math.min(startIdx + rowsPerPage, displayRows.length);
  const currentRows = displayRows.slice(startIdx, endIdx);

  const selectStyle = {
    padding: '0.5rem 0.75rem',
    backgroundColor: '#1e293b',
    color: '#e2e8f0',
    border: '1px solid #334155',
    borderRadius: '6px',
    fontSize: '0.875rem',
    cursor: 'pointer',
    outline: 'none',
    minWidth: '120px',
  };

  return (
    <div className="chart-card">
      <div className="chart-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <span>Sample Data (Transactions)</span>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
            {displayRows.length} {aggregation === 'day' ? 'records' : aggregation === 'month' ? 'months' : 'years'}
          </span>
          <button
            onClick={loadSampleData}
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

      {/* Aggregation Controls */}
      <div style={{ 
        display: 'flex', 
        gap: '0.5rem', 
        alignItems: 'center', 
        marginTop: '1rem',
        padding: '0.75rem',
        backgroundColor: '#1e293b',
        borderRadius: '8px',
        flexWrap: 'wrap'
      }}>
        <FiCalendar style={{ color: '#94a3b8' }} />
        <span style={{ color: '#94a3b8', fontSize: '0.875rem', marginRight: '0.5rem' }}>
          View By:
        </span>
        {['day', 'month', 'year'].map((level) => (
          <button
            key={level}
            onClick={() => handleAggregationChange(level)}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: aggregation === level ? '#3b82f6' : 'transparent',
              color: aggregation === level ? 'white' : '#94a3b8',
              border: aggregation === level ? 'none' : '1px solid #334155',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: aggregation === level ? 'bold' : 'normal',
              transition: 'all 0.2s ease',
            }}
          >
            {level === 'day' ? 'Daily' : level === 'month' ? 'Monthly' : 'Yearly'}
          </button>
        ))}
      </div>

      {/* Year, Month, Day Selection */}
      <div style={{ 
        display: 'flex', 
        gap: '1rem', 
        alignItems: 'center', 
        marginTop: '0.75rem',
        padding: '0.75rem',
        backgroundColor: '#0f172a',
        borderRadius: '8px',
        flexWrap: 'wrap'
      }}>
        {/* Year Selection - Always visible */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label style={{ color: '#94a3b8', fontSize: '0.875rem', fontWeight: 'bold' }}>Year:</label>
          <select
            value={selectedYear}
            onChange={(e) => handleYearChange(e.target.value)}
            style={selectStyle}
          >
            <option value="all">All Years (1998-2026)</option>
            {availableYears.map(year => (
              <option key={year} value={year}>{year}</option>
            ))}
          </select>
        </div>

        {/* Month Selection - Visible when year is selected and not yearly aggregation */}
        {selectedYear !== 'all' && aggregation !== 'year' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ color: '#94a3b8', fontSize: '0.875rem', fontWeight: 'bold' }}>Month:</label>
            <select
              value={selectedMonth}
              onChange={(e) => handleMonthChange(e.target.value)}
              style={selectStyle}
            >
              <option value="all">All Months</option>
              {availableMonthsForYear.map(month => (
                <option key={month} value={month}>{monthNames[month]} ({month})</option>
              ))}
            </select>
            {availableMonthsForYear.length === 0 && (
              <span style={{ fontSize: '0.75rem', color: '#f59e0b' }}>No data for {selectedYear}</span>
            )}
          </div>
        )}

        {/* Day Selection - Visible when month is selected and daily aggregation */}
        {selectedYear !== 'all' && selectedMonth !== 'all' && aggregation === 'day' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <label style={{ color: '#94a3b8', fontSize: '0.875rem', fontWeight: 'bold' }}>Day:</label>
            <select
              value={selectedDay}
              onChange={(e) => handleDayChange(e.target.value)}
              style={selectStyle}
            >
              <option value="all">All Days</option>
              {availableDaysForMonth.map(day => (
                <option key={day} value={day}>{parseInt(day)}</option>
              ))}
            </select>
            {availableDaysForMonth.length === 0 && (
              <span style={{ fontSize: '0.75rem', color: '#f59e0b' }}>No data</span>
            )}
          </div>
        )}

        {/* Clear Filters */}
        {hasActiveFilters && (
          <button
            onClick={() => {
              setSelectedYear('all');
              setSelectedMonth('all');
              setSelectedDay('all');
              setCurrentPage(0);
            }}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#ef4444',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.875rem',
            }}
          >
            Clear Filters
          </button>
        )}

        {/* Filter Status */}
        <span style={{ 
          marginLeft: 'auto', 
          fontSize: '0.8rem', 
          color: '#10b981',
          fontWeight: 'bold',
          padding: '0.25rem 0.5rem',
          backgroundColor: '#1e293b',
          borderRadius: '4px',
        }}>
          {getFilterDescription()}
        </span>
      </div>

      {/* Data Table */}
      <div style={{ overflowX: 'auto', marginTop: '1rem' }}>
        <table style={{ 
          width: '100%', 
          borderCollapse: 'collapse',
          fontSize: '0.8rem',
        }}>
          <thead>
            <tr style={{ backgroundColor: '#1e293b' }}>
              {displayColumns.map((col, idx) => (
                <th
                  key={idx}
                  style={{
                    padding: '0.75rem 0.5rem',
                    textAlign: col === 'date' || col === 'period' ? 'left' : 'right',
                    color: '#e2e8f0',
                    borderBottom: '2px solid #334155',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {formatColumnName(col)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {currentRows.length === 0 ? (
              <tr>
                <td 
                  colSpan={displayColumns.length} 
                  style={{ 
                    padding: '2rem', 
                    textAlign: 'center', 
                    color: '#94a3b8' 
                  }}
                >
                  No data found for the selected filters.
                </td>
              </tr>
            ) : (
              currentRows.map((row, rowIdx) => (
                <tr
                  key={rowIdx}
                  style={{
                    backgroundColor: rowIdx % 2 === 0 ? '#0f172a' : '#1e293b',
                  }}
                >
                  {displayColumns.map((col, colIdx) => (
                    <td
                      key={colIdx}
                      style={{
                        padding: '0.5rem',
                        textAlign: col === 'date' || col === 'period' ? 'left' : 'right',
                        color: '#94a3b8',
                        borderBottom: '1px solid #334155',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {formatValue(row[col], col)}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          gap: '1rem',
          marginTop: '1rem',
          padding: '0.5rem',
        }}>
          <button
            onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
            disabled={currentPage === 0}
            style={{
              padding: '0.5rem',
              backgroundColor: currentPage === 0 ? '#334155' : '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: currentPage === 0 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <FiChevronLeft />
          </button>
          <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
            Page {currentPage + 1} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
            disabled={currentPage === totalPages - 1}
            style={{
              padding: '0.5rem',
              backgroundColor: currentPage === totalPages - 1 ? '#334155' : '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: currentPage === totalPages - 1 ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <FiChevronRight />
          </button>
        </div>
      )}

      {/* Info */}
      <div style={{ 
        marginTop: '1rem', 
        padding: '0.5rem 0.75rem', 
        backgroundColor: '#1e293b', 
        borderRadius: '8px',
        fontSize: '0.75rem',
        color: '#64748b',
        textAlign: 'center'
      }}>
        {aggregation === 'day' 
          ? 'Showing individual daily transactions.'
          : aggregation === 'month'
          ? 'Data aggregated (summed) by month.'
          : 'Data aggregated (summed) by year.'
        }
        {' '}Select year, month, and day to filter data.
      </div>
    </div>
  );
};

export default SampleDataDisplay;
