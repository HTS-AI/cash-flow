import React, { useState } from 'react';
import { FiTrendingUp, FiTrendingDown, FiInfo } from 'react-icons/fi';

const BalanceCard = ({ balance, change, summary }) => {
  const [isFlipped, setIsFlipped] = useState(false);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(amount);
  };

  const isPositive = change >= 0;

  const handleCardClick = () => {
    setIsFlipped(!isFlipped);
  };

  return (
    <div className={`flip-card ${isFlipped ? 'flipped' : ''}`} onClick={handleCardClick}>
      <div className="flip-card-inner">
        {/* Front of card */}
        <div className="flip-card-front card">
          <div className="card-header">
            <div>
              <div className="card-title">TOTAL BALANCE</div>
              <div className="card-value">{formatCurrency(balance)}</div>
              <div className={`card-change ${isPositive ? 'positive' : 'negative'}`}>
                {isPositive ? <FiTrendingUp /> : <FiTrendingDown />}
                <span>{isPositive ? '+' : ''}{change.toFixed(2)}% from last year {summary?.previousYear ? `(${summary.previousYear})` : ''}</span>
              </div>
            </div>
          </div>
          <div className="card-hint" style={{ marginTop: '1rem', fontSize: '0.75rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <FiInfo />
            <span>Click to see details</span>
          </div>
        </div>

        {/* Back of card */}
        <div className="flip-card-back card">
          <div className="card-header">
            <div className="card-title">TOTAL BALANCE Details</div>
          </div>
          <div className="card-details" style={{ overflowY: 'auto', maxHeight: 'calc(100% - 80px)' }}>
            {summary && (
              <>
                <div className="detail-row">
                  <span className="detail-label">Total Balance:</span>
                  <span className="detail-value">{formatCurrency(balance)}</span>
                </div>
                <div className="detail-divider"></div>
                <div className="detail-explanation">
                  <strong>Calculation:</strong>
                  <p>Total Balance represents the cumulative financial position, calculated from historical cash flows.</p>
                  <p>The change percentage reflects the year-over-year growth in total balance, comparing to {summary?.previousYear || 'the previous year'}.</p>
                  {summary?.currentYear && summary?.previousYear && (
                    <p>This compares the total balance for {summary.currentYear} against the total balance for {summary.previousYear}, showing the percentage change between these two years.</p>
                  )}
                  {summary.expenseBreakdown && Object.keys(summary.expenseBreakdown).length > 0 && (
                    <>
                      <p style={{ marginTop: '1rem' }}><strong>Expense Breakdown:</strong></p>
                      {Object.entries(summary.expenseBreakdown).map(([category, amount]) => (
                        <p key={category} style={{ marginLeft: '1rem', fontSize: '0.875rem' }}>
                          {category}: {formatCurrency(amount)}
                        </p>
                      ))}
                    </>
                  )}
                </div>
              </>
            )}
          </div>
          <div className="card-hint" style={{ marginTop: '1rem', fontSize: '0.75rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <FiInfo />
            <span>Click to flip back</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BalanceCard;
