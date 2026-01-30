import React, { useState } from 'react';
import { FiTrendingUp, FiTrendingDown, FiInfo } from 'react-icons/fi';

const MetricCard = ({ title, value, change, icon, iconColor, details, summary }) => {
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
            <div className="card-title">{title}</div>
            <div className="card-icon" style={{ background: `${iconColor}20`, color: iconColor }}>
              {icon}
            </div>
          </div>
          <div className="card-value">{formatCurrency(value)}</div>
          <div className={`card-change ${isPositive ? 'positive' : 'negative'}`}>
            {isPositive ? <FiTrendingUp /> : <FiTrendingDown />}
            <span>{isPositive ? '+' : ''}{change.toFixed(2)}%</span>
          </div>
          <div className="card-hint" style={{ marginTop: '1rem', fontSize: '0.75rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <FiInfo />
            <span>Click to see details</span>
          </div>
        </div>

        {/* Back of card */}
        <div className="flip-card-back card">
          <div className="card-header">
            <div className="card-title">{title} Details</div>
            <div className="card-icon" style={{ background: `${iconColor}20`, color: iconColor }}>
              {icon}
            </div>
          </div>
          <div className="card-details" style={{ overflowY: 'auto', maxHeight: 'calc(100% - 80px)' }}>
            {details || (
              <>
                <div className="detail-row">
                  <span className="detail-label">Current Value:</span>
                  <span className="detail-value">{formatCurrency(value)}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Change from Last Month:</span>
                  <span className={`detail-value ${isPositive ? 'positive' : 'negative'}`}>
                    {isPositive ? '+' : ''}{change.toFixed(2)}%
                  </span>
                </div>
                {title === 'Income' && summary && (
                  <>
                    <div className="detail-divider"></div>
                    <div className="detail-explanation">
                      <strong>Calculation:</strong>
                      <p>Income represents the total cash inflow for the last month, including all revenue sources.</p>
                      <p>Change percentage compares this month's income to the previous month.</p>
                    </div>
                  </>
                )}
                {title === 'Expense' && summary && (
                  <>
                    <div className="detail-divider"></div>
                    <div className="detail-explanation">
                      <strong>Calculation:</strong>
                      <p>Expense represents the total cash outflow for the last month, including vendor payments, salaries, rent, and operational expenses.</p>
                      <p>Change percentage compares this month's expenses to the previous month.</p>
                    </div>
                  </>
                )}
                {title === 'Net Cashflow' && summary && (
                  <>
                    <div className="detail-divider"></div>
                    <div className="detail-explanation">
                      <strong>Calculation:</strong>
                      <p>Net Cashflow = Income - Expense</p>
                      <p>This shows the difference between money coming in and going out. A positive value indicates more income than expenses.</p>
                      <p>Change percentage is calculated as: Income Change - Expense Change</p>
                    </div>
                  </>
                )}
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

export default MetricCard;
