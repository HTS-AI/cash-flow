import React from 'react';

const RecentTransactions = ({ data }) => {
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(amount);
  };

  const formatDate = (month) => {
    if (!month) return '';
    const [year, monthNum] = month.split('-');
    const date = new Date(year, parseInt(monthNum) - 1);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric' 
    });
  };

  // Create transaction-like data from monthly data
  const transactions = data.map((item, index) => ({
    name: `Monthly Summary - ${item.month}`,
    date: formatDate(item.month),
    amount: item.cash_outflow,
    type: 'expense',
  })).reverse();

  if (transactions.length === 0) {
    return (
      <div className="transactions-container">
        <div className="transactions-title">Recent Transactions</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>
          No transactions available
        </div>
      </div>
    );
  }

  return (
    <div className="transactions-container">
      <div className="transactions-title">Recent Transactions</div>
      {transactions.map((transaction, index) => (
        <div key={index} className="transaction-item">
          <div className="transaction-info">
            <div className="transaction-name">{transaction.name}</div>
            <div className="transaction-date">{transaction.date}</div>
          </div>
          <div className={`transaction-amount ${transaction.type === 'income' ? 'positive' : 'negative'}`}>
            {transaction.type === 'income' ? '+' : '-'}{formatCurrency(Math.abs(transaction.amount))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default RecentTransactions;
