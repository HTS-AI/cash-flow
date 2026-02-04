import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { cashflowAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';
import PredictionsPanel from '../components/PredictionsPanel';
import ShapExplainability from '../components/ShapExplainability';
import MonthlySpendingChart from '../components/MonthlySpendingChart';
import ExpenseCategoriesChart from '../components/ExpenseCategoriesChart';
import IncomeCategoriesChart from '../components/IncomeCategoriesChart';
import IncomeExpenseChart from '../components/IncomeExpenseChart';
import NetCashflowYearChart from '../components/NetCashflowYearChart';
import FinancialSummaryChart from '../components/FinancialSummaryChart';
import PredictedVsActualChart from '../components/PredictedVsActualChart';
import ForecastAccuracyChart from '../components/ForecastAccuracyChart';
import SampleDataDisplay from '../components/SampleDataDisplay';
import { FiRefreshCw, FiLogOut, FiUser } from 'react-icons/fi';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [yearOverYearData, setYearOverYearData] = useState([]);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // First check if backend is available
      try {
        await cashflowAPI.healthCheck();
      } catch (healthError) {
        setError('Backend server is not running. Please start the backend server on port 5000.');
        setLoading(false);
        return;
      }

      // Load all data in parallel
      const [summaryRes, historicalRes, predictionsRes, modelRes, yearOverYearRes] = await Promise.all([
        cashflowAPI.getSummary().catch(err => ({ data: { success: false, error: err.message } })),
        cashflowAPI.getHistoricalData().catch(err => ({ data: { success: false, error: err.message } })),
        cashflowAPI.getPredictions().catch(err => ({ data: { success: false, error: err.message } })),
        cashflowAPI.getModelInfo().catch(err => ({ data: { success: false, error: err.message } })),
        cashflowAPI.getYearOverYear().catch(err => ({ data: { success: false, error: err.message } }))
      ]);

      if (summaryRes.data.success) {
        setSummary(summaryRes.data.data);
      }

      if (historicalRes.data.success) {
        // Prefer the full historical series if provided by the backend
        setHistoricalData(historicalRes.data.allData || historicalRes.data.data || []);
      }

      if (predictionsRes.data.success) {
        setPredictions(predictionsRes.data.data);
      }

      if (modelRes.data.success) {
        setModelInfo(modelRes.data.data);
      }

      if (yearOverYearRes.data.success) {
        setYearOverYearData(yearOverYearRes.data.data);
      }

      // Check for any errors in responses
      const errors = [];
      if (!summaryRes.data.success) errors.push('Summary: ' + (summaryRes.data.error || summaryRes.data.message));
      if (!historicalRes.data.success) errors.push('Historical data: ' + (historicalRes.data.error || historicalRes.data.message));
      if (!predictionsRes.data.success && predictionsRes.data.message !== 'No predictions found. Please generate predictions first.') {
        errors.push('Predictions: ' + (predictionsRes.data.error || predictionsRes.data.message));
      }
      if (!modelRes.data.success && modelRes.data.message !== 'Model not found. Please train a model first.') {
        errors.push('Model info: ' + (modelRes.data.error || modelRes.data.message));
      }

      if (errors.length > 0) {
        setError('Some data could not be loaded: ' + errors.join('; '));
      }
    } catch (err) {
      console.error('Error loading dashboard data:', err);
      if (err.response) {
        if (err.response.status === 404) {
          setError('Backend API not found. Make sure the backend server is running on http://localhost:5000');
        } else {
          setError(`API Error (${err.response.status}): ${err.response.data?.error || err.message}`);
        }
      } else if (err.request) {
        setError('Cannot connect to backend server. Please ensure the backend is running on http://localhost:5000');
      } else {
        setError(err.message || 'Failed to load dashboard data');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const handleGeneratePredictions = async (months) => {
    try {
      setRefreshing(true);
      const response = await cashflowAPI.makePrediction(months);
      if (response.data.success) {
        setPredictions(response.data.data);
        await loadDashboardData(); // Reload to get updated data
      }
    } catch (err) {
      console.error('Error generating predictions:', err);
      setError(err.message || 'Failed to generate predictions');
    } finally {
      setRefreshing(false);
    }
  };

  const handleModelRetrained = async () => {
    // Reload dashboard data to get updated model info
    await loadDashboardData();
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header */}
      <div className="dashboard-header">
        <h1 className="dashboard-title">Cash Flow Prediction</h1>
        <div className="header-actions">
          <div className="user-info">
            <FiUser className="user-icon" />
            <span className="user-name">{user?.userId}</span>
          </div>
          <button className="icon-button" onClick={handleRefresh} disabled={refreshing} title="Refresh">
            <FiRefreshCw style={{ animation: refreshing ? 'spin 1s linear infinite' : 'none' }} />
          </button>
          <button className="icon-button logout-button" onClick={handleLogout} title="Logout">
            <FiLogOut />
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* 1. Future Predictions Panel (TOP) */}
      <PredictionsPanel
        predictions={predictions}
        modelInfo={modelInfo}
        onGeneratePredictions={handleGeneratePredictions}
        onModelRetrained={handleModelRetrained}
      />

      {/* 2. SHAP Explainability */}
      <div className="charts-container">
        <ShapExplainability />
      </div>

      {/* 3. Predicted vs Actual Comparison */}
      <div className="charts-container">
        <PredictedVsActualChart />
      </div>

      {/* 4. Forecast Accuracy Tracking */}
      <div className="charts-container">
        <ForecastAccuracyChart />
      </div>

      {/* 5. All Graphical Representations */}
      <div className="charts-container">
        <MonthlySpendingChart data={historicalData} predictions={predictions} />
      </div>

      {/* 5. Category Breakdown Charts */}
      {summary && (
        <div className="charts-container">
          <ExpenseCategoriesChart expenseBreakdown={summary.expenseBreakdown} />
          <IncomeCategoriesChart incomeBreakdown={summary.incomeBreakdown} />
        </div>
      )}

      {/* Year-over-Year Comparison Charts */}
      {yearOverYearData.length > 0 && (
        <>
          <div className="charts-container">
            <IncomeExpenseChart data={yearOverYearData} />
          </div>
          <div className="charts-container">
            <NetCashflowYearChart data={yearOverYearData} />
            <FinancialSummaryChart data={yearOverYearData} />
          </div>
        </>
      )}

      {/* 4. Sample Data Display (BOTTOM) */}
      <div className="charts-container">
        <SampleDataDisplay />
      </div>
    </div>
  );
};

export default Dashboard;
