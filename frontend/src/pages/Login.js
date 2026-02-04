import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { FiUser, FiLock, FiLogIn } from 'react-icons/fi';
import './Login.css';

const Login = () => {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    // Simulate a brief delay for better UX
    await new Promise(resolve => setTimeout(resolve, 500));

    const result = login(userId, password);
    
    if (result.success) {
      navigate('/dashboard');
    } else {
      setError(result.error);
    }
    
    setIsLoading(false);
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <div className="login-logo">
            <span className="logo-icon">💰</span>
          </div>
          <h1 className="login-title">Cash Flow Forecast</h1>
          <p className="login-subtitle">Sign in to access your dashboard</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="login-error">
              <span>{error}</span>
            </div>
          )}

          <div className="form-group">
            <label htmlFor="userId" className="form-label">
              <FiUser className="form-icon" />
              User ID
            </label>
            <input
              type="text"
              id="userId"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="form-input"
              placeholder="Enter your user ID"
              required
              autoComplete="username"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password" className="form-label">
              <FiLock className="form-icon" />
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="form-input"
              placeholder="Enter your password"
              required
              autoComplete="current-password"
            />
          </div>

          <button 
            type="submit" 
            className="login-button"
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="button-loading">
                <span className="spinner-small"></span>
                Signing in...
              </span>
            ) : (
              <span className="button-content">
                <FiLogIn className="button-icon" />
                Sign In
              </span>
            )}
          </button>
        </form>

        <div className="login-footer">
          <p>Powered by HTS AI</p>
        </div>
      </div>
    </div>
  );
};

export default Login;
