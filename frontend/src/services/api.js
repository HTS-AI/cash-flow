import axios from 'axios';

// In production (same origin), use relative URL '/api'
// In development, use 'http://localhost:5000/api'
const API_BASE_URL = process.env.REACT_APP_API_URL || 
  (process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:5000/api');

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const cashflowAPI = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Model info
  getModelInfo: () => api.get('/model/info'),
  getFeatureImportance: () => api.get('/model/feature-importance'),

  // Predictions
  makePrediction: (forecastMonths) => api.post('/predict', { forecastMonths }),
  getPredictions: () => api.get('/predictions'),

  // Data
  getHistoricalData: () => api.get('/data/historical'),
  getSummary: () => api.get('/data/summary'),
  getYearOverYear: (month = 0) => api.get(`/data/year-over-year?month=${month}`),
  getSampleData: () => api.get('/data/sample'),
  getPredictedVsActual: (months = 12) => api.get(`/data/predicted-vs-actual?months=${months}`),
  getForecastAccuracy: () => api.get('/data/forecast-accuracy'),

  // Training
  trainModel: (forecastMonths) => api.post('/model/train', { forecastMonths }),
};

export default api;
