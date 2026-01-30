import axios from 'axios';

// FastAPI backend runs on port 5000
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

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
  getYearOverYear: () => api.get('/data/year-over-year'),
  getSampleData: () => api.get('/data/sample'),

  // Training
  trainModel: (forecastMonths) => api.post('/model/train', { forecastMonths }),
};

export default api;
