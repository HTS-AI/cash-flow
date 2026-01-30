# Cash Flow Prediction Web Application

A modern React + Node.js web application for cash flow prediction with a beautiful finance dashboard UI.

## Features

- 📊 **Finance Dashboard**: Dark mode UI with real-time financial metrics
- 🔮 **Predictions**: Generate cash flow predictions using ML models
- 📈 **Visualizations**: Interactive charts for spending, expenses, and trends
- 💰 **Balance Tracking**: Monitor total balance, income, and expenses
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
├── backend/          # Node.js Express API server
│   ├── server.js    # Main server file
│   └── package.json # Backend dependencies
├── frontend/         # React application
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── pages/       # Page components
│   │   ├── services/    # API services
│   │   └── App.js       # Main app component
│   └── package.json     # Frontend dependencies
└── ml_system/       # Python ML system (existing)
```

## Installation

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
npm install
```

3. Start the server:
```bash
npm start
# or for development with auto-reload:
npm run dev
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React app:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## Usage

1. **Start Backend**: Make sure the Node.js backend is running
2. **Start Frontend**: Start the React development server
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Generate Predictions**: Click "Generate Predictions" to create forecasts

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/model/info` - Get model information
- `POST /api/predict` - Generate predictions
- `GET /api/predictions` - Get existing predictions
- `GET /api/data/historical` - Get historical data
- `GET /api/data/summary` - Get summary statistics
- `POST /api/model/train` - Train new model

## Requirements

- Node.js 14+ and npm
- Python 3.8+ (for ML system)
- ML system files in `ml_system/` directory
- Trained model (`best_model.pkl`) in `ml_system/` directory

## Features Overview

### Dashboard Components

1. **Balance Card**: Shows total balance with month-over-month change
2. **Metric Cards**: Display income, expenses, and net cashflow
3. **Monthly Spending Chart**: Line chart showing spending trends
4. **Expense Categories Chart**: Pie chart breaking down expenses
5. **Predictions Panel**: Generate and view future predictions
6. **Recent Transactions**: List of recent financial activity

### Dark Mode UI

The application features a modern dark mode interface with:
- Gradient accents
- Smooth animations
- Responsive design
- Interactive charts

## Development

### Backend Development

The backend uses Express.js and communicates with the Python ML system via `python-shell`. Make sure Python is in your PATH.

### Frontend Development

The frontend uses React with:
- Recharts for data visualization
- React Icons for icons
- Axios for API calls
- Custom CSS for styling

## Troubleshooting

1. **Model not found**: Train a model first using `python ml_system/app.py train`
2. **API connection errors**: Ensure backend is running on port 5000
3. **Python errors**: Check that Python and required packages are installed
4. **Port conflicts**: Change ports in `backend/server.js` and `frontend/package.json`

## License

This project is part of the Cash Flow Prediction System.
