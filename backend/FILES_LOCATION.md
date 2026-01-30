# File Locations in Backend

## All Files Are Now in Backend Directory

All necessary files have been moved to the `backend/` directory for better organization.

## File Structure

```
backend/
├── main.py                           # FastAPI application
├── start_server.py                   # Server startup script
├── cashflow_prediction_1998_2025_v1.csv  # Data file
├── best_model.pkl                    # Trained model
├── best_model_info.json              # Model metadata
├── future_predictions.csv            # Predictions output (generated)
├── ml_components/                    # ML Python modules
│   ├── app.py
│   ├── data_preparation.py
│   ├── data_analysis.py
│   ├── data_generate.py
│   ├── lstm_cashflow_prediction.py
│   └── shap_explainability.py
└── requirements.txt                  # Python dependencies
```

## File Paths in Code

The FastAPI server (`main.py`) uses these paths:

```python
BACKEND_DIR = Path(__file__).parent  # Points to backend/
DATA_FILE = BACKEND_DIR / 'cashflow_prediction_1998_2025_v1.csv'
MODEL_FILE = BACKEND_DIR / 'best_model.pkl'
MODEL_INFO_FILE = BACKEND_DIR / 'best_model_info.json'
PREDICTIONS_FILE = BACKEND_DIR / 'future_predictions.csv'
```

All files are relative to the `backend/` directory.

## Benefits

1. ✅ **Self-contained** - All backend files in one directory
2. ✅ **Portable** - Easy to move or deploy the backend
3. ✅ **Organized** - Clear separation from frontend
4. ✅ **No root clutter** - Root directory stays clean

## Running the Backend

From the `backend/` directory:

```bash
cd backend
python start_server.py
```

The server will automatically find all files in the same directory.
