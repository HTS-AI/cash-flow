# FastAPI Backend - Quick Start

## Installation

1. **Install Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

## Running the Server

### Option 1: Using uvicorn directly
```bash
cd backend
uvicorn main:app --reload --port 5000
```

### Option 2: Using the start script
```bash
cd backend
python start_server.py
```

### Option 3: Using Python module
```bash
cd backend
python -m uvicorn main:app --reload --port 5000
```

## API Documentation

Once the server is running, visit:

- **Swagger UI (Interactive)**: http://localhost:5000/docs
- **ReDoc (Alternative)**: http://localhost:5000/redoc
- **OpenAPI JSON**: http://localhost:5000/openapi.json

## API Endpoints

All endpoints are prefixed with `/api`:

- `GET /api/health` - Health check
- `GET /api/model/info` - Get model information
- `POST /api/predict` - Generate predictions
- `GET /api/predictions` - Get existing predictions
- `GET /api/data/historical` - Get historical data
- `GET /api/data/summary` - Get summary statistics
- `POST /api/model/train` - Train new model
- `GET /api/files/status` - Check file status

## Testing the API

### Using Swagger UI (Recommended)
1. Start the server
2. Open http://localhost:5000/docs
3. Click on any endpoint
4. Click "Try it out"
5. Fill in parameters
6. Click "Execute"

### Using curl

**Health check:**
```bash
curl http://localhost:5000/api/health
```

**Get model info:**
```bash
curl http://localhost:5000/api/model/info
```

**Make prediction:**
```bash
curl -X POST "http://localhost:5000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"forecastMonths": 3}'
```

**Get summary:**
```bash
curl http://localhost:5000/api/data/summary
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:5000/api/health")
print(response.json())

# Make prediction
response = requests.post(
    "http://localhost:5000/api/predict",
    json={"forecastMonths": 3}
)
print(response.json())
```

## File Structure

```
backend/
├── main.py                 # FastAPI application
├── start_server.py         # Server startup script
├── ml_components/          # ML Python modules
│   ├── __init__.py
│   ├── app.py
│   ├── data_preparation.py
│   ├── data_analysis.py
│   ├── data_generate.py
│   ├── lstm_cashflow_prediction.py
│   └── shap_explainability.py
├── requirements.txt        # Python dependencies
└── README_FASTAPI.md       # This file
```

## Required Files (in parent directory)

The backend expects these files in the root directory:

- `cashflow_prediction_1998_2025_v1.csv` - Data file
- `best_model.pkl` - Trained model (after training)
- `best_model_info.json` - Model metadata (after training)
- `future_predictions.csv` - Predictions output (after prediction)

## Troubleshooting

### Port already in use
Change the port in `start_server.py` or use:
```bash
uvicorn main:app --reload --port 8000
```

### Module not found errors
Make sure all files are in `backend/ml_components/` directory.

### Model not found
Train a model first:
```bash
cd ..
python app.py train 1
```

## Development

The server runs with `--reload` flag by default, so it will automatically restart when you make changes to the code.
