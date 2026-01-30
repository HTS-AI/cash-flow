# Backend API Server

Express.js API server for the Cash Flow Prediction System.

## Structure

The backend expects the following Python files in the **parent directory** (root of the project):

- `app.py` - Main integration script
- `data_preparation.py` - Data preprocessing
- `data_analysis.py` - Data analysis
- `data_generate.py` - Data generation
- `lstm_cashflow_prediction.py` - Model training
- `shap_explainability.py` - SHAP analysis

And these data/model files:
- `cashflow_prediction_1998_2025_v1.csv` - Main data file
- `best_model.pkl` - Trained model (generated after training)
- `best_model_info.json` - Model metadata (generated after training)
- `future_predictions.csv` - Predictions output (generated after prediction)

## Installation

```bash
npm install
```

## Running

```bash
npm start
```

Or with auto-reload:
```bash
npm run dev
```

## Configuration

The server looks for Python files in the parent directory (one level up from `backend/`).

To change this, modify `ML_SYSTEM_PATH` in `server.js`:

```javascript
const ML_SYSTEM_PATH = path.join(__dirname, '..'); // Points to parent directory
```

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/model/info` - Get model information
- `POST /api/predict` - Generate predictions
- `GET /api/predictions` - Get existing predictions
- `GET /api/data/historical` - Get historical data
- `GET /api/data/summary` - Get summary statistics
- `POST /api/model/train` - Train new model

## Python Requirements

Make sure Python dependencies are installed. See `requirements.txt` for the list.

Install with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Python not found
- Make sure Python is in your PATH
- Test with: `python --version`

### Model not found
- Train a model first: `python app.py train 1`
- Check that `best_model.pkl` exists in the root directory

### File not found errors
- Verify all Python files are in the root directory (parent of `backend/`)
- Check that `ML_SYSTEM_PATH` in `server.js` points to the correct location
