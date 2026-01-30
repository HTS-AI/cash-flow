# Backend Setup Guide

## File Structure

The backend expects the following structure:

```
cashflow_v1/
├── backend/              # Backend Node.js server
│   ├── server.js        # Main server file
│   ├── package.json     # Node.js dependencies
│   └── ...
├── app.py               # Python ML integration script
├── data_preparation.py  # Data preprocessing
├── data_analysis.py     # Data analysis
├── data_generate.py     # Data generation
├── lstm_cashflow_prediction.py  # Model training
├── shap_explainability.py       # SHAP analysis
├── cashflow_prediction_1998_2025_v1.csv  # Data file
├── best_model.pkl       # Trained model (after training)
└── best_model_info.json # Model info (after training)
```

## Required Files

### Python Files (in root directory):
- ✅ `app.py` - Main integration script
- ✅ `data_preparation.py` - Data preprocessing module
- ✅ `data_analysis.py` - Data analysis module
- ✅ `data_generate.py` - Data generation script
- ✅ `lstm_cashflow_prediction.py` - Model training script
- ✅ `shap_explainability.py` - SHAP explainability script

### Data Files (in root directory):
- ✅ `cashflow_prediction_1998_2025_v1.csv` - Main data file

### Model Files (generated after training):
- ⚠️ `best_model.pkl` - Trained model (create by running: `python app.py train 1`)
- ⚠️ `best_model_info.json` - Model metadata (created automatically)

## Installation Steps

1. **Install Node.js dependencies:**
   ```bash
   cd backend
   npm install
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install pandas numpy scikit-learn scipy prophet xgboost lightgbm catboost plotly shap openpyxl
   ```

3. **Train a model (first time only):**
   ```bash
   cd ..  # Go to root directory
   python app.py train 1
   ```

4. **Start the server:**
   ```bash
   cd backend
   npm start
   ```

## Verification

When you start the server, it will check for required files and display:

```
🚀 Cash Flow Prediction API server running on port 5000
📊 ML System Path: C:\Users\Admin\cashflow_v1
📁 Looking for Python files in: C:\Users\Admin\cashflow_v1

📋 Checking required files:
   ✅ app.py
   ✅ data_preparation.py
   ✅ cashflow_prediction_1998_2025_v1.csv
   ✅ best_model.pkl (model ready)

🌐 API available at: http://localhost:5000/api
```

If any files are missing (❌), the server will warn you.

## Testing

Test the API with:

```bash
# Health check
curl http://localhost:5000/api/health

# Or open in browser:
http://localhost:5000/api/health
```

You should see:
```json
{"status":"ok","message":"Cash Flow Prediction API is running"}
```

## Troubleshooting

### "Python files not found"
- Make sure all Python files are in the root directory (parent of `backend/`)
- Check that `ML_SYSTEM_PATH` in `server.js` points to the correct location

### "Model not found"
- Train a model: `python app.py train 1`
- Check that `best_model.pkl` exists in the root directory

### "Python not found"
- Make sure Python is installed and in your PATH
- Test with: `python --version`
- On Windows, you might need: `py --version` or `python3 --version`

### Port already in use
- Change `PORT` in `server.js` (line 11)
- Or kill the process using port 5000
