const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs-extra');
const path = require('path');

const execAsync = promisify(exec);

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Path to ML system (root directory where Python files are located)
const ML_SYSTEM_PATH = path.join(__dirname, '..');

// Helper function to run Python scripts
async function runPythonScript(scriptPath, args = []) {
  try {
    const command = `python "${path.join(ML_SYSTEM_PATH, scriptPath)}" ${args.join(' ')}`;
    const { stdout, stderr } = await execAsync(command, {
      cwd: ML_SYSTEM_PATH,
      maxBuffer: 10 * 1024 * 1024 // 10MB buffer
    });
    
    if (stderr && !stderr.includes('Warning')) {
      console.warn('Python stderr:', stderr);
    }
    
    return { stdout, stderr };
  } catch (error) {
    throw new Error(`Python script execution failed: ${error.message}`);
  }
}

// Helper function to read CSV file
function readCSV(filePath) {
  try {
    const data = fs.readFileSync(filePath, 'utf8');
    const lines = data.split('\n').filter(line => line.trim());
    const headers = lines[0].split(',');
    const rows = lines.slice(1).map(line => {
      const values = line.split(',');
      const obj = {};
      headers.forEach((header, index) => {
        obj[header.trim()] = values[index]?.trim() || '';
      });
      return obj;
    });
    return rows;
  } catch (error) {
    throw new Error(`Error reading CSV: ${error.message}`);
  }
}

// API Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Cash Flow Prediction API is running' });
});

// Get model info
app.get('/api/model/info', (req, res) => {
  try {
    const modelInfoPath = path.join(ML_SYSTEM_PATH, 'best_model_info.json');
    if (fs.existsSync(modelInfoPath)) {
      const modelInfo = JSON.parse(fs.readFileSync(modelInfoPath, 'utf8'));
      res.json({ success: true, data: modelInfo });
    } else {
      res.json({ success: false, message: 'Model not found. Please train a model first.' });
    }
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Make predictions
app.post('/api/predict', async (req, res) => {
  try {
    const { forecastMonths = 1 } = req.body;
    
    if (forecastMonths < 1 || forecastMonths > 12) {
      return res.status(400).json({ success: false, error: 'forecastMonths must be between 1 and 12' });
    }

    // Check if model exists
    const modelPath = path.join(ML_SYSTEM_PATH, 'best_model.pkl');
    if (!fs.existsSync(modelPath)) {
      return res.status(404).json({ 
        success: false, 
        error: 'Model not found. Please train a model first.' 
      });
    }

    // Run prediction using app.py
    const result = await runPythonScript('app.py', ['predict', forecastMonths.toString()]);
    
    // Read predictions CSV
    const predictionsPath = path.join(ML_SYSTEM_PATH, 'future_predictions.csv');
    if (fs.existsSync(predictionsPath)) {
      const predictions = readCSV(predictionsPath);
      res.json({ success: true, data: predictions });
    } else {
      res.status(500).json({ success: false, error: 'Predictions file not generated' });
    }
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get historical data
app.get('/api/data/historical', async (req, res) => {
  try {
    const dataPath = path.join(ML_SYSTEM_PATH, 'cashflow_prediction_1998_2025_v1.csv');
    if (!fs.existsSync(dataPath)) {
      return res.status(404).json({ success: false, error: 'Data file not found' });
    }

    // Read and aggregate data (last 12 months for dashboard)
    const data = readCSV(dataPath);
    
    // Parse dates and aggregate by month
    const monthlyData = {};
    data.forEach(row => {
      if (row.date) {
        const date = new Date(row.date);
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        
        if (!monthlyData[monthKey]) {
          monthlyData[monthKey] = {
            month: monthKey,
            cash_inflow: 0,
            cash_outflow: 0,
            vendor_payment: 0,
            salary_payment: 0,
            rent: 0,
            operational_expense: 0
          };
        }
        
        monthlyData[monthKey].cash_inflow += parseFloat(row.cash_inflow_usd || 0);
        monthlyData[monthKey].cash_outflow += parseFloat(row.cash_outflow_usd || 0);
        monthlyData[monthKey].vendor_payment += parseFloat(row.vendor_payment_usd || 0);
        monthlyData[monthKey].salary_payment += parseFloat(row.salary_payment_usd || 0);
        monthlyData[monthKey].rent += parseFloat(row.rent_usd || 0);
        monthlyData[monthKey].operational_expense += parseFloat(row.operational_expense_usd || 0);
      }
    });

    // Convert to array and sort
    const monthlyArray = Object.values(monthlyData).sort((a, b) => a.month.localeCompare(b.month));
    
    // Get last 12 months
    const last12Months = monthlyArray.slice(-12);
    
    res.json({ success: true, data: last12Months, allData: monthlyArray });
  } catch (error) {
    console.error('Data fetch error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get summary statistics
app.get('/api/data/summary', async (req, res) => {
  try {
    const dataPath = path.join(ML_SYSTEM_PATH, 'cashflow_prediction_1998_2025_v1.csv');
    if (!fs.existsSync(dataPath)) {
      return res.status(404).json({ success: false, error: 'Data file not found' });
    }

    const data = readCSV(dataPath);
    
    // Calculate summary statistics
    let totalInflow = 0;
    let totalOutflow = 0;
    let totalVendor = 0;
    let totalSalary = 0;
    let totalRent = 0;
    let totalOperational = 0;
    
    data.forEach(row => {
      totalInflow += parseFloat(row.cash_inflow_usd || 0);
      totalOutflow += parseFloat(row.cash_outflow_usd || 0);
      totalVendor += parseFloat(row.vendor_payment_usd || 0);
      totalSalary += parseFloat(row.salary_payment_usd || 0);
      totalRent += parseFloat(row.rent_usd || 0);
      totalOperational += parseFloat(row.operational_expense_usd || 0);
    });

    // Get last month data for comparison
    const lastMonth = data.slice(-30); // Last 30 days
    let lastMonthInflow = 0;
    let lastMonthOutflow = 0;
    
    lastMonth.forEach(row => {
      lastMonthInflow += parseFloat(row.cash_inflow_usd || 0);
      lastMonthOutflow += parseFloat(row.cash_outflow_usd || 0);
    });

    // Calculate previous month for comparison
    const previousMonth = data.slice(-60, -30);
    let prevMonthInflow = 0;
    let prevMonthOutflow = 0;
    
    previousMonth.forEach(row => {
      prevMonthInflow += parseFloat(row.cash_inflow_usd || 0);
      prevMonthOutflow += parseFloat(row.cash_outflow_usd || 0);
    });

    const inflowChange = prevMonthInflow > 0 ? ((lastMonthInflow - prevMonthInflow) / prevMonthInflow) * 100 : 0;
    const outflowChange = prevMonthOutflow > 0 ? ((lastMonthOutflow - prevMonthOutflow) / prevMonthOutflow) * 100 : 0;

    const summary = {
      totalBalance: totalInflow - totalOutflow,
      totalInflow,
      totalOutflow,
      lastMonthInflow,
      lastMonthOutflow,
      inflowChange: parseFloat(inflowChange.toFixed(2)),
      outflowChange: parseFloat(outflowChange.toFixed(2)),
      expenseBreakdown: {
        vendor: totalVendor,
        salary: totalSalary,
        rent: totalRent,
        operational: totalOperational
      }
    };

    res.json({ success: true, data: summary });
  } catch (error) {
    console.error('Summary error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Train model
app.post('/api/model/train', async (req, res) => {
  try {
    const { forecastMonths = 1 } = req.body;
    
    if (forecastMonths < 1 || forecastMonths > 12) {
      return res.status(400).json({ success: false, error: 'forecastMonths must be between 1 and 12' });
    }

    // Run training
    await runPythonScript('app.py', ['train', forecastMonths.toString()]);
    
    // Read updated model info
    const modelInfoPath = path.join(ML_SYSTEM_PATH, 'best_model_info.json');
    if (fs.existsSync(modelInfoPath)) {
      const modelInfo = JSON.parse(fs.readFileSync(modelInfoPath, 'utf8'));
      res.json({ success: true, data: modelInfo, message: 'Model trained successfully' });
    } else {
      res.status(500).json({ success: false, error: 'Model training completed but info file not found' });
    }
  } catch (error) {
    console.error('Training error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get predictions (if they exist)
app.get('/api/predictions', (req, res) => {
  try {
    const predictionsPath = path.join(ML_SYSTEM_PATH, 'future_predictions.csv');
    if (fs.existsSync(predictionsPath)) {
      const predictions = readCSV(predictionsPath);
      res.json({ success: true, data: predictions });
    } else {
      res.json({ success: false, message: 'No predictions found. Please generate predictions first.' });
    }
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Cash Flow Prediction API server running on port ${PORT}`);
  console.log(`📊 ML System Path: ${ML_SYSTEM_PATH}`);
  console.log(`📁 Looking for Python files in: ${ML_SYSTEM_PATH}`);
  
  // Verify key files exist
  const keyFiles = [
    'app.py',
    'data_preparation.py',
    'cashflow_prediction_1998_2025_v1.csv'
  ];
  
  console.log(`\n📋 Checking required files:`);
  keyFiles.forEach(file => {
    const filePath = path.join(ML_SYSTEM_PATH, file);
    const exists = fs.existsSync(filePath);
    console.log(`   ${exists ? '✅' : '❌'} ${file}`);
    if (!exists && file === 'app.py') {
      console.log(`   ⚠️  Warning: ${file} not found. Predictions and training may fail.`);
    }
  });
  
  // Check for model
  const modelPath = path.join(ML_SYSTEM_PATH, 'best_model.pkl');
  if (fs.existsSync(modelPath)) {
    console.log(`   ✅ best_model.pkl (model ready)`);
  } else {
    console.log(`   ⚠️  best_model.pkl not found. Train a model first: python app.py train 1`);
  }
  
  console.log(`\n🌐 API available at: http://localhost:${PORT}/api`);
});
