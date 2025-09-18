# Credit Card Fraud Detection - Quick Start Guide

## Prerequisites
- Python 3.8 or higher
- Git (for cloning)

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd credit-card-fraud-detection

# Create virtual environment
python -m venv fraud_detection_env

# Activate virtual environment
# On macOS/Linux:
source fraud_detection_env/bin/activate
# On Windows:
fraud_detection_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
1. Download the Kaggle Credit Card Fraud Detection dataset
2. Place `creditcard.csv` in the `data/` folder

### 3. Model Training
```bash
# Train the model using Jupyter notebook
jupyter notebook notebooks/train_model.ipynb

# Or run the Python script directly
python src/model/train_model.py
```

### 4. Start the API
```bash
# Start the Flask API server
cd src/api
python flask_api.py
```

The API will be available at `http://127.0.0.1:5000`

### 5. Test the System
1. Open `notebooks/web_ui.html` in your browser
2. Click "Fill with Sample Data"
3. Click "Predict" to see results

## API Endpoints

### Single Prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, ..., "Amount": 149.62, "Time": 0.0
  }'
```

### Batch Prediction
```bash
curl -X POST http://127.0.0.1:5000/batch_predict \
  -F "file=@your_transactions.csv"
```

## Testing
```bash
# Run unit tests
python -m pytest tests/

# Test model performance
python tests/test_model.py
```

## CLI Helpers
```bash
# Single prediction
python scripts/predict_single.py --payload path/to/sample.json

# Batch predictions
python scripts/batch_predict.py --csv path/to/transactions.csv --out results.csv
```

## Deployment
```bash
# Deploy to AWS Lambda
./deploy_aws.sh
```

## Docker
```bash
docker build -t fraud-api .
docker run --rm -p 5000:5000 fraud-api
```

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure virtual environment is activated
2. **Model not found**: Run training notebook first
3. **CORS errors**: Install flask-cors package

### Support
- Check the full README.md for detailed documentation
- Review the Jupyter notebooks for examples
- Test with the web UI for quick validation
