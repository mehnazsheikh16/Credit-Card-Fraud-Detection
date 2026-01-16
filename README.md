# Credit Card Fraud Detection

An end-to-end project that trains a machine learning model on the Kaggle credit card fraud dataset and serves real-time predictions through a Flask API. Includes single-transaction scoring, batch CSV scoring, and a simple browser-based UI for demos.

## Features

- Model training notebook (Random Forest with basic tuning)
- REST API with:
  - POST /predict — single JSON transaction
  - POST /batch_predict — CSV upload → CSV with predictions
  - GET /health — liveness probe
- Simple web UI (HTML/JS) for quick testing
- CORS enabled for browser calls
- Portable setup with requirements.txt (Docker optional)

## Dataset

Kaggle Credit Card Fraud Detection. Features V1–V28 are PCA-anonymized; Amount and Time are provided as-is. Due to class imbalance, the training notebook undersamples the majority class for a more useful recall on fraud.

Link: https://www.kaggle.com/mlg-ulb/creditcardfraud

## Quick Start

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Train the model (optional if a model file already exists)
- Open and run: notebooks/train_model.ipynb  
- Output: src/model/rf_model.joblib

3) Start the API
```bash
cd notebooks
python flask_api.py
# Server runs on http://127.0.0.1:5000 (or the PORT env var if set)
```

4) Health check
```bash
curl http://127.0.0.1:5000/health
```

5) Single prediction (example)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.53, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
    "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
    "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
    "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

6) Batch prediction (CSV)
```bash
curl -X POST -F "file=@test_transactions.csv" \
  http://127.0.0.1:5000/batch_predict -o batch_predictions.csv
```

7) Web UI
- With the API running, open notebooks/web_ui.html in a browser
- Use “Fill with Sample Data” to test quickly

## Project Structure

```
.
├── notebooks/
│   ├── train_model.ipynb      # Data prep, training, evaluation, save model
│   └── flask_api.py           # Flask API (run from here)
│   └── web_ui.html            # Simple browser UI
├── src/
│   └── model/
│       └── rf_model.joblib    # Saved model (created by the notebook)
├── requirements.txt
└── README.md
```

## Notes on Inputs

- Expected features (30): Time, V1–V28, Amount
- The API maps by feature names and enforces model order internally
- For batch CSV, headers must include all required feature names; order is auto-aligned

## Metrics

Evaluation metrics are shown in notebooks/train_model.ipynb after training. Results may vary by run due to sampling and random seeds.
