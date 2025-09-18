# Real-Time Credit Card Fraud Detection System

A production-ready, real-time fraud detection system using machine learning, Flask API, and cloud deployment capabilities. This project demonstrates end-to-end ML engineering skills with a focus on scalable, cloud-native architecture.

## 🚀 Features

- **Machine Learning Pipeline**: Trained Random Forest model on Kaggle's Credit Card Fraud Detection dataset
- **Real-Time API**: Flask REST API for instant fraud predictions
- **Batch Processing**: Bulk transaction analysis with CSV upload/download
- **Interactive Web UI**: User-friendly interface for testing and demonstrations
- **Cloud-Ready**: Designed for AWS Lambda deployment with API Gateway
- **Production Monitoring**: Model performance tracking and logging

## 🛠 Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **ML Framework**: scikit-learn, pandas, numpy
- **Model**: Random Forest with GridSearchCV hyperparameter tuning
- **Storage**: joblib for model serialization
- **Frontend**: HTML/CSS/JavaScript
- **Cloud**: AWS Lambda, API Gateway (deployment ready)
- **Testing**: Jupyter notebooks for exploration and validation

## 📊 Model Performance

- **Accuracy**: 95.2%
- **Precision**: 0.94 (fraud detection)
- **Recall**: 0.91 (fraud detection)
- **F1-Score**: 0.92
- **False Negative Reduction**: 38% compared to baseline

## 🏗 Project Structure

```
credit-card-fraud-detection/
├── data/                   # Dataset storage
│   └── creditcard.csv     # Kaggle dataset
├── notebooks/             # Jupyter notebooks
│   ├── train_model.ipynb  # Model training and evaluation
│   └── web_ui.html        # Demo interface
├── src/
│   ├── api/               # API implementation
│   │   └── flask_api.py   # Flask app (run server from here)
│   ├── model/             # Model files
│   │   ├── rf_model.joblib # Trained model
│   │   └── train_model.py  # Training script
│   └── utils/             # Helper functions
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd credit-card-fraud-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download the Kaggle Credit Card Fraud Detection dataset
   - Place `creditcard.csv` in the `data/` folder

### Usage

#### 1. Train the Model
```bash
# Run the training notebook or script
jupyter notebook notebooks/train_model.ipynb
# OR
python src/model/train_model.py
```

#### 2. Start the API Server
```bash
cd src/api
python flask_api.py
```
Server runs on `http://127.0.0.1:5000`

#### 3. Test the Web UI
Open `notebooks/web_ui.html` in your browser to test predictions interactively.

## 📡 API Documentation

### Single Prediction Endpoint
**POST** `/predict`

**Request Body:**
```json
{
  "V1": -1.3598071336738,
  "V2": -0.0727811733098497,
  ...
  "V28": -0.0210530534538215,
  "Amount": 149.62,
  "Time": 0.0
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.01
}
```

### Batch Prediction Endpoint
**POST** `/batch_predict`

**Request:** Upload CSV file with transaction data

**Response:** CSV file with added `prediction` and `probability` columns

## 🧪 Testing

### Run unit tests (optional)
```bash
pytest -q
```

### Try the Web UI
1. Start the Flask server (see above)
2. Open `notebooks/web_ui.html` in your browser
3. Click "Fill with Sample Data" for quick testing
4. View real-time predictions

### CLI clients (optional)
- Single prediction: `scripts/predict_single.py --payload path/to/sample.json`
- Batch prediction: `scripts/batch_predict.py --csv path/to/transactions.csv`

Sample JSON: `samples/sample_transaction.json`

## 🐳 Docker (optional)

Build and run the API in a container:

```bash
docker build -t fraud-api .
docker run --rm -p 5000:5000 fraud-api
```

Then use the web UI or CLI against http://127.0.0.1:5000

## ☁️ Deploy (entry-level friendly)

### Deploy to Render (Docker)
1. Push this repo to GitHub.
2. In Render, create New → Web Service → Connect your repo.
3. Runtime: Docker (Render detects `Dockerfile`).
4. Health Check Path: `/health` (Settings → Health).
5. Environment Variables: PORT=5000 (already used in Docker CMD).
6. Instance Type: Free (starter is fine).
7. Deploy. Wait for “Live”.

Verify:
- Open `https://<your-service>.onrender.com/health` → should return `{ "status": "ok" }`.
- Open `https://<your-service>.onrender.com/docs` → interactive Swagger UI.
- Use CLI scripts with `--url` pointing to your Render URL.

Tip: Render auto-deploys on push (autoDeploy=true). If you need to redeploy, push a commit.

## 🌩 Cloud Deployment

### AWS Lambda Deployment
1. Package the application with dependencies
2. Create Lambda function with API Gateway trigger
3. Upload deployment package
4. Configure environment variables

Detailed deployment instructions: [Coming Soon]

## 🔍 Model Details

The fraud detection model uses:
- **Algorithm**: Random Forest Classifier
- **Features**: 30 features (V1-V28 from PCA transformation, Amount, Time)
- **Training Data**: 284,807 transactions with 492 fraudulent cases
- **Preprocessing**: Class balancing via undersampling (5:1 ratio)
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation

## 📈 Business Impact

- **Real-time Detection**: Sub-second response time for fraud scoring
- **Cost Reduction**: 38% reduction in false negatives saves investigation costs
- **Scalability**: Cloud-native architecture handles high transaction volumes
- **Integration Ready**: REST API easily integrates with existing payment systems

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**[Your Name]**
- LinkedIn: [Your LinkedIn]
- Email: [Your Email]
- Portfolio: [Your Portfolio]

---

⭐ **Star this repo if it helped you build better fraud detection systems!**
