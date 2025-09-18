"""
Utility functions for the Credit Card Fraud Detection system
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_model(model_path=None):
    """Load the trained fraud detection model"""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'rf_model.joblib')
    
    return joblib.load(model_path)

def preprocess_transaction(transaction_data):
    """
    Preprocess transaction data for prediction
    
    Args:
        transaction_data (dict): Transaction features
    
    Returns:
        np.array: Preprocessed features ready for model input
    """
    required_features = [
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Time"
    ]
    
    # Ensure all required features are present
    for feature in required_features:
        if feature not in transaction_data:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Extract features in correct order
    features = [transaction_data[feature] for feature in required_features]
    
    return np.array([features])

def validate_prediction_response(prediction, probability):
    """
    Validate model prediction response
    
    Args:
        prediction: Model prediction (0 or 1)
        probability: Model probability output
    
    Returns:
        dict: Validated response
    """
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'fraud_detected': bool(prediction),
        'confidence': float(probability) if prediction else float(1 - probability)
    }

def log_prediction(transaction_data, prediction, probability):
    """
    Log prediction for monitoring and analysis
    
    Args:
        transaction_data (dict): Original transaction data
        prediction (int): Model prediction
        probability (float): Model probability
    """
    # In a production system, this would log to a database or monitoring service
    import datetime
    
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'amount': transaction_data.get('Amount', 0),
        'prediction': prediction,
        'probability': probability,
        'status': 'logged'
    }
    
    print(f"Prediction logged: {log_entry}")
    return log_entry
