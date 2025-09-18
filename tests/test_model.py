import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import pytest

def test_model_performance():
    """Test the trained model's performance metrics"""
    
    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'model', 'rf_model.joblib')
    model = joblib.load(model_path)
    
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Number of features: {model.n_features_in_}")
    
    # You would load test data here and evaluate
    # For now, we'll just verify the model loads correctly
    
    # No explicit return; assertions should be added when a test dataset is available

def test_model_prediction_format():
    """Test that model returns predictions in the expected format"""
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'model', 'rf_model.joblib')
    model = joblib.load(model_path)
    
    # Create sample data with correct shape and valid feature names
    # Use the model's own feature order when available
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
    else:
        # Fallback to an expected common order used in training
        cols = [
            "Time",
            "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
            "Amount"
        ]
    sample_df = pd.DataFrame([np.random.randn(len(cols))], columns=cols)
    
    prediction = model.predict(sample_df)
    probability = model.predict_proba(sample_df)
    
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
    assert probability.shape == (1, 2)
    assert np.sum(probability[0]) == pytest.approx(1.0, rel=1e-7)
    
    print("Model prediction format test passed!")

if __name__ == "__main__":
    test_model_performance()
    # test_model_prediction_format()  # Uncomment when pytest is available
