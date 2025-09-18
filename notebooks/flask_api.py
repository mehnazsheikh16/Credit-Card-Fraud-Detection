from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import io

# Load the trained model (relative to repo root structure)
MODEL_PATH = '../src/model/rf_model.joblib'
model = joblib.load(MODEL_PATH)

# Expected feature order: prefer the model's trained order
DEFAULT_FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
FEATURES = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else DEFAULT_FEATURES

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True, 'expected_features': FEATURES})


def build_single_frame(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError('Payload must be a JSON object of feature_name: value')
    normalized = {str(k).strip(): v for k, v in payload.items()}
    missing = [f for f in FEATURES if f not in normalized]
    if missing:
        raise ValueError(f'Missing required features: {missing}')
    ordered = {f: normalized[f] for f in FEATURES}
    X = pd.DataFrame([ordered], columns=FEATURES).astype(float)
    return X


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=False)
        X = build_single_frame(data)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        return jsonify({'prediction': pred, 'probability': proba})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded. Use form field name "file".'}), 400
        file = request.files['file']
        df = pd.read_csv(file)

        # Normalize headers
        df.columns = [str(c).strip() for c in df.columns]

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            return jsonify({'error': f'Missing required features in CSV: {missing}'}), 400

        X = df.loc[:, FEATURES].astype(float)
        preds = model.predict(X)
        probas = model.predict_proba(X)[:, 1]

        out = df.copy()
        out['prediction'] = preds
        out['probability'] = probas

        # Stream CSV back
        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        mem = io.BytesIO(csv_buf.getvalue().encode('utf-8'))
        mem.seek(0)
        return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='batch_predictions.csv')
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
