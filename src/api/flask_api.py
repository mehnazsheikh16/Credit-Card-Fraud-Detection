
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd
from flask import Response

# Load the trained model with a robust path relative to this file
THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.normpath(os.path.join(THIS_DIR, '..', 'model', 'rf_model.joblib'))
model = joblib.load(MODEL_PATH)

app = Flask(__name__)
CORS(app)

@app.get('/health')
def health():
    """Liveness probe for the API server."""
    return jsonify({
        'status': 'ok',
        'model_loaded': bool(model),
        'model_path': os.path.relpath(MODEL_PATH, start=THIS_DIR)
    })

@app.get('/openapi.json')
def openapi_spec():
        spec_path = os.path.join(THIS_DIR, 'openapi.json')
        try:
                with open(spec_path, 'r') as f:
                        data = f.read()
                return Response(data, mimetype='application/json')
        except FileNotFoundError:
                return jsonify({'error': 'OpenAPI spec not found'}), 404

@app.get('/docs')
def docs():
        # Minimal Swagger UI embedding from CDN for simplicity
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>API Docs</title>
            <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css" />
        </head>
        <body>
            <div id="swagger-ui"></div>
            <script src="https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"></script>
            <script>
                window.onload = () => {
                    window.ui = SwaggerUIBundle({
                        url: '/openapi.json',
                        dom_id: '#swagger-ui',
                    });
                }
            </script>
        </body>
        </html>
        """
        return Response(html, mimetype='text/html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Enforce feature order and validate required keys
    required_features = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount"
    ]
    missing = [f for f in required_features if f not in data]
    if missing:
        return jsonify({
            'error': 'Missing required features',
            'missing': missing
        }), 400
    features = np.array([[float(data[f]) for f in required_features]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    return jsonify({
        'prediction': int(prediction),
        'probability': float(proba)
    })


# Batch prediction endpoint (must be after app = Flask(__name__))
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    df = pd.read_csv(file)
    # Ensure all required features are present
    required_features = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
        "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount"
    ]
    print('CSV columns:', df.columns.tolist())  # Debug print
    if not all(f in df.columns for f in required_features):
        return jsonify({'error': 'Missing required features in CSV'}), 400
    # Reorder columns to match model expectation
    df = df[required_features]
    X = df.astype(float)
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    df['prediction'] = preds
    df['probability'] = probas
    # Save to a new CSV in memory
    from io import StringIO
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='batch_predictions.csv'
    )

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', '5000'))
    app.run(host='0.0.0.0', debug=debug, port=port)
