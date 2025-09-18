#!/bin/bash

# AWS Lambda Deployment Script for Credit Card Fraud Detection

echo "Starting AWS Lambda deployment setup..."

# Create deployment directory
mkdir -p deployment
cd deployment

# Create a virtual environment for packaging
python -m venv lambda_env
source lambda_env/bin/activate

# Install dependencies
pip install -r ../requirements.txt --target ./package

# Copy application files
cp -r ../src ./package/
cp ../notebooks/flask_api.py ./package/

# Create Lambda handler
cat > ./package/lambda_function.py << 'EOF'
import json
from flask_api import app

def lambda_handler(event, context):
    """AWS Lambda handler function"""
    
    # Handle different event types (API Gateway, direct invoke, etc.)
    if 'httpMethod' in event:
        # API Gateway event
        from werkzeug.wrappers import Request
        from werkzeug.serving import WSGIRequestHandler
        
        # Convert API Gateway event to WSGI environ
        environ = {
            'REQUEST_METHOD': event['httpMethod'],
            'PATH_INFO': event['path'],
            'QUERY_STRING': event.get('queryStringParameters', '') or '',
            'CONTENT_TYPE': event.get('headers', {}).get('Content-Type', ''),
            'CONTENT_LENGTH': str(len(event.get('body', '') or '')),
            'wsgi.input': StringIO(event.get('body', '') or ''),
            'wsgi.version': (1, 0),
            'wsgi.url_scheme': 'https',
            'wsgi.errors': sys.stderr,
            'wsgi.multithread': False,
            'wsgi.multiprocess': True,
            'wsgi.run_once': False
        }
        
        # Add headers
        for key, value in event.get('headers', {}).items():
            key = key.upper().replace('-', '_')
            if key not in ('CONTENT_TYPE', 'CONTENT_LENGTH'):
                environ[f'HTTP_{key}'] = value
        
        # Create response
        response_data = {}
        
        def start_response(status, headers):
            response_data['statusCode'] = int(status.split()[0])
            response_data['headers'] = dict(headers)
        
        # Process with Flask app
        response = app(environ, start_response)
        response_data['body'] = ''.join(response)
        
        return response_data
    
    else:
        # Direct invoke event
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Fraud Detection API is running',
                'event': event
            })
        }
EOF

# Create deployment package
cd package
zip -r ../fraud-detection-lambda.zip .
cd ..

echo "Deployment package created: deployment/fraud-detection-lambda.zip"
echo "Upload this file to AWS Lambda with the following configuration:"
echo "- Runtime: Python 3.9+"
echo "- Handler: lambda_function.lambda_handler"
echo "- Memory: 512 MB (minimum recommended)"
echo "- Timeout: 30 seconds"

deactivate
