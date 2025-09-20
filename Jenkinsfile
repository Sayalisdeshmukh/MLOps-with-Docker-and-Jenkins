pipeline {
    agent any
    
    environment {
        PROCESSED_DATA_DIR = "${WORKSPACE}/processed_data"
        MODEL_DIR = "${WORKSPACE}/model_dir"
        RESULTS_DIR = "${WORKSPACE}/results_dir"
        RAW_DATA_DIR = "${WORKSPACE}"
        RAW_DATA_FILE = "adult.csv"
        MLFLOW_TRACKING_URI = "http://localhost:5001"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python environment...'
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
                
                echo 'Creating output directories...'
                sh '''
                    mkdir -p "${PROCESSED_DATA_DIR}"
                    mkdir -p "${MODEL_DIR}"
                    mkdir -p "${RESULTS_DIR}"
                '''
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                echo 'Running data preprocessing...'
                sh '''
                    . .venv/bin/activate
                    python preprocessing.py
                '''
                
                echo 'Verifying processed data...'
                sh '''
                    if [ ! -f "${PROCESSED_DATA_DIR}/train.csv" ] || [ ! -f "${PROCESSED_DATA_DIR}/test.csv" ]; then
                        echo "Error: Processed data files not found!"
                        exit 1
                    fi
                    echo "Processed data files created successfully"
                    wc -l "${PROCESSED_DATA_DIR}"/*.csv
                '''
            }
        }
        
        stage('Model Training') {
            steps {
                echo 'Training model with MLflow tracking...'
                sh '''
                    . .venv/bin/activate
                    python train.py
                '''
                
                echo 'Verifying model artifacts...'
                sh '''
                    if [ ! -f "${MODEL_DIR}/logit_model.joblib" ]; then
                        echo "Error: Model file not found!"
                        exit 1
                    fi
                    if [ ! -f "${RESULTS_DIR}/train_metadata.json" ]; then
                        echo "Error: Training metadata not found!"
                        exit 1
                    fi
                    echo "Model and metadata created successfully"
                    ls -la "${MODEL_DIR}/" "${RESULTS_DIR}/"
                '''
            }
        }
        
        stage('Model Evaluation') {
            steps {
                echo 'Evaluating model performance...'
                sh '''
                    . .venv/bin/activate
                    python evaluate.py
                '''
                
                echo 'Checking evaluation results...'
                sh '''
                    if [ ! -f "${RESULTS_DIR}/eval_metadata.json" ]; then
                        echo "Error: Evaluation metadata not found!"
                        exit 1
                    fi
                    echo "Evaluation completed successfully"
                    cat "${RESULTS_DIR}/eval_metadata.json"
                '''
            }
        }
        
        stage('Model Validation Gate') {
            steps {
                echo 'Validating model performance against thresholds...'
                sh '''
                    . .venv/bin/activate
                    python -c "
import json
import sys

# Load evaluation results
with open('${RESULTS_DIR}/eval_metadata.json', 'r') as f:
    results = json.load(f)

accuracy = results['test_accuracy']
f1_score = results['test_f1_score']

print(f'Model Performance:')
print(f'  Accuracy: {accuracy:.4f}')
print(f'  F1 Score: {f1_score:.4f}')

# Define thresholds
min_accuracy = 0.80
min_f1_score = 0.60

if accuracy < min_accuracy:
    print(f'FAIL: Accuracy {accuracy:.4f} below threshold {min_accuracy}')
    sys.exit(1)

if f1_score < min_f1_score:
    print(f'FAIL: F1 score {f1_score:.4f} below threshold {min_f1_score}')
    sys.exit(1)

print('PASS: Model meets performance thresholds')
"
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image for API service...'
                sh '''
                    # Create Dockerfile if it doesn't exist
                    cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model_dir/ ./model_dir/
COPY results_dir/ ./results_dir/

# Set environment variables
ENV MODEL_DIR=/app/model_dir
ENV RESULTS_DIR=/app/results_dir

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

                    # Build Docker image
                    docker build -t adult-income-api:${BUILD_NUMBER} .
                    docker tag adult-income-api:${BUILD_NUMBER} adult-income-api:latest
                '''
            }
        }
        
        stage('Test API') {
            steps {
                echo 'Testing API endpoints...'
                sh '''
                    # Start the API in background
                    . .venv/bin/activate
                    python -c "
import subprocess
import time
import requests
import json

# Start API server in background
api_process = subprocess.Popen(['python', 'app.py'])
time.sleep(10)  # Wait for server to start

try:
    # Test health endpoint
    response = requests.get('http://localhost:8000/health')
    print(f'Health check status: {response.status_code}')
    print(f'Health response: {response.json()}')
    
    # Test model info endpoint
    response = requests.get('http://localhost:8000/model/info')
    print(f'Model info status: {response.status_code}')
    print(f'Model info: {response.json()}')
    
    print('API tests passed!')
    
finally:
    # Clean up
    api_process.terminate()
    api_process.wait()
"
                '''
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up...'
            sh '''
                # Stop any running processes
                pkill -f "python.*app.py" || true
                pkill -f "uvicorn" || true
                
                # Archive artifacts
                if [ -d "mlruns" ]; then
                    tar -czf mlflow-artifacts.tar.gz mlruns/
                fi
            '''
            
            // Archive important artifacts
            archiveArtifacts artifacts: 'model_dir/*.joblib, results_dir/*.json, results_dir/*.png, mlflow-artifacts.tar.gz', 
                            allowEmptyArchive: true
        }
        
        success {
            echo 'Pipeline completed successfully!'
            echo 'Model has been trained, evaluated, and validated.'
            echo 'Docker image built and API tested.'
        }
