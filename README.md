
MLOps Enhancement: Adult Income Prediction with Comprehensive Experiment Tracking
This project extends the MLOps-with-Docker-and-Jenkins repository by implementing comprehensive MLflow experiment tracking, automated model versioning, and production-ready deployment for the Adult Income dataset classification task.
What Was Implemented
Core Enhancement: Automated versioning and artifact logging with MLflow integration
The implementation adds enterprise-grade MLOps capabilities to the original baseline, transforming a simple training script into a production-ready machine learning system with full lifecycle management.
Key Components Added
1. Comprehensive Experiment Tracking

MLflow integration with remote tracking server
Parameter, metric, and artifact logging for every training run
Git commit hash and data version tracking for full reproducibility
Model registry with automated staging transitions (Staging → Production)

2. Enhanced Training Pipeline

Improved model convergence using StandardScaler preprocessing
Stratified cross-validation with detailed performance metrics
Robust error handling and status reporting
Automated model registration and versioning

3. Production Evaluation System

Comprehensive evaluation script with multiple metrics (accuracy, F1, precision, recall, ROC-AUC)
Automated confusion matrix and classification report generation
Statistical analysis and visualization artifacts
Separate evaluation experiment tracking

4. REST API Service

FastAPI-based inference service with Pydantic input validation
Health monitoring and status endpoints
Prometheus metrics collection for observability
Production-ready error handling and logging

5. CI/CD Automation

Complete Jenkins pipeline with quality gates
Automated testing and validation thresholds
Docker containerization and deployment
Artifact archival and management

Performance Results
The enhanced system achieves strong performance on the Adult Income dataset:

Test Accuracy: 84.6%
F1-Score: 66.4%
Precision: 72.3%
Recall: 61.5%
ROC-AUC: 90.2%

These results demonstrate the model's effectiveness while the MLflow integration ensures all experiments are tracked and reproducible.
Quick Start
Prerequisites

Python 3.9+
MLflow
Docker (optional)

Setup and Execution
bash# 1. Clone and navigate to repository
cd MLOps-with-Docker-and-Jenkins
source .venv/bin/activate

# 2. Install enhanced dependencies
pip install -r requirements.txt

# 3. Start MLflow tracking server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 --port 5001

# 4. Set environment variables (new terminal)
export PROCESSED_DATA_DIR="$PWD/processed_data"
export MODEL_DIR="$PWD/model_dir"
export RESULTS_DIR="$PWD/results_dir"
export RAW_DATA_DIR="$PWD"
export RAW_DATA_FILE="adult.csv"

mkdir -p "$PROCESSED_DATA_DIR" "$MODEL_DIR" "$RESULTS_DIR"

# 5. Execute ML pipeline
python preprocessing.py  # Data preparation
python train.py         # Model training with MLflow logging
python evaluate.py      # Comprehensive evaluation
python app.py           # Start API service
Access Points

MLflow UI: http://localhost:5001
API Documentation: http://localhost:8000/docs
Health Check: http://localhost:8000/health
Prometheus Metrics: http://localhost:8000/metrics

Architecture Overview
Raw Data → Preprocessing → Training (MLflow) → Evaluation → Model Registry
    ↓           ↓              ↓               ↓           ↓
adult.csv → train/test.csv → Model + Metrics → Validation → Staging/Production
                                                               ↓
                                                         FastAPI Service
                                                               ↓
                                                      Prometheus Metrics
API Usage Example
bash# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
  }'
Jenkins Pipeline
The included Jenkinsfile provides complete CI/CD automation:

Environment Setup: Python environment and dependency installation
Data Processing: Automated preprocessing with validation
Model Training: MLflow-tracked training with artifact logging
Evaluation: Performance assessment and metrics validation
Quality Gates: Automated threshold checking (accuracy > 80%, F1 > 60%)
Containerization: Docker image building and testing
Deployment: API service testing and validation

File Structure
├── preprocessing.py      # Original data processing
├── train.py             # Enhanced training with MLflow
├── evaluate.py          # Comprehensive evaluation (new)
├── app.py              # FastAPI inference service (new)
├── test.py             # Original test script
├── requirements.txt    # Updated dependencies
├── Jenkinsfile         # CI/CD pipeline (new)
├── README.md           # This documentation
├── REFLECTION.md       # AI assistant usage reflection
└── adult.csv           # Dataset
Key Features Demonstrated
Enterprise MLOps Practices

Experiment tracking and reproducibility
Model versioning and registry management
Automated quality gates and validation
Production monitoring and observability

Technical Implementation

Remote MLflow server with SQLite backend
Prometheus metrics integration
RESTful API with comprehensive validation
Docker containerization ready

Development Workflow

Git-based version control integration
Automated CI/CD with Jenkins
Quality assurance and testing
Documentation and reflection

Assumptions and Limitations
Current Implementation

Uses local SQLite backend for MLflow (production would require remote database and object storage)
Simplified categorical feature handling in API (production needs complete preprocessing pipeline matching training)
Single model deployment (could extend to A/B testing and model comparison frameworks)

Production Considerations

Requires MLflow server setup for team collaboration
API categorical encoding needs enhancement for robust inference
Monitoring and alerting would benefit from integration with enterprise observability platforms

Future Enhancements

Data drift detection and monitoring
Advanced model comparison and A/B testing
Integration with feature stores
Kubernetes deployment with auto-scaling
Enhanced security and authentication

