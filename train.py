import pandas as pd
import json
import os
from joblib import dump
import subprocess
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# MLflow imports
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

"""
Enhanced training script for the Adult Income dataset with MLflow integration.

This script trains a logistic regression classifier on the processed training data
and includes comprehensive MLflow experiment tracking to log parameters, 
cross-validation metrics, model artifacts and reproducibility metadata.
"""

def main():
    # Set path to inputs
    PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
    train_data_file = "train.csv"
    train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

    # Read data
    print("Loading training data...")
    df = pd.read_csv(train_data_path, sep=",")

    # Split data into dependent and independent variables
    X_train = df.drop("income", axis=1)
    y_train = df["income"]

    print(f"Training data shape: {X_train.shape}")

    # -----------------------------------------------------------------------------
    # MLflow experiment configuration
    # -----------------------------------------------------------------------------
    # Set MLflow to log runs to the remote tracking server
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("adult-income-training")

    # Capture the current git commit for reproducibility
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit_hash = "unknown"

    # Specify the data version identifier
    data_version = "adult-income-v1"

    # -----------------------------------------------------------------------------
    # Model training and experiment logging
    # -----------------------------------------------------------------------------
    # Create a pipeline with scaling to help convergence
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Starting MLflow run...")
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")
        
        # Log model hyperparameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("cv_splits", 3)
        mlflow.log_param("scaling", "StandardScaler")
        mlflow.log_param("random_state", 42)

        # Fit the model
        print("Training model...")
        pipe.fit(X_train, y_train)

        # Cross-validation to estimate validation accuracy
        print("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
        val_accuracy = float(cv_scores.mean())
        val_std = float(cv_scores.std())

        print(f"Cross-validation accuracy: {val_accuracy:.4f} (+/- {val_std * 2:.4f})")

        # Log validation metrics
        mlflow.log_metric("cv_accuracy", val_accuracy)
        mlflow.log_metric("cv_accuracy_std", val_std)

        # Persist the model artifact
        MODEL_DIR = os.environ["MODEL_DIR"]
        model_name = "logit_model.joblib"
        model_path = os.path.join(MODEL_DIR, model_name)
        
        print(f"Saving model to {model_path}")
        dump(pipe, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        # Also use MLflow's built-in model logging
        mlflow.sklearn.log_model(pipe, "sklearn_model")

        # Persist and log validation metadata
        train_metadata = {
            "validation_acc": val_accuracy,
            "validation_std": val_std,
            "cv_splits": 3,
            "model_type": "LogisticRegression_with_StandardScaler"
        }
        
        RESULTS_DIR = os.environ["RESULTS_DIR"]
        results_path = os.path.join(RESULTS_DIR, "train_metadata.json")
        
        print(f"Saving metadata to {results_path}")
        with open(results_path, "w") as outfile:
            json.dump(train_metadata, outfile, indent=2)
        mlflow.log_artifact(results_path, artifact_path="metadata")

        # Set reproducibility tags
        mlflow.set_tag("git_commit", commit_hash)
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("environment", "development")

        print("MLflow logging complete!")

    # -----------------------------------------------------------------------------
    # Model Registry Integration
    # -----------------------------------------------------------------------------
    print("Registering model in MLflow Model Registry...")
    
    client = MlflowClient()
    model_uri = f"runs:/{run.info.run_id}/sklearn_model"
    
    # Create registered model if it doesn't exist
    try:
        client.create_registered_model("adult-income-model")
        print("Created new registered model: adult-income-model")
    except mlflow.exceptions.RestException:
        print("Using existing registered model: adult-income-model")

    # Create model version
    version_info = client.create_model_version(
        name="adult-income-model",
        source=model_uri,
        run_id=run.info.run_id,
        description=f"Logistic regression with StandardScaler. CV accuracy: {val_accuracy:.4f}"
    )

    print(f"Created model version: {version_info.version}")

    # Assign stage (Staging)
    client.transition_model_version_stage(
        name="adult-income-model",
        version=version_info.version,
        stage="Staging"
    )

    print(f"Model version {version_info.version} transitioned to Staging")
    print("Training complete! View results at http://localhost:5001")

if __name__ == "__main__":
    main()
