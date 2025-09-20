import pandas as pd
import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from joblib import load

import mlflow
import mlflow.sklearn

"""
Enhanced evaluation script for the Adult Income dataset with MLflow integration.

This script loads the trained model and test data, computes comprehensive 
evaluation metrics, generates visualizations, and logs everything to MLflow.
"""

def plot_confusion_matrix(cm, classes, path):
    """Create and save a confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("adult-income-evaluation")

    # Load environment variables
    processed_data_dir = os.environ["PROCESSED_DATA_DIR"]
    model_dir = os.environ["MODEL_DIR"]
    results_dir = os.environ["RESULTS_DIR"]

    # Load model and test data
    print("Loading model and test data...")
    model_path = os.path.join(model_dir, "logit_model.joblib")
    test_data_path = os.path.join(processed_data_dir, "test.csv")
    
    model = load(model_path)
    df = pd.read_csv(test_data_path, sep=",")

    X_test = df.drop("income", axis=1)
    y_test = df["income"]

    print(f"Test data shape: {X_test.shape}")

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['≤50K', '>50K']
    
    # Save confusion matrix plot
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=class_names)
    report_path = os.path.join(results_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(class_report)

    # Capture git commit and data version
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        commit_hash = "unknown"
    
    data_version = "adult-income-v1"

    # Start MLflow run and log evaluation results
    print("Logging evaluation results to MLflow...")
    
    with mlflow.start_run() as run:
        print(f"MLflow evaluation run ID: {run.info.run_id}")
        
        # Log all metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_roc_auc", roc_auc)
        
        # Log parameters
        mlflow.log_param("test_data_size", len(y_test))
        mlflow.log_param("positive_class_ratio", y_test.mean())
        
        # Log artifacts
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="reports")
        
        # Set tags for reproducibility
        mlflow.set_tag("git_commit", commit_hash)
        mlflow.set_tag("data_version", data_version)
        mlflow.set_tag("evaluation_type", "final_test")

        # Save evaluation metadata
        eval_metadata = {
            "test_accuracy": accuracy,
            "test_f1_score": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_roc_auc": roc_auc,
            "test_data_size": len(y_test),
            "confusion_matrix": cm.tolist(),
            "class_names": class_names
        }
        
        eval_path = os.path.join(results_dir, "eval_metadata.json")
        with open(eval_path, "w") as outfile:
            json.dump(eval_metadata, outfile, indent=2)
        mlflow.log_artifact(eval_path, artifact_path="metadata")

    print("Evaluation complete! Results logged to MLflow.")
    print(f"View results at http://localhost:5001")

if __name__ == "__main__":
    main()
