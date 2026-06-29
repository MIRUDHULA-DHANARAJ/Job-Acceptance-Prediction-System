import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

def load_split_data(data_dir: str = "data/processed"):
    """Loads the pre-split and engineered datasets for training and validation."""
    print("📂 Loading engineered datasets...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    
    # Separate features (X) and target label (y)
    X_train = train.drop(columns=['status'])
    y_train = train['status']
    
    X_val = val.drop(columns=['status'])
    y_val = val['status']
    
    return X_train, y_train, X_val, y_val

def train_and_log_model(model_name: str, model, X_train, y_train, X_val, y_val):
    """
    Trains a model, calculates performance metrics, and logs everything to MLflow.
    """
    # Start an isolated MLflow tracking run block
    with mlflow.start_run(run_name=model_name):
        print(f"\n🚀 Training model: [{model_name}]...")
        
        # 1. Fit the model on the training data
        model.fit(X_train, y_train)
        
        # 2. Generate predictions on validation data
        preds = model.predict(X_val)
        # Class probabilities needed to calculate ROC-AUC score safely
        probs = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else preds
        
        # 3. Compute evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_val, preds),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds),
            "f1_score": f1_score(y_val, preds),
            "roc_auc": roc_auc_score(y_val, probs)
        }
        
        print(f"📊 Metrics -> F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # 4. Log parameters automatically from the scikit-learn/xgboost object
        params = model.get_params() if hasattr(model, "get_params") else {}
        # Clean up nested/heavy parameters to keep MLflow UI lightweight
        clean_params = {k: str(v) for k, v in params.items() if len(str(v)) < 100}
        mlflow.log_params(clean_params)
        
        # 5. Log metrics to the run
        mlflow.log_metrics(metrics)
        
        # 6. Log and save the actual model binary artifact framework specifically
        if "XGBoost" in model_name:
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
            
        return metrics["roc_auc"], model

def main():
    # Tell MLflow to store logging files locally inside a folder named 'mlruns'
    
    # Tells MLflow to store parameters and metrics inside a local SQLite database file
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Job_Placement_Prediction")
    # Tells MLflow to store parameters and metrics inside a local SQLite database file
    
    # Load our processed feature sets
    X_train, y_train, X_val, y_val = load_split_data()
    
    # Dictionary containing our candidate models with basic hyperparameters
    candidate_models = {
        "Logistic_Regression_Baseline": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost_Classifier": XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42)
    }
    
    best_score = 0.0
    best_model = None
    best_model_name = ""
    
    # Iterate through our models, train them, and track their performance
    for name, model in candidate_models.items():
        roc_auc, trained_model = train_and_log_model(name, model, X_train, y_train, X_val, y_val)
        
        # Check if this model outperformed our previous best model based on ROC-AUC
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = trained_model
            best_model_name = name
            
    print(f"\n🏆 Champion Model Selected: {best_model_name} (ROC-AUC: {best_score:.4f})")

if __name__ == "__main__":
    main()