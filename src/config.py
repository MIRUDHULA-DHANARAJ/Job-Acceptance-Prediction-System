"""
config.py

All the file paths and settings used across the other scripts, kept in
one place. If a path changes, you only need to change it here.
"""

import os

DATA_PROCESSED_DIR = "data/processed"

TRAIN_PATH = os.path.join(DATA_PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_PROCESSED_DIR, "test.csv")

LABEL_ENCODERS_PATH = os.path.join(DATA_PROCESSED_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(DATA_PROCESSED_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(DATA_PROCESSED_DIR, "feature_columns.pkl")

MODELS_DIR = "models"

TARGET_COLUMN = "placement_status"  # change if your real label column is named differently

# MLflow settings
# We use a local SQLite file instead of the older "just save files in a
# folder" method, because the newer MLflow Model Registry features (like
# the "production" alias we use in train.py) need a database, not files.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "job-acceptance-prediction"
MLFLOW_MODEL_NAME = "job-acceptance-classifier"

RANDOM_STATE = 42