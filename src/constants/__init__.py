import os

# ── Artifact directories ──────────────────────────────────────────────────────
ARTIFACTS_DIR = "artifacts"
RAW_DATA_DIR = os.path.join(ARTIFACTS_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(ARTIFACTS_DIR, "processed_data")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
REPORT_DIR = os.path.join(ARTIFACTS_DIR, "reports")

# ── File names ────────────────────────────────────────────────────────────────
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
PREPROCESSOR_FILE = "preprocessor.pkl"
BEST_MODEL_FILE = "best_model.pkl"
METRICS_FILE = "metrics.json"
SCHEMA_FILE = "config/schema.yaml"
MODEL_CONFIG_FILE = "config/model.yaml"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "AutoML_Experiment"

# ── Train/test split ──────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Problem types ─────────────────────────────────────────────────────────────
CLASSIFICATION = "classification"
REGRESSION = "regression"

# ── Thresholds ────────────────────────────────────────────────────────────────
# If target has ≤ this many unique values → treat as classification
CLASSIFICATION_UNIQUE_THRESHOLD = 20
