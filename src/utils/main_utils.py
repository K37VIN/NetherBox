import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.exception import MyException
from src.logger import logger

# ── File I/O ──────────────────────────────────────────────────────────────────


def read_yaml(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        raise MyException(f"Failed to read YAML: {path}", sys) from e


def write_yaml(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def save_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"JSON saved to {path}")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_object(path: str, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Object saved to {path}")


def load_object(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Dataset helpers ───────────────────────────────────────────────────────────


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    logger.info(f"Loaded dataset {path} — shape: {df.shape}")
    return df


def get_feature_target_split(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
