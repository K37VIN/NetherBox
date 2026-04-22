from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    problem_type: str
    target_column: str
    n_features: int
    n_samples: int


@dataclass
class DataValidationArtifact:
    is_valid: bool
    report_file_path: str
    message: str


@dataclass
class DataTransformationArtifact:
    train_transformed_path: str
    test_transformed_path: str
    preprocessor_path: str


@dataclass
class ModelTrainerArtifact:
    best_model_path: str
    best_model_name: str
    best_score: float
    all_scores: Dict[str, float]
    problem_type: str


@dataclass
class ModelEvaluationArtifact:
    metrics: Dict[str, float]
    best_model_name: str
    mlflow_run_id: Optional[str]
