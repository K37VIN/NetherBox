from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionConfig:
    raw_data_dir: str
    train_file_path: str
    test_file_path: str
    test_size: float
    random_state: int


@dataclass
class DataValidationConfig:
    schema_file_path: str
    report_file_path: str


@dataclass
class DataTransformationConfig:
    processed_data_dir: str
    preprocessor_file_path: str


@dataclass
class ModelTrainerConfig:
    model_dir: str
    best_model_path: str
    model_config_path: str


@dataclass
class ModelEvaluationConfig:
    metrics_file_path: str
    mlflow_uri: str = "sqlite:///automl.db"
    experiment_name: str = "AutoML_Experiment"
    metadata_db_path: str = "automl.db"


@dataclass
class TrainingPipelineConfig:
    dataset_path: str
    target_column: str
    problem_type: Optional[str] = None
