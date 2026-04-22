

import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.constants import (
    ARTIFACTS_DIR,
    BEST_MODEL_FILE,
    EXPERIMENT_NAME,
    METRICS_FILE,
    MODEL_CONFIG_FILE,
    MODEL_DIR,
    PREPROCESSOR_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    SCHEMA_FILE,
    TEST_FILE,
    TEST_SIZE,
    TRAIN_FILE,
    MLFLOW_TRACKING_URI,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from src.exception import MyException
from src.logger import logger


class TrainingPipeline:
    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        problem_type: str = None,  # None = auto-detect
    ):
        self.pipeline_config = TrainingPipelineConfig(
            dataset_path=dataset_path,
            target_column=target_column,
            problem_type=problem_type,
        )

    # ── Config builders ───────────────────────────────────────────────────────

    def _ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            raw_data_dir=RAW_DATA_DIR,
            train_file_path=os.path.join(RAW_DATA_DIR, TRAIN_FILE),
            test_file_path=os.path.join(RAW_DATA_DIR, TEST_FILE),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    def _validation_config(self) -> DataValidationConfig:
        return DataValidationConfig(
            schema_file_path=SCHEMA_FILE,
            report_file_path=os.path.join(REPORT_DIR, "validation_report.json"),
        )

    def _transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            processed_data_dir=PROCESSED_DATA_DIR,
            preprocessor_file_path=os.path.join(MODEL_DIR, PREPROCESSOR_FILE),
        )

    def _trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            model_dir=MODEL_DIR,
            best_model_path=os.path.join(MODEL_DIR, BEST_MODEL_FILE),
            model_config_path=MODEL_CONFIG_FILE,
        )

    def _evaluation_config(self) -> ModelEvaluationConfig:
        return ModelEvaluationConfig(
            metrics_file_path=os.path.join(REPORT_DIR, METRICS_FILE),
            mlflow_uri=MLFLOW_TRACKING_URI,
            experiment_name=EXPERIMENT_NAME,
        )

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def run(self):
        try:
            logger.info("╔══════════════════════════════════════╗")
            logger.info("║      AutoML Training Pipeline        ║")
            logger.info("╚══════════════════════════════════════╝")

            # 1. Ingest
            ingestion = DataIngestion(self._ingestion_config())
            ingestion_artifact = ingestion.initiate(
                dataset_path=self.pipeline_config.dataset_path,
                target_column=self.pipeline_config.target_column,
                problem_type_hint=self.pipeline_config.problem_type,
            )

            # 2. Validate
            validation = DataValidation(self._validation_config())
            validation_artifact = validation.initiate(ingestion_artifact)
            if not validation_artifact.is_valid:
                raise ValueError(f"Data validation failed: {validation_artifact.message}")

            # 3. Transform
            transformation = DataTransformation(self._transformation_config())
            transformation_artifact = transformation.initiate(ingestion_artifact)

            # 4. Train
            trainer = ModelTrainer(self._trainer_config())
            trainer_artifact = trainer.initiate(transformation_artifact, ingestion_artifact)

            # 5. Evaluate
            evaluation = ModelEvaluation(self._evaluation_config())
            evaluation_artifact = evaluation.initiate(
                transformation_artifact, trainer_artifact, ingestion_artifact
            )

            # 6. Push
            pusher = ModelPusher()
            final_model_path = pusher.initiate(
                trainer_artifact, transformation_artifact, evaluation_artifact
            )

            logger.info("╔══════════════════════════════════════╗")
            logger.info("║        Pipeline Complete! ✓          ║")
            logger.info(f"║  Best model : {trainer_artifact.best_model_name:<22}║")
            logger.info(f"║  Score      : {trainer_artifact.best_score:<22}║")
            logger.info(f"║  Type       : {trainer_artifact.problem_type:<22}║")
            logger.info("╚══════════════════════════════════════╝")

            return {
                "best_model": trainer_artifact.best_model_name,
                "best_score": trainer_artifact.best_score,
                "all_scores": trainer_artifact.all_scores,
                "problem_type": trainer_artifact.problem_type,
                "test_metrics": evaluation_artifact.metrics,
                "model_path": final_model_path,
                "mlflow_run_id": evaluation_artifact.mlflow_run_id,
            }

        except Exception as e:
            raise MyException(str(e), sys) from e
