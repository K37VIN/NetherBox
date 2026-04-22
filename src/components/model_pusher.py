import sys
from pathlib import Path

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.estimator import AutoMLEstimator
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import load_object

FINAL_MODEL_PATH = "artifacts/models/automl_estimator.pkl"


class ModelPusher:
    def __init__(self, final_model_path: str = FINAL_MODEL_PATH):
        self.final_model_path = final_model_path

    def initiate(
        self,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
        evaluation_artifact: ModelEvaluationArtifact,
    ) -> str:
        try:
            logger.info("=== Model Pusher started ===")

            preprocessor = load_object(transformation_artifact.preprocessor_path)
            model = load_object(trainer_artifact.best_model_path)

            estimator = AutoMLEstimator(
                preprocessor=preprocessor,
                model=model,
                problem_type=trainer_artifact.problem_type,
                model_name=trainer_artifact.best_model_name,
            )  
            estimator.save(self.final_model_path)

            logger.info(f"Production model saved to: {self.final_model_path}")
            logger.info("=== Model Pusher complete ===")

            return self.final_model_path

        except Exception as e:
            raise MyException(str(e), sys) from e
