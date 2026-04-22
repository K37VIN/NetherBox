import sys
from typing import Dict

import numpy as np
import yaml
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.constants import CLASSIFICATION, RANDOM_STATE
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import ensure_dir, save_object


CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVC": SVC(probability=True, random_state=RANDOM_STATE),
}

REGRESSION_MODELS = {
    "Ridge": Ridge(),
    "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "SVR": SVR(),
}


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _load_model_registry(self, problem_type: str) -> dict:
        try:
            with open(self.config.model_config_path) as f:
                cfg = yaml.safe_load(f) or {}
            custom = cfg.get(problem_type, {})
            if custom:
                logger.info(f"Using model config from {self.config.model_config_path}")
                return custom
        except FileNotFoundError:
            pass

        models = CLASSIFICATION_MODELS if problem_type == CLASSIFICATION else REGRESSION_MODELS
        logger.info(f"Using default {problem_type} model registry ({len(models)} models)")
        return models

    
    def _cv_score(self, model, X: np.ndarray, y: np.ndarray, problem_type: str) -> float:
        scoring = "f1_weighted" if problem_type == CLASSIFICATION else "r2"

        n_samples = len(y)

        if problem_type == CLASSIFICATION:
            # Count samples per class
            unique, counts = np.unique(y, return_counts=True)
            min_class_count = counts.min()

            # CV cannot exceed smallest class size
            max_cv = min_class_count
        else:
            max_cv = n_samples

        # Use at most 5 folds, but adapt if dataset is small
        cv = min(5, max_cv)

        # Safety fallback
        if cv < 2:
            logger.warning("Very small dataset — using cv=2 fallback")
            cv = 2

        logger.info(f"Using cv={cv} for cross-validation")

        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return float(scores.mean())

    def initiate(
        self,
        transformation_artifact: DataTransformationArtifact,
        ingestion_artifact: DataIngestionArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logger.info("=== Model Training started ===")

            data_train = np.load(transformation_artifact.train_transformed_path, allow_pickle=True)
            X_train, y_train = data_train["X"], data_train["y"]

            problem_type = ingestion_artifact.problem_type
            model_registry = self._load_model_registry(problem_type)

            scores: Dict[str, float] = {}

            for name, model in model_registry.items():
                logger.info(f"  Evaluating [{name}] ...")
                score = self._cv_score(model, X_train, y_train, problem_type)
                scores[name] = round(score, 4)
                logger.info(f"  [{name}] CV score = {score:.4f}")

            best_name = max(scores, key=scores.get)
            best_score = scores[best_name]
            best_model = model_registry[best_name]

            # Refit best model on full training data
            best_model.fit(X_train, y_train)

            ensure_dir(self.config.model_dir)
            save_object(self.config.best_model_path, best_model)

            logger.info(f"Best model: {best_name} (CV score={best_score:.4f})")
            logger.info("=== Model Training complete ===")

            return ModelTrainerArtifact(
                best_model_path=self.config.best_model_path,
                best_model_name=best_name,
                best_score=best_score,
                all_scores=scores,
                problem_type=problem_type,
            )

        except Exception as e:
            raise MyException(str(e), sys) from e
