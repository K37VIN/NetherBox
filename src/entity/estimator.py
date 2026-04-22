import pickle
from pathlib import Path
from src.exception import MyException
from src.logger import logger
import sys


class AutoMLEstimator:
    """Wraps the best trained model + preprocessor for end-to-end prediction."""

    def __init__(self, preprocessor, model, problem_type: str, model_name: str):
        self.preprocessor = preprocessor
        self.model = model
        self.problem_type = problem_type
        self.model_name = model_name

    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict(X_transformed)
        except Exception as e:
            raise MyException(str(e), sys) from e

    def predict_proba(self, X):
        """Only valid for classifiers that support probability estimates."""
        try:
            if not hasattr(self.model, "predict_proba"):
                raise ValueError(f"{self.model_name} does not support predict_proba")
            X_transformed = self.preprocessor.transform(X)
            return self.model.predict_proba(X_transformed)
        except Exception as e:
            raise MyException(str(e), sys) from e

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"AutoMLEstimator saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AutoMLEstimator":
        with open(path, "rb") as f:
            estimator = pickle.load(f)
        logger.info(f"AutoMLEstimator loaded from {path}")
        return estimator
   