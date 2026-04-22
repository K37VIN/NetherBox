import sys

import pandas as pd

from src.entity.estimator import AutoMLEstimator
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import load_dataset

DEFAULT_MODEL_PATH = "artifacts/models/automl_estimator.pkl"


class PredictionPipeline:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.estimator: AutoMLEstimator = None

    def _load_model(self):
        if self.estimator is None:
            logger.info(f"Loading model from {self.model_path}")
            self.estimator = AutoMLEstimator.load(self.model_path)

    def predict(self, data) -> list:
        """
        Run predictions on a CSV path, DataFrame, or dict.

        Returns a list of predictions.
        """
        try:
            self._load_model()

            if isinstance(data, str):
                df = load_dataset(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError(f"Unsupported input type: {type(data)}")

            preds = self.estimator.predict(df)
            logger.info(f"Predictions generated for {len(preds)} samples")
            return preds.tolist()

        except Exception as e:
            raise MyException(str(e), sys) from e
