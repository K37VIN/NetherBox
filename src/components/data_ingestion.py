import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    CLASSIFICATION,
    CLASSIFICATION_UNIQUE_THRESHOLD,
    RANDOM_STATE,
    REGRESSION,
    TEST_SIZE,
)
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import ensure_dir, load_dataset


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    

    def detect_problem_type(self, y: pd.Series, user_hint: str = None) -> str:
        """
        Determine classification or regression automatically.
        A user-supplied hint ("classification" / "regression") always wins.
        """
        if user_hint and user_hint.lower() in (CLASSIFICATION, REGRESSION):
            logger.info(f"Problem type overridden by user: {user_hint}")
            return user_hint.lower()

        n_unique = y.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(y)

        if not is_numeric:
            logger.info("Target is non-numeric → classification")
            return CLASSIFICATION

        if n_unique <= CLASSIFICATION_UNIQUE_THRESHOLD:
            logger.info(
                f"Target has {n_unique} unique values (≤{CLASSIFICATION_UNIQUE_THRESHOLD}) → classification"
            )
            return CLASSIFICATION

        logger.info(
            f"Target has {n_unique} unique values and is numeric → regression"
        )
        return REGRESSION

    

    def initiate(self, dataset_path: str, target_column: str, problem_type_hint: str = None) -> DataIngestionArtifact:
        try:
            logger.info("=== Data Ingestion started ===")

            df = load_dataset(dataset_path)

            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )

            y = df[target_column]
            problem_type = self.detect_problem_type(y, problem_type_hint)

            
            train_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if problem_type == CLASSIFICATION else None,
            )

            ensure_dir(self.config.raw_data_dir)
            train_df.to_csv(self.config.train_file_path, index=False)
            test_df.to_csv(self.config.test_file_path, index=False)

            logger.info(
                f"Train: {train_df.shape}, Test: {test_df.shape} "
                f"| Problem type: {problem_type}"
            )
            logger.info("=== Data Ingestion complete ===")

            return DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
                problem_type=problem_type,
                target_column=target_column,
                n_features=df.shape[1] - 1,
                n_samples=len(df),
            )

        except Exception as e:
            raise MyException(str(e), sys) from e  