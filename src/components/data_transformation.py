import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import ensure_dir, save_object


HIGH_CARDINALITY_THRESHOLD = 15

class DataTransformation:
     
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.label_encoder = None


    def _split_columns(self,df:pd.DataFrame,target_col: str):
        feature_df = df.drop(columns=[target_col])
        num_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

        num_cols = [c for c in num_cols if feature_df[c].nunique() > 1]
        cat_cols = [c for c in cat_cols if feature_df[c].nunique() > 1]

        return num_cols, cat_cols
    
    def _build_preprocessor(self,num_cols: list,cat_cols: list) -> ColumnTransformer:
        numeric_pipeline = Pipeline([
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ])
        
        categorical_pipeline = Pipeline([
           ("imputer", SimpleImputer(strategy="most_frequent")), # <--- Added comma
           ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=HIGH_CARDINALITY_THRESHOLD,
         )),
      ])

        transformers = []
        if num_cols:
            transformers.append(("num", numeric_pipeline, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_pipeline, cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _encode_target(self, y_train: pd.Series, y_test: pd.Series, problem_type: str):
        if problem_type == "classification" and y_train.dtype == object:
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
            self.label_encoder = le
            logger.info(f"Target label-encoded. Classes: {list(le.classes_)}")
            return y_train_enc, y_test_enc
        return y_train.values, y_test.values
    def initiate(self, ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logger.info("=== Data Transformation started ===")

            train_df = pd.read_csv(ingestion_artifact.train_file_path).drop_duplicates()
            test_df = pd.read_csv(ingestion_artifact.test_file_path)
            target = ingestion_artifact.target_column

            # Drop rows where target is null
            train_df = train_df.dropna(subset=[target])
            test_df = test_df.dropna(subset=[target])

            num_cols, cat_cols = self._split_columns(train_df, target)
            logger.info(f"Numeric cols ({len(num_cols)}): {num_cols}")
            logger.info(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

            preprocessor = self._build_preprocessor(num_cols, cat_cols)

            X_train = train_df.drop(columns=[target])
            X_test = test_df.drop(columns=[target])
            y_train, y_test = self._encode_target(
                train_df[target], test_df[target], ingestion_artifact.problem_type
            )

            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            # Save transformed arrays + preprocessor
            import numpy as np

            ensure_dir(self.config.processed_data_dir)
            train_path = f"{self.config.processed_data_dir}/train.npz"
            test_path = f"{self.config.processed_data_dir}/test.npz"
            np.savez(train_path, X=X_train_t, y=y_train)
            np.savez(test_path, X=X_test_t, y=y_test)

            save_object(self.config.preprocessor_file_path, preprocessor)

            logger.info(
                f"Transformed shapes — train: {X_train_t.shape}, test: {X_test_t.shape}"
            )
            logger.info("=== Data Transformation complete ===")

            return DataTransformationArtifact(
                train_transformed_path=train_path,
                test_transformed_path=test_path,
                preprocessor_path=self.config.preprocessor_file_path,
            )

        except Exception as e:
            raise MyException(str(e), sys) from e
