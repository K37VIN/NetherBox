import sys
import sqlite3
import numpy as np
from datetime import datetime


try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.constants import CLASSIFICATION
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import ModelEvaluationConfig
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import load_object, save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """Initializes the evaluation component with SQLite-backed configuration."""
        self.config = config

    def _log_to_sqlite(self, run_id: str, metrics: dict, trainer_artifact: ModelTrainerArtifact):
        """
        Writes custom pipeline execution metadata into a local SQLite table.
        This provides a lightweight alternative to MongoDB for audit trails.
        """
        try:
            
            db_name = self.config.mlflow_uri.split("///")[-1]
            conn = sqlite3.connect(db_name)
            cursor = conn.cursor()

            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pipeline_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT,
                    model_name TEXT,
                    accuracy REAL,
                    f1_weighted REAL,
                    status TEXT
                )
            ''')

            
            cursor.execute('''
                INSERT INTO pipeline_history (run_id, timestamp, model_name, accuracy, f1_weighted, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trainer_artifact.best_model_name,
                metrics.get("accuracy", 0.0),
                metrics.get("f1_weighted", 0.0),
                "SUCCESS"
            ))

            conn.commit()
            conn.close()
            logger.info("Custom pipeline metadata successfully saved to SQLite.")
        except Exception as e:
            logger.warning(f"Failed to write to SQLite metadata table: {e}")

    def _compute_metrics(self, y_true, y_pred, problem_type: str) -> dict:
        """Calculates evaluation metrics based on the problem type."""
        if problem_type == CLASSIFICATION:
            return {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4)
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            return {
                "r2": round(float(r2_score(y_true, y_pred)), 4),
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "rmse": round(float(mse ** 0.5), 4),
            }

    def initiate(
            self,
            transformation_artifact: DataTransformationArtifact,
            trainer_artifact: ModelTrainerArtifact,
            ingestion_artifact: DataIngestionArtifact,
    ) -> ModelEvaluationArtifact:
        """Executes the evaluation logic and logs results to MLflow and SQLite."""
        try:
            logger.info("=== Model Evaluation started ===")

            
            data_test = np.load(transformation_artifact.test_transformed_path, allow_pickle=True)
            X_test, y_test = data_test["X"], data_test["y"]
            model = load_object(trainer_artifact.best_model_path)

            
            y_pred = model.predict(X_test)
            metrics = self._compute_metrics(y_test, y_pred, trainer_artifact.problem_type)

           
            save_json(self.config.metrics_file_path, {
                "model": trainer_artifact.best_model_name,
                "problem_type": trainer_artifact.problem_type,
                "test_metrics": metrics,
                "all_cv_scores": trainer_artifact.all_scores,
            })

            for k, v in metrics.items():
                logger.info(f"  {k}: {v}")

            run_id = "LOCAL_RUN"
            if MLFLOW_AVAILABLE:
                
                mlflow.set_tracking_uri(self.config.mlflow_uri)
                mlflow.set_experiment(self.config.experiment_name)

                with mlflow.start_run(run_name=trainer_artifact.best_model_name) as run:
                    run_id = run.info.run_id
                    
                    
                    for model_name, score in trainer_artifact.all_scores.items():
                        clean_name = model_name.replace(" ", "_")
                        mlflow.log_metric(f"cv_{clean_name}", score)
                    
                    
                    mlflow.log_metrics(metrics)
                    mlflow.log_params({
                        "best_model": trainer_artifact.best_model_name,
                        "problem_type": trainer_artifact.problem_type,
                        "n_features": ingestion_artifact.n_features
                    })

                    mlflow.sklearn.log_model(sk_model=model, name="best_model")
                    mlflow.log_artifact(self.config.metrics_file_path)

                logger.info(f"MLflow run recorded. Run ID: {run_id}")
                
                
                self._log_to_sqlite(run_id, metrics, trainer_artifact)
            else:
                logger.warning("MLflow is unavailable. Results saved to JSON only.")

            logger.info("=== Model Evaluation complete ===")

            return ModelEvaluationArtifact(
                metrics=metrics,
                best_model_name=trainer_artifact.best_model_name,
                mlflow_run_id=run_id,
            )

        except Exception as e:
            raise MyException(str(e), sys) from e