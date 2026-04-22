import sys
from pathlib import Path

import pandas as pd

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import MyException
from src.logger import logger
from src.utils.main_utils import ensure_dir, save_json


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _check_dataframe(self, df: pd.DataFrame, target_col: str) -> dict:
        """Run all checks and return a structured report dict."""
        report = {}
        issues = []

        report["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_report = {col: f"{pct}%" for col, pct in missing_pct.items() if pct > 0}
        report["missing_values"] = missing_report
        high_missing = [c for c, p in missing_pct.items() if p > 50]
        if high_missing:
            issues.append(f"Columns with >50% missing: {high_missing}")

        n_dupes = df.duplicated().sum()
        report["duplicate_rows"] = int(n_dupes)
        if n_dupes > 0:
            issues.append(f"{n_dupes} duplicate rows found (will be dropped)")

        if target_col in df.columns:
            target_missing = df[target_col].isnull().sum()
            report["target_missing"] = int(target_missing)
            if target_missing > 0:
                issues.append(f"Target column has {target_missing} missing values")
        else:
            issues.append(f"Target column '{target_col}' not found!")

        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        report["constant_columns"] = constant_cols
        if constant_cols:
            issues.append(f"Constant (zero-variance) columns: {constant_cols}")

        report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

        report["issues"] = issues
        report["is_valid"] = len([i for i in issues if "not found" in i]) == 0

        return report

    def initiate(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logger.info("=== Data Validation started ===")

            train_df = pd.read_csv(ingestion_artifact.train_file_path)
            test_df = pd.read_csv(ingestion_artifact.test_file_path)
            target = ingestion_artifact.target_column

            train_report = self._check_dataframe(train_df, target)
            test_report = self._check_dataframe(test_df, target)

            full_report = {
                "train": train_report,
                "test": test_report,
                "overall_valid": train_report["is_valid"] and test_report["is_valid"],
            }

            ensure_dir(str(Path(self.config.report_file_path).parent))
            save_json(self.config.report_file_path, full_report)

            all_issues = train_report["issues"] + test_report["issues"]
            is_valid = full_report["overall_valid"]

            if all_issues:
                for issue in all_issues:
                    logger.warning(f"Validation warning: {issue}")
            else:
                logger.info("No data issues found.")

            logger.info("=== Data Validation complete ===")

            return DataValidationArtifact(
                is_valid=is_valid,
                report_file_path=self.config.report_file_path,
                message="; ".join(all_issues) if all_issues else "All checks passed",
            )

        except Exception as e:
            raise MyException(str(e), sys) from e
