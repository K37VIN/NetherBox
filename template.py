import os
from pathlib import Path

project_name = "src"

list_of_files = [
    # ── Core package init files ───────────────────────────────────────────────
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_pusher.py",
    f"{project_name}/configuration/__init__.py",
    f"{project_name}/configuration/mongo_db_connection.py",
    f"{project_name}/configuration/aws_connection.py",
    f"{project_name}/cloud_storage/__init__.py",
    f"{project_name}/cloud_storage/aws_storage.py",
    f"{project_name}/data_access/__init__.py",
    f"{project_name}/data_access/proj1_data.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/entity/estimator.py",
    f"{project_name}/entity/s3_estimator.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipline/__init__.py",
    f"{project_name}/pipline/training_pipeline.py",
    f"{project_name}/pipline/prediction_pipeline.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/main_utils.py",
    # ── Root-level project files ──────────────────────────────────────────────
    "app.py",
    "demo.py",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    # ── Docker ────────────────────────────────────────────────────────────────
    "Dockerfile",
    ".dockerignore",
    # ── DVC pipeline definition ───────────────────────────────────────────────
    "dvc.yaml",
    # ── Config files ──────────────────────────────────────────────────────────
    "config/model.yaml",
    "config/schema.yaml",
    # ── GitHub Actions CI/CD ──────────────────────────────────────────────────
    ".github/workflows/ci_cd.yml",
    # ── Tests ─────────────────────────────────────────────────────────────────
    "tests/__init__.py",
    "tests/test_pipeline.py",
    # ── Notebooks (for EDA / experimentation) ─────────────────────────────────
    "notebooks/.gitkeep",
    # ── Artifact output directories (kept empty in git) ───────────────────────
    "artifacts/raw_data/.gitkeep",
    "artifacts/processed_data/.gitkeep",
    "artifacts/models/.gitkeep",
    "artifacts/reports/.gitkeep",
    # ── Git / project meta ────────────────────────────────────────────────────
    ".gitignore",
    "README.md",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create parent directories if they don't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # Create the file only if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # touch — content is added by the developer
        print(f"[CREATED]  {filepath}")
    else:
        print(f"[EXISTS]   {filepath}")
